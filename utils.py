import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.models as models
import kornia
import numpy as np

# =====================================================================================
# 1. Loss Functions
# =====================================================================================

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.blocks = nn.Sequential(*list(vgg.features)[:23]).eval()
        for param in self.blocks.parameters():
            param.requires_grad = False
        self.transform = nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.resize = resize

    def forward(self, input, target):
        self.mean = self.mean.to(input.device)
        self.std = self.std.to(input.device)
        self.blocks = self.blocks.to(input.device)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        return F.l1_loss(self.blocks(input), self.blocks(target))


class PurpleFringingLossV2(nn.Module):
    def __init__(self, penalty_weight: float = 10.0, edge_threshold: float = 0.2,
                 purple_hue_range: tuple = (260.0, 340.0)):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.edge_threshold = edge_threshold
        self.min_hue, self.max_hue = purple_hue_range
        self.base_loss = nn.L1Loss(reduction='none')

    def forward(self, output_img: torch.Tensor, gt_img: torch.Tensor) -> torch.Tensor:
        device = output_img.device
        gt_img = gt_img.to(device)
        pixelwise_error = self.base_loss(output_img, gt_img)
        gt_grayscale = kornia.color.rgb_to_grayscale(gt_img)
        edge_map = kornia.filters.sobel(gt_grayscale)
        edge_mask = (edge_map > self.edge_threshold).float()
        output_hsv = kornia.color.rgb_to_hsv(output_img)
        hue, saturation = output_hsv[:, 0:1, :, :], output_hsv[:, 1:2, :, :]
        is_purple_hue = ((hue >= self.min_hue) & (hue <= self.max_hue)).float()
        purple_saturation_weight = is_purple_hue * saturation
        fringing_mask = edge_mask * purple_saturation_weight
        penalty_map = 1.0 + (self.penalty_weight - 1.0) * fringing_mask
        penalized_loss = pixelwise_error * penalty_map.detach()
        return torch.mean(penalized_loss)

# =====================================================================================
# 2. Dataset Class
# =====================================================================================

class PairedImageDataset(Dataset):
    def __init__(self, image_paths, root_dir, gt_dir, transform=None, get_path=False):
        self.image_paths = image_paths
        self.root_dir = Path(root_dir)
        self.gt_dir = Path(gt_dir)
        self.transform = transform
        self.get_path = get_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        input_path = self.image_paths[idx]
        try:
            relative_path = Path(input_path).relative_to(self.root_dir)
            gt_path = self.gt_dir / relative_path
            if not gt_path.exists(): return None

            input_image = Image.open(input_path).convert('RGB')
            gt_image = Image.open(gt_path).convert('RGB')

            if self.transform:
                input_image_t = self.transform(input_image)
                gt_image_t = self.transform(gt_image)

            if self.get_path:
                return input_image_t, gt_image_t, str(input_path)
            else:
                return input_image_t, gt_image_t
        except Exception as e:
            print(f"Error loading image pair: {input_path}, {e}")
            return None


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return (None,) * 3

    num_items = len(batch[0])
    if num_items == 3:
        inputs, gts, paths = zip(*batch)
        return torch.stack(inputs), torch.stack(gts), paths
    else:
        inputs, gts = zip(*batch)
        return torch.stack(inputs), torch.stack(gts)

# =====================================================================================
# 3. Inference Helper
# =====================================================================================

def sliding_window_inference(model, full_image_tensor, crop_size, stride, device):
    """
    Performs sliding window inference on high-resolution images to avoid OOM issues,
    using weighted averaging to smooth overlapping regions.
    """
    model.eval()
    b, c, h, w = full_image_tensor.shape

    if h < crop_size or w < crop_size:
        h_pad = max(0, crop_size - h)
        w_pad = max(0, crop_size - w)
        padded_image = F.pad(full_image_tensor, (0, w_pad, 0, h_pad), 'reflect')
        with torch.no_grad():
            output_dict = model(padded_image)
            return output_dict['lut_output'][:, :, :h, :w]

    sum_map = torch.zeros_like(full_image_tensor, dtype=torch.float32, device=device)
    weight_map = torch.zeros_like(full_image_tensor, dtype=torch.float32, device=device)

    weight_patch = torch.ones((1, 1, crop_size, crop_size), device=device)

    y_steps = int(np.ceil((h - crop_size) / stride)) + 1 if h > crop_size else 1
    x_steps = int(np.ceil((w - crop_size) / stride)) + 1 if w > crop_size else 1

    for i in range(y_steps):
        for j in range(x_steps):
            y_start = i * stride
            x_start = j * stride

            if y_start + crop_size > h:
                y_start = h - crop_size
            if x_start + crop_size > w:
                x_start = w - crop_size

            y_end = y_start + crop_size
            x_end = x_start + crop_size

            patch = full_image_tensor[:, :, y_start:y_end, x_start:x_end]
            
            with torch.no_grad():
                output_dict = model(patch)
                output_patch = output_dict['lut_output']

            sum_map[:, :, y_start:y_end, x_start:x_end] += output_patch
            weight_map[:, :, y_start:y_end, x_start:x_end] += weight_patch

    output_tensor = sum_map / torch.clamp(weight_map, min=1e-8)

    return output_tensor
