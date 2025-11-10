import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# --- 核心依赖 ---
from model_HSV import CASTWithLUT
from utils import PairedImageDataset, collate_fn, sliding_window_inference


# =====================================================================================
# 1. 测试函数
# =====================================================================================
def test(args):
    """主测试流程：使用最佳模型，在测试集上评估并记录最终指标。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    model_path = checkpoint_dir / 'best_model.pth'
    log_file_path = checkpoint_dir / 'training_log.txt' # Will append to this log

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find 'best_model.pth' in checkpoint_dir: {checkpoint_dir}")

    print("Loading test dataset...")
    test_paths = sorted([p for p in Path(args.test_input_dir).rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    if not test_paths:
        print(f"No images found in test input directory: {args.test_input_dir}")
        return

    transform = transforms.ToTensor()
    test_dataset = PairedImageDataset(test_paths, args.test_input_dir, args.test_gt_dir, transform, get_path=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
                             collate_fn=collate_fn)

    cast_config = {
        'image_size': args.crop_size,
        'vocab_size': args.vocab_size,
        'embedding_dim': args.embedding_dim,
        'sequence_length': (args.crop_size // 16) ** 2,
        'commitment_cost': 0.25,
        'decay': 0.99
    }
    lut_config = {
        'vocab_size': args.vocab_size,
        'embedding_dim': args.embedding_dim,
        'num_luts': args.num_luts,
        'hidden_dim': 512,
        'lut_1d_size': 256
    }
    model = CASTWithLUT(cast_config, lut_config).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Model loaded successfully from {model_path}.")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        for input_img, gt_img, path_tuple in tqdm(test_loader, desc="Testing images"):
            if input_img is None: continue

            output_tensor = sliding_window_inference(model, input_img.to(device), args.crop_size, args.stride, device)

            psnr_metric.update(output_tensor, gt_img.to(device))
            ssim_metric.update(output_tensor, gt_img.to(device))

            relative_path = Path(path_tuple[0]).relative_to(args.test_input_dir)
            save_path = output_dir / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(output_tensor.cpu().clamp(0, 1), str(save_path.with_suffix('.png')))

    final_psnr = psnr_metric.compute()
    final_ssim = ssim_metric.compute()

    result_message = (
        f"\n--- FINAL TEST RESULTS ---\n"
        f"Model: {model_path}\n"
        f"Test Set: {args.test_input_dir}\n"
        f"Final Average PSNR: {final_psnr:.4f}\n"
        f"Final Average SSIM: {final_ssim:.4f}\n"
        f"---------------------------\n"
    )

    print(result_message)
    if log_file_path.exists():
        with open(log_file_path, 'a') as f:
            f.write(result_message)
            f.flush(); os.fsync(f.fileno())
        print(f"✅ Final results have been appended to {log_file_path}")
    else:
        print(f"Warning: Log file not found at {log_file_path}. Results only printed to console.")

    print(f"\n✨ Testing complete. Visual results saved to: {output_dir.resolve()}")


# =====================================================================================
# 2. 主程序入口
# =====================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test CAST-LUT Model")

    # Paths
    parser.add_argument('--test_input_dir', type=str, default='./data/test/input')
    parser.add_argument('--test_gt_dir', type=str, default='./data/test/gt')
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints", help="Directory where 'best_model.pth' is located.")
    parser.add_argument('--output_dir', type=str, default="./test_results", help="Directory to save final test results.")

    # Model Parameters
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_luts', type=int, default=16, help="Number of LUTs to fuse (N_L).")
    
    # Test Hyperparameters
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256, help="Crop size for sliding window inference.")
    parser.add_argument('--stride', type=int, default=128, help="Stride for sliding window inference.")

    args = parser.parse_args()

    print("🚀 Starting Test Process...")
    test(args)
