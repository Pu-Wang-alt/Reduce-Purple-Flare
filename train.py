import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# --- 核心依赖 ---
from model_HSV import CASTWithLUT
from utils import VGGPerceptualLoss, PurpleFringingLossV2, PairedImageDataset, collate_fn


# =====================================================================================
# 1. 评估与日志函数
# =====================================================================================
def evaluate_and_log(model, dataloader, device, output_dir, log_file, epoch, current_val_loss):
    """
    在验证集上评估模型，记录指标，并保存可视化结果。
    此版本会保留原始数据集的子文件夹结构。
    """
    print(f"\n--- Running evaluation for Epoch {epoch} ---")
    model.eval()

    eval_output_path = Path(output_dir)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    val_root_dir = dataloader.dataset.root_dir

    with torch.no_grad():
        for input_img, gt_img, path_tuple in tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", leave=False):
            if input_img is None: continue

            input_img_b, gt_img_b = input_img.to(device), gt_img.to(device)
            output_dict = model(input_img_b)
            output_img_b = output_dict['lut_output'].clamp(0, 1)

            psnr_metric.update(output_img_b, gt_img_b)
            ssim_metric.update(output_img_b, gt_img_b)

            input_path = Path(path_tuple[0])
            relative_path = input_path.relative_to(val_root_dir)

            restored_save_path = eval_output_path / "restored_latest" / relative_path
            comparison_save_path = eval_output_path / "comparison_latest" / relative_path

            restored_save_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_save_path.parent.mkdir(parents=True, exist_ok=True)

            save_image(output_img_b.squeeze(0), restored_save_path.with_suffix('.png'))

            comparison_grid = torch.cat([input_img.squeeze(0), output_img_b.cpu().squeeze(0), gt_img.squeeze(0)], dim=2)
            save_image(comparison_grid, comparison_save_path.with_suffix('.png'))

    epoch_psnr = psnr_metric.compute()
    epoch_ssim = ssim_metric.compute()

    log_message = (f"Epoch: {epoch:03d} | Val_L1: {current_val_loss:.6f} | "
                   f"PSNR: {epoch_psnr:.4f} | SSIM: {epoch_ssim:.4f}\n")

    print(log_message, end='')
    with open(log_file, 'a') as f:
        f.write(log_message)
        f.flush()
        os.fsync(f.fileno())

    model.train()


# =====================================================================================
# 2. 训练函数
# =====================================================================================
def train(args):
    """主训练流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_msg = f"Using device: {device}\n"
    print(device_msg, end='')

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.eval_output_dir).mkdir(parents=True, exist_ok=True)

    log_file_path = Path(args.checkpoint_dir) / "training_log.txt"
    with open(log_file_path, 'w') as f:
        f.write("--- Training Log ---\n")
        f.flush(); os.fsync(f.fileno())
        f.write(device_msg)
        f.flush(); os.fsync(f.fileno())
        f.write(f"Args: {vars(args)}\n\n")
        f.flush(); os.fsync(f.fileno())

    print("Loading datasets...")
    train_paths = sorted(
        [p for p in Path(args.train_input_dir).rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    val_paths = sorted(
        [p for p in Path(args.val_input_dir).rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])

    train_dataset = PairedImageDataset(train_paths, args.train_input_dir, args.train_gt_dir, transform)
    val_dataset = PairedImageDataset(val_paths, args.val_input_dir, args.val_gt_dir, val_transform, get_path=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, collate_fn=collate_fn)
    val_loader_for_loss = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    eval_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
                             collate_fn=collate_fn)

    loaded_msg = f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.\n"
    print(loaded_msg, end='')
    with open(log_file_path, 'a') as f:
        f.write(loaded_msg)

    cast_config = {
        'image_size': args.image_size, 'vocab_size': args.vocab_size, 'embedding_dim': args.embedding_dim,
        'sequence_length': (args.image_size // 16) ** 2, 'commitment_cost': 0.25, 'decay': 0.99
    }
    lut_config = {
        'vocab_size': args.vocab_size,
        'embedding_dim': args.embedding_dim,
        'num_luts': args.num_luts,
        'hidden_dim': 512,
        'lut_1d_size': 256
    }
    model = CASTWithLUT(cast_config, lut_config).to(device)

    if os.path.exists(args.pretrained_cast_path):
        pretrained_msg = f"Loading pretrained CAST weights from {args.pretrained_cast_path}...\n"
        print(pretrained_msg, end='')
        with open(log_file_path, 'a') as f:
            f.write(pretrained_msg)

        cast_state_dict = torch.load(args.pretrained_cast_path, map_location=device)
        if 'model_state_dict' in cast_state_dict:
            cast_state_dict = cast_state_dict['model_state_dict']
        model.cast.load_state_dict(cast_state_dict)
        for param in model.cast.parameters():
            param.requires_grad = False

        frozen_msg = "✅ CAST weights loaded and frozen.\n"
        print(frozen_msg, end='')
        with open(log_file_path, 'a') as f:
            f.write(frozen_msg)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    criterion_l1 = nn.L1Loss().to(device)
    criterion_fringe = PurpleFringingLossV2(penalty_weight=args.fringe_penalty_weight).to(device)
    criterion_perc = VGGPerceptualLoss(resize=False).to(device)

    epochs_no_improve = 0
    best_val_loss = float('inf')

    start_msg = "\n--- Starting Training ---\n"
    print(start_msg, end='')
    with open(log_file_path, 'a') as f:
        f.write(start_msg)

    for epoch in range(1, args.epochs + 1):
        epoch_header = f"\n--- Epoch {epoch}/{args.epochs} ---\n"
        print(epoch_header, end='')
        with open(log_file_path, 'a') as f:
            f.write(epoch_header)
            f.flush(); os.fsync(f.fileno())

        model.train()
        epoch_total_loss, epoch_loss_l1, epoch_loss_fringe, epoch_loss_perc, epoch_loss_quant = 0.0, 0.0, 0.0, 0.0, 0.0
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")

        for input_imgs, gt_imgs in train_pbar:
            if input_imgs is None: continue
            input_imgs, gt_imgs = input_imgs.to(device), gt_imgs.to(device)
            optimizer.zero_grad()
            output_dict = model(input_imgs)
            output_imgs = output_dict['lut_output']
            loss_l1 = criterion_l1(output_imgs, gt_imgs)
            loss_fringe = criterion_fringe(output_imgs, gt_imgs)
            loss_perc = criterion_perc(output_imgs, gt_imgs)
            quant_loss = output_dict['quantization_loss']
            total_loss = (args.lambda_l1 * loss_l1 + args.lambda_fringe * loss_fringe +
                          args.lambda_perc * loss_perc + args.lambda_quant * quant_loss)
            total_loss.backward()
            optimizer.step()
            epoch_total_loss += total_loss.item()
            epoch_loss_l1 += loss_l1.item()
            epoch_loss_fringe += loss_fringe.item()
            epoch_loss_perc += loss_perc.item()
            epoch_loss_quant += quant_loss.item()
            train_pbar.set_postfix(
                Total=f"{total_loss.item():.4f}", L1=f"{loss_l1.item():.4f}",
                Fringe=f"{loss_fringe.item():.4f}", Perc=f"{loss_perc.item():.4f}",
                Quant=f"{quant_loss.item():.4f}")

        num_batches = len(train_loader)
        if num_batches > 0:
            avg_losses_msg = (f"Avg Train Losses -> Total: {epoch_total_loss / num_batches:.4f} | "
                              f"L1: {epoch_loss_l1 / num_batches:.4f} | "
                              f"Fringe: {epoch_loss_fringe / num_batches:.4f} | "
                              f"Perceptual: {epoch_loss_perc / num_batches:.4f} | "
                              f"Quant: {epoch_loss_quant / num_batches:.4f}\n")
            print(avg_losses_msg, end='')
            with open(log_file_path, 'a') as f:
                f.write(avg_losses_msg)
                f.flush(); os.fsync(f.fileno())

        model.eval()
        val_loss_l1 = 0.0
        with torch.no_grad():
            for input_imgs, gt_imgs, _ in tqdm(val_loader_for_loss, desc="Validating Loss"):
                if input_imgs is None: continue
                input_imgs, gt_imgs = input_imgs.to(device), gt_imgs.to(device)
                output_dict = model(input_imgs)
                val_loss_l1 += criterion_l1(output_dict['lut_output'], gt_imgs).item()
        avg_val_l1 = val_loss_l1 / len(val_loader_for_loss) if len(val_loader_for_loss) > 0 else 0.0

        evaluate_and_log(
            model=model, dataloader=eval_loader, device=device,
            output_dir=args.eval_output_dir, log_file=log_file_path,
            epoch=epoch, current_val_loss=avg_val_l1)

        if avg_val_l1 < best_val_loss:
            best_val_loss = avg_val_l1
            epochs_no_improve = 0
            best_model_path = Path(args.checkpoint_dir) / 'best_model.pth'
            torch.save(model.state_dict(), best_model_path)

            save_msg = f"🎉 New best model saved to {best_model_path} (Val L1: {best_val_loss:.6f})\n"
            print(save_msg, end='')
            with open(log_file_path, 'a') as f:
                f.write(save_msg)
                f.flush(); os.fsync(f.fileno())
        else:
            epochs_no_improve += 1
            patience_msg = f"Validation loss did not improve for {epochs_no_improve}/{args.patience} epochs.\n"
            print(patience_msg, end='')
            with open(log_file_path, 'a') as f:
                f.write(patience_msg)
                f.flush(); os.fsync(f.fileno())

        epoch_model_path = Path(args.checkpoint_dir) / f'epoch_{epoch}.pth'
        torch.save(model.state_dict(), epoch_model_path)

        if epochs_no_improve >= args.patience:
            stop_msg = f"\nEarly stopping triggered after {args.patience} epochs with no improvement.\n"
            print(stop_msg, end='')
            with open(log_file_path, 'a') as f:
                f.write(stop_msg)
                f.flush(); os.fsync(f.fileno())
            break
            
    print("\n✅ Training Finished.")


# =====================================================================================
# 3. 主程序入口
# =====================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CAST-LUT Model")

    # Paths
    parser.add_argument('--train_input_dir', type=str, default='./data/train/input')
    parser.add_argument('--train_gt_dir', type=str, default='./data/train/gt')
    parser.add_argument('--val_input_dir', type=str, default='./data/val/input')
    parser.add_argument('--val_gt_dir', type=str, default='./data/val/gt')
    parser.add_argument('--pretrained_cast_path', type=str, default='./pretrained/cast_hsv_best.pth')
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints")
    parser.add_argument('--eval_output_dir', type=str, default="./eval_results")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_JULAG, '--num_workers', type=int, default=4)
    parser.add_argument('--num_luts', type=int, default=16, help="Number of LUTs to fuse (N_L).")

    # Loss Weights
    parser.add_argument('--lambda-l1', type=float, default=1.0)
    parser.add_argument('--lambda-perc', type=float, default=0.1)
    parser.add_argument('--lambda-fringe', type=float, default=2.0)
    parser.add_argument('--lambda-quant', type=float, default=0.1)
    parser.add_argument('--fringe-penalty-weight', type=float, default=15.0)

    # Model Parameters
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--embedding_dim', type=int, default=128)

    args = parser.parse_args()
    
    print("🚀 Starting Training Process...")
    train(args)
