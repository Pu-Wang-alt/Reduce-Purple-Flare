import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import os
from tqdm import tqdm
import datetime

# Mixed precision training support
from torch.cuda.amp import autocast, GradScaler

from model_HSV import CAST


def validate(model, dataloader, criterion, device, use_mixed_precision, current_beta):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        for i, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_mixed_precision):
                outputs = model(images)
                recon_image = outputs["recon_image"]
                quantization_loss = outputs["quantization_loss"]
                recon_loss = criterion(recon_image, images)
                total_loss = recon_loss + current_beta * quantization_loss
            val_loss += total_loss.item()
            progress_bar.set_postfix({'avg_val_loss': val_loss / (i + 1)})
    return val_loss / len(dataloader)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_mixed_precision = args.use_mixed_precision and torch.cuda.is_available()
    scaler = GradScaler() if use_mixed_precision else None
    if use_mixed_precision:
        print("Mixed precision training enabled")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])

    print("Loading training set...")
    train_dataset = datasets.ImageFolder(root=args.train_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    print(f"Training set loaded: {len(train_dataset)} images.")

    print("Loading validation set...")
    val_dataset = datasets.ImageFolder(root=args.val_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    print(f"Validation set loaded: {len(val_dataset)} images.")

    if args.gradient_accumulation_steps > 1:
        print(
            f"Gradient accumulation enabled: steps={args.gradient_accumulation_steps}, effective batch size={args.batch_size * args.gradient_accumulation_steps}")

    model = CAST(
        image_size=args.image_size,
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        sequence_length=(args.image_size // 16) ** 2,
        commitment_cost=args.commitment_cost,
        decay=args.decay
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - args.lr_warmup_steps)

    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "cast_hsv_best.pth")
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, "cast_hsv_latest.pth")

    start_epoch = 0
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"Resuming from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if use_mixed_precision and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"Successfully loaded. Resuming from Epoch {start_epoch + 1}. Best val loss: {best_val_loss:.4f}")
        else:
            print(f"Error: Checkpoint file not found: {args.resume_from}")
            return

    recon_criterion = nn.L1Loss()
    print("Using L1 Loss (Mean Absolute Error) for reconstruction.")

    global_step = start_epoch * (len(train_loader) // args.gradient_accumulation_steps)
    for epoch in range(start_epoch, args.epochs):
        if epoch < args.beta_warmup_epochs:
            current_beta = args.beta * max(0.0, float(epoch)) / float(args.beta_warmup_epochs)
        else:
            current_beta = args.beta
        print(f"--- Epoch {epoch + 1}/{args.epochs} | Current Beta: {current_beta:.4f} ---")

        model.train()
        train_total_loss = 0.0
        optimizer.zero_grad()
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Training]")

        for batch_idx, (images, _) in enumerate(progress_bar_train):
            if global_step < args.lr_warmup_steps:
                lr_scale = min(1.0, float(global_step + 1) / float(args.lr_warmup_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.learning_rate * lr_scale

            images = images.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_mixed_precision):
                outputs = model(images)
                recon_image = outputs["recon_image"]
                quantization_loss = outputs["quantization_loss"]
                recon_loss = recon_criterion(recon_image, images)
                total_loss = recon_loss + current_beta * quantization_loss
                total_loss_scaled = total_loss / args.gradient_accumulation_steps

            if use_mixed_precision:
                scaler.scale(total_loss_scaled).backward()
            else:
                total_loss_scaled.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                if use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

                if global_step >= args.lr_warmup_steps:
                    lr_scheduler.step()

                global_step += 1

            train_total_loss += total_loss.item()
            progress_bar_train.set_postfix({'loss': f'{total_loss.item():.4f}'})

        avg_train_loss = train_total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Training Summary: Avg Total Loss: {avg_train_loss:.4f}")

        avg_val_loss = validate(model, val_loader, recon_criterion, device, use_mixed_precision, current_beta)
        print(f"Epoch {epoch + 1} Validation Summary: Avg Total Loss: {avg_val_loss:.4f}")

        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
        }
        if use_mixed_precision:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()

        torch.save(checkpoint_data, latest_checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data['best_val_loss'] = best_val_loss
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"🎉 New best model saved. Validation loss: {best_val_loss:.4f}")

    print("Training finished.")


def main():
    parser = argparse.ArgumentParser(description='Train CAST (HSV) Model')

    # --- Path Arguments ---
    parser.add_argument('--train_path', type=str, default='./data/imagenet/train', help='Root directory for training set')
    parser.add_argument('--val_path', type=str, default='./data/imagenet/val', help='Root directory for validation set')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint file to resume training from')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_cast_hsv', help='Directory to save model checkpoints')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=100, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Optimizer learning rate')
    parser.add_argument('--lr_warmup_steps', type=int, default=1000, help='Number of learning rate warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW optimizer')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Enable mixed precision training (AMP)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker threads for data loading')

    # --- Model Hyperparameters ---
    parser.add_argument('--image_size', type=int, default=224, help='Image size (square)')
    parser.add_argument('--vocab_size', type=int, default=4096, help='Codebook vocabulary size')
    parser.addGargument('--embedding_dim', type=int, default=128, help='Codebook embedding dimension')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='Commitment cost for VQ-VAE')
    parser.add_argument('--decay', type=float, default=0.99, help='Decay factor for EMA updates in VectorQuantizer')

    # --- Loss Weight Parameters ---
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for quantization loss in total loss')
    parser.add_argument('--beta_warmup_epochs', type=int, default=10, help='Number of epochs for beta warmup')

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("--- Starting CAST (HSV) Training ---")
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 30)

    train(args)


if __name__ == "__main__":
    main()
