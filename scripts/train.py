"""
train.py - Training Script for PhysRetinexNet
Trains the 44.8K parameter physics-guided network on low-light paired datasets (e.g. LOL-v2 Real / LOL-v1).

Usage:
  python scripts/train.py --train_low_dir Dataset/LOL-v2/Train/Low --epochs 50 --batch_size 8 --lr 5e-4
"""

import os
import sys
import time
import argparse
import glob
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.phys_retinex import PhysRetinexNet
from losses.physretinex_loss import PhysRetinexV5Losses

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class PairedLOLDataset(Dataset):
    """Paired Low-Light Dataset for LOL-v1 / LOL-v2."""
    def __init__(self, low_dir, patch_size=256, is_training=True):
        self.patch_size = patch_size
        self.is_training = is_training
        
        valid_exts = {".png", ".jpg", ".jpeg"}
        p_low = Path(low_dir)
        self.low_files = sorted([str(f) for f in p_low.glob("*.*") if f.suffix.lower() in valid_exts])
        if not self.low_files:
            raise FileNotFoundError(f"No image files found in {low_dir}")
            
        print(f"[Dataset] Found {len(self.low_files)} images in {low_dir}")

    def __len__(self):
        return len(self.low_files)

    def _find_normal_path(self, low_path):
        for old_s, new_s in [("/Low/", "/Normal/"), ("\\Low\\", "\\Normal\\"), ("/low/", "/normal/"), ("\\low\\", "\\normal\\"), ("/low/", "/high/"), ("\\low\\", "\\high\\")]:
            candidate = low_path.replace(old_s, new_s)
            if os.path.exists(candidate):
                return candidate
        return low_path

    def __getitem__(self, idx):
        low_path = self.low_files[idx]
        norm_path = self._find_normal_path(low_path)
        
        img_low = Image.open(low_path).convert('RGB')
        arr_low = np.asarray(img_low, dtype=np.float32) / 255.0
        
        if os.path.exists(norm_path):
            img_norm = Image.open(norm_path).convert('RGB')
            arr_norm = np.asarray(img_norm, dtype=np.float32) / 255.0
        else:
            arr_norm = arr_low.copy()

        if self.is_training and self.patch_size is not None:
            h, w, _ = arr_low.shape
            ps = self.patch_size
            if h > ps and w > ps:
                top = random.randint(0, h - ps)
                left = random.randint(0, w - ps)
                arr_low = arr_low[top:top+ps, left:left+ps, :]
                arr_norm = arr_norm[top:top+ps, left:left+ps, :]
                if random.random() > 0.5:
                    arr_low = np.fliplr(arr_low)
                    arr_norm = np.fliplr(arr_norm)
                if random.random() > 0.5:
                    arr_low = np.flipud(arr_low)
                    arr_norm = np.flipud(arr_norm)
        elif not self.is_training and self.patch_size is not None:
            h, w, _ = arr_low.shape
            ps = self.patch_size
            if h > ps and w > ps:
                top = (h - ps) // 2
                left = (w - ps) // 2
                arr_low = arr_low[top:top+ps, left:left+ps, :]
                arr_norm = arr_norm[top:top+ps, left:left+ps, :]

        t_low = torch.from_numpy(arr_low.copy()).permute(2, 0, 1).float()
        t_norm = torch.from_numpy(arr_norm.copy()).permute(2, 0, 1).float()
        return t_low, t_norm

def main():
    parser = argparse.ArgumentParser(description="PhysRetinexNet Training")
    parser.add_argument("--train_low_dir", type=str, required=True, help="Path to training low-light images directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=256, help="Training patch crop size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum cosine annealing learning rate")
    parser.add_argument("--delta_max", type=float, default=0.05, help="Reflectance noise headroom bound")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"TRAINING PHYSRETINEXNET")
    print(f"Device: {device} | Epochs: {args.epochs} | LR: {args.lr} | Delta_Max: {args.delta_max}")
    print("=" * 60)

    dataset = PairedLOLDataset(args.train_low_dir, patch_size=args.patch_size, is_training=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = PhysRetinexNet(delta_max=args.delta_max).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable Parameters: {total_params:,} (Standard: 44,765)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    loss_suite = PhysRetinexV5Losses(device=device)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for low_batch, norm_batch in loader:
            low_batch = low_batch.to(device)
            norm_batch = norm_batch.to(device)

            optimizer.zero_grad()
            out = model(low_batch)

            I_enh = out['enhanced']
            R_clean = out['R']
            noise_res = out['noise_map']
            L_low = out['L_l']

            l_rec = loss_suite.loss_reconstruction(low_batch, R_clean, noise_res, L_low)
            l_enh_l1 = loss_suite.loss_enh_l1(I_enh, norm_batch)
            l_ssim = loss_suite.loss_enh_ssim(I_enh, norm_batch)
            l_lpips = loss_suite.loss_enh_lpips(I_enh, norm_batch)
            l_color = loss_suite.loss_color(I_enh, norm_batch)
            l_lum = loss_suite.loss_lum(I_enh, norm_batch)
            l_noise = loss_suite.loss_noise(noise_res, L_low)

            total_loss = (
                0.50 * l_rec +
                1.00 * l_enh_l1 +
                0.20 * l_ssim +
                0.10 * l_lpips +
                0.50 * l_color +
                0.50 * l_lum +
                0.10 * l_noise
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(total_loss.item())

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        elapsed = time.time() - t0
        lr_curr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] Loss: {avg_loss:.4f} | LR: {lr_curr:.2e} | Time: {elapsed:.1f}s")

        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_dir / "physretinex_best.pth")

        # Periodic checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), ckpt_dir / f"physretinex_epoch_{epoch:03d}.pth")

    print(f"\n[Training Finished] Best checkpoint saved to: {ckpt_dir / 'physretinex_best.pth'}")

if __name__ == "__main__":
    main()
