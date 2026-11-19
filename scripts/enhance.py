"""
enhance.py - Quick Inference Script for PhysRetinexNet
Enhances a single low-light image or an entire directory of low-light images.

Usage:
  python scripts/enhance.py --input path/to/low_image.png --output path/to/enhanced.png
  python scripts/enhance.py --input path/to/input_folder --output path/to/output_folder --checkpoint checkpoints/physretinex_best.pth
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.phys_retinex import PhysRetinexNet

def enhance_image(model, img_bgr, device):
    """Enhances a single BGR image using PhysRetinexNet."""
    H, W = img_bgr.shape[:2]
    # Pad to multiple of 8 if required
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        img_bgr = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, dict):
            enhanced_t = out['enhanced']
        elif isinstance(out, (tuple, list)):
            enhanced_t = out[0]
        else:
            enhanced_t = out

    enh_np = enhanced_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enh_np = np.clip(enh_np, 0.0, 1.0)
    if pad_h > 0 or pad_w > 0:
        enh_np = enh_np[:H, :W, :]

    enh_bgr = cv2.cvtColor((enh_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return enh_bgr

def main():
    parser = argparse.ArgumentParser(description="PhysRetinexNet Low-Light Image Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output image or directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/physretinex_best.pth", help="Path to model weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[Device] Using {device}")

    # Load Model
    model = PhysRetinexNet().to(device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path

    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print(f"[Model] Successfully loaded weights from: {ckpt_path}")
    else:
        print(f"[Warning] Checkpoint not found at {ckpt_path}. Running with initialized weights.")

    model.eval()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.is_file():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(in_path))
        if img is None:
            raise ValueError(f"Could not read image: {in_path}")
        enh = enhance_image(model, img, device)
        cv2.imwrite(str(out_path), enh)
        print(f"[Saved] Enhanced image saved to: {out_path}")
    elif in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        files = sorted([f for f in in_path.glob("*.*") if f.suffix.lower() in valid_exts])
        print(f"[Processing] Found {len(files)} images in {in_path}...")
        for idx, f in enumerate(files, 1):
            img = cv2.imread(str(f))
            if img is None: continue
            enh = enhance_image(model, img, device)
            cv2.imwrite(str(out_path / f.name), enh)
            if idx % 20 == 0 or idx == len(files):
                print(f"  Processed {idx}/{len(files)} images...")
        print(f"[Saved] All enhanced images saved to: {out_path}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {in_path}")

if __name__ == "__main__":
    main()
