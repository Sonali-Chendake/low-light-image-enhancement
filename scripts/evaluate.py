"""
evaluate.py - No-Reference (NR) Benchmark Evaluation Script
Evaluates enhanced images against original low-light inputs using the audited
strict No-Reference (NR) metric suite.

Supported metrics:
  - Noise: Flat shadow noise sigma (MAD on Laplacian), Shadow NSR, Global noise sigma
  - Highlights: Global clipping %, Highlight ROI clipping %, Gradient entropy, Edge density
  - Color & Tone: Hasler-Susstrunk CCI, CIELAB Chroma (C*ab), Mean luminance (mu)
  - NR-IQA: NIQE, BRISQUE, PIQE

Usage:
  python scripts/evaluate.py --low_dir path/to/low --enh_dir path/to/enhanced --output_csv results/eval_results.csv
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.nr_metrics import (
    compute_global_noise_sigma,
    compute_shadow_noise_and_nsr,
    compute_highlight_metrics,
    compute_color_metrics,
    compute_niqe,
    compute_brisque,
    compute_piqe
)

def evaluate_pair(low_img_bgr, enh_img_bgr, compute_nriqa=True):
    """Evaluates all strict NR metrics on a single (input, enhanced) pair."""
    results = {}
    
    # Noise metrics
    results['global_noise_sigma'] = compute_global_noise_sigma(enh_img_bgr)
    sh_sigma, nsr, sh_mean = compute_shadow_noise_and_nsr(enh_img_bgr, low_img_bgr)
    results['shadow_noise_sigma'] = sh_sigma
    results['shadow_nsr'] = nsr
    results['shadow_mean_lum'] = sh_mean
    
    # Highlight & Overexposure metrics
    g_clip, roi_clip, entropy, edge_dens = compute_highlight_metrics(enh_img_bgr, low_img_bgr)
    results['global_clipping_pct'] = g_clip
    results['roi_clipping_pct'] = roi_clip
    results['highlight_gradient_entropy'] = entropy
    results['highlight_edge_density'] = edge_dens
    
    # Color & Tone metrics
    cci, chroma, lum = compute_color_metrics(enh_img_bgr)
    results['cci'] = cci
    results['cielab_chroma'] = chroma
    results['mean_luminance'] = lum
    
    # NR-IQA baselines
    if compute_nriqa:
        try:
            results['niqe'] = compute_niqe(enh_img_bgr)
        except Exception:
            results['niqe'] = float('nan')
        try:
            results['brisque'] = compute_brisque(enh_img_bgr)
        except Exception:
            results['brisque'] = float('nan')
        try:
            results['piqe'] = compute_piqe(enh_img_bgr)
        except Exception:
            results['piqe'] = float('nan')
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Strict No-Reference Evaluation for LLIE")
    parser.add_argument("--low_dir", type=str, required=True, help="Directory containing original low-light images")
    parser.add_argument("--enh_dir", type=str, required=True, help="Directory containing enhanced output images")
    parser.add_argument("--output_csv", type=str, default="results/evaluation_metrics.csv", help="Output CSV path")
    parser.add_argument("--skip_nriqa", action="store_true", help="Skip compute-heavy perceptual metrics (NIQE/BRISQUE/PIQE)")
    args = parser.parse_args()

    low_dir = Path(args.low_dir)
    enh_dir = Path(args.enh_dir)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    enh_files = sorted([f for f in enh_dir.glob("*.*") if f.suffix.lower() in valid_exts])
    if not enh_files:
        raise FileNotFoundError(f"No valid images found in {enh_dir}")

    records = []
    print(f"[Evaluation] Processing {len(enh_files)} images from {enh_dir.name}...")

    for f_enh in enh_files:
        # Match input image by stem or direct name
        candidates = list(low_dir.glob(f"{f_enh.stem}.*")) + list(low_dir.glob(f"{f_enh.name}"))
        if not candidates:
            # Try stripped suffixes like '_enh', '_out', etc.
            stem_clean = f_enh.stem.replace("_enhanced", "").replace("_enh", "").replace("_out", "")
            candidates = list(low_dir.glob(f"{stem_clean}.*"))
        
        if not candidates:
            print(f"[Warning] Could not find matching low-light input for {f_enh.name}, skipping.")
            continue
            
        f_low = candidates[0]
        enh_bgr = cv2.imread(str(f_enh))
        low_bgr = cv2.imread(str(f_low))
        
        if enh_bgr is None or low_bgr is None:
            continue
            
        res = evaluate_pair(low_bgr, enh_bgr, compute_nriqa=not args.skip_nriqa)
        res['image_name'] = f_enh.name
        records.append(res)

    df = pd.DataFrame(records)
    cols = ['image_name'] + [c for c in df.columns if c != 'image_name']
    df = df[cols]
    df.to_csv(out_csv, index=False)
    print(f"\n[Saved] Detailed per-image evaluation written to: {out_csv}")

    # Summary Statistics
    numeric_df = df.select_dtypes(include=[np.number])
    means = numeric_df.mean()
    print("\n" + "="*50)
    print(f"BENCHMARK SUMMARY ({len(records)} images)")
    print("="*50)
    for k, v in means.items():
        print(f"  {k:28s}: {v:10.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
