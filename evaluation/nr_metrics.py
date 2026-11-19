"""
nr_metrics.py - Standardized No-Reference (NR) Image Quality Assessment Suite
Single source of truth for the Springer SIVP strict No-Reference evaluation.

Metrics included:
  1. Noise Estimation:
     - Input-anchored flat-region shadow noise estimate (MAD on Laplacian response)
     - Shadow Noise-to-Signal Ratio (NSR)
     - Global Immerkaer noise sigma
  2. Overexposure & Highlight Evaluation:
     - Global clipping ratio (% pixels max(R,G,B) >= 250)
     - Input-anchored highlight ROI clipping ratio (% of top 10% brightest pixels clipped)
     - Highlight ROI gradient entropy (discrete Shannon entropy H >= 0)
     - Highlight ROI edge density (% of highlight pixels exceeding gradient threshold)
  3. Color Statistics:
     - Hasler-Susstrunk Colorfulness Index (CCI, [0, 255] scale)
     - Mean CIELAB Chroma (C*ab)
     - Mean Luminance (μ)
  4. Perceptual NR-IQA Baselines:
     - NIQE (scikit-video canonical MSCN features)
     - BRISQUE (piq PyTorch implementation, [0, 1] RGB tensor)
     - PIQE (pypiqe perceptual distortion, [0, 255] BGR uint8)
"""

import os
import cv2
import torch
import numpy as np
import scipy.io
import scipy.linalg
from os.path import dirname, join
from skimage.color import rgb2lab
import skvideo.measure
from skvideo.measure.niqe import _niqe_extract_subband_feats
from skvideo.utils import compute_image_mscn_transform
import piq
import pypiqe

# -------------------------------------------------------------------------
# 1. Noise Estimators
# -------------------------------------------------------------------------
def compute_global_noise_sigma(img_bgr):
    """Immerkaer Laplacian noise estimator on full image grayscale."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    filtered = cv2.filter2D(gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    inner = filtered[2:-2, 2:-2]
    H, W = inner.shape
    return float(np.sum(np.abs(inner)) * np.sqrt(0.5 * np.pi) / (6.0 * H * W))

def compute_shadow_noise_and_nsr(img_enh_bgr, img_low_bgr, shadow_thresh=0.20, min_pixels=100, flat_percentile=30):
    """
    Robust noise estimation in homogeneous shadow regions:
      - Shadow region is anchored to original low-light input (Y_low < shadow_thresh).
      - Flat locations are selected using input-image Sobel gradients (grad <= flat_percentile).
      - Noise standard deviation is estimated via Median Absolute Deviation (MAD) on Laplacian response.
      - NSR = sigma_est / (mean_shadow + 1e-6).
    """
    if img_low_bgr is None:
        raise ValueError("Original low-light input is required for anchored shadow noise estimation.")
    if img_low_bgr.shape[:2] != img_enh_bgr.shape[:2]:
        img_low_bgr = cv2.resize(img_low_bgr, (img_enh_bgr.shape[1], img_enh_bgr.shape[0]), interpolation=cv2.INTER_AREA)

    low_gray = cv2.cvtColor(img_low_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    enh_gray = cv2.cvtColor(img_enh_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 1. Shadow selection anchored to original input
    shadow_mask = (low_gray < shadow_thresh)
    if np.count_nonzero(shadow_mask) < min_pixels:
        shadow_mask = (low_gray <= np.percentile(low_gray, 30))

    # 2. Select homogeneous locations using input-image gradients
    gx = cv2.Sobel(low_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(low_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    shadow_grad = grad[shadow_mask]
    if shadow_grad.size == 0:
        return float("nan"), float("nan"), float("nan")

    flat_limit = np.percentile(shadow_grad, flat_percentile)
    flat_mask = shadow_mask & (grad <= flat_limit)

    # Exclude borders affected by convolution padding
    flat_mask[:2, :] = False
    flat_mask[-2:, :] = False
    flat_mask[:, :2] = False
    flat_mask[:, -2:] = False

    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(enh_gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    values = response[flat_mask]
    if values.size < min_pixels:
        return float("nan"), float("nan"), float("nan")

    # 3. Robust noise estimate via MAD (scaling factor 0.67448975 * sqrt(36) = 0.67448975 * 6.0)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    sigma_est = float(mad / (0.67448975 * 6.0))
    mean_shadow = float(np.mean(enh_gray[flat_mask]))
    nsr = float(sigma_est / (mean_shadow + 1e-6))
    return sigma_est, nsr, mean_shadow

# -------------------------------------------------------------------------
# 2. Overexposure & Highlight Evaluation
# -------------------------------------------------------------------------
def compute_highlight_metrics(img_enh_bgr, img_low_bgr, clip_val=250, roi_percentile=90, edge_threshold=0.05):
    """
    Evaluates global clipping and detail preservation in a fixed, input-anchored highlight ROI.
    Returns:
      global_clip_pct: % of all output pixels with max(R,G,B) >= clip_val
      roi_clip_pct: % of highlight ROI pixels that reach clip_val
      high_entropy: Shannon entropy H >= 0 of normalized gradient magnitude distribution
      edge_density: % of highlight ROI pixels with gradient magnitude > edge_threshold
    """
    if img_low_bgr is None:
        raise ValueError("Original low-light input is required for anchored highlight evaluation.")
    if img_low_bgr.shape[:2] != img_enh_bgr.shape[:2]:
        img_low_bgr = cv2.resize(img_low_bgr, (img_enh_bgr.shape[1], img_enh_bgr.shape[0]), interpolation=cv2.INTER_AREA)

    # 1. Global clipping in the enhanced image
    global_clipped = np.max(img_enh_bgr, axis=2) >= clip_val
    global_clip_pct = float(np.mean(global_clipped) * 100.0)

    # 2. Define highlight ROI anchored to original input
    low_gray = cv2.cvtColor(img_low_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    enh_gray = cv2.cvtColor(img_enh_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    roi_thresh = np.percentile(low_gray, roi_percentile)
    roi_mask = (low_gray >= roi_thresh)
    if np.count_nonzero(roi_mask) < 50:
        roi_mask = (low_gray >= np.percentile(low_gray, 80))

    # 3. Highlight ROI clipping
    roi_clipped = global_clipped & roi_mask
    roi_clip_pct = float(np.sum(roi_clipped) / (np.count_nonzero(roi_mask) + 1e-5) * 100.0)

    # 4. Highlight-region gradient entropy (discrete Shannon entropy H >= 0)
    gx = cv2.Sobel(enh_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enh_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    roi_grads = grad_mag[roi_mask]

    if roi_grads.size > 0:
        counts, _ = np.histogram(roi_grads, bins=32, range=(0.0, 1.0))
        total_counts = np.sum(counts)
        if total_counts > 0:
            p = counts[counts > 0] / total_counts
            high_entropy = float(-np.sum(p * np.log2(p)))
        else:
            high_entropy = 0.0
    else:
        high_entropy = 0.0

    # 5. Highlight edge density
    edge_map = (grad_mag > edge_threshold) & roi_mask
    edge_density = float(np.sum(edge_map) / (np.count_nonzero(roi_mask) + 1e-5) * 100.0)

    return global_clip_pct, roi_clip_pct, high_entropy, edge_density

# -------------------------------------------------------------------------
# 3. Color Statistics
# -------------------------------------------------------------------------
def compute_color_metrics(img_bgr):
    """
    Computes:
      - Hasler-Susstrunk Colorfulness Index (CCI) on standard [0, 255] scale
      - Mean CIELAB Chroma (C*ab)
      - Mean Luminance (μ)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # Hasler-Susstrunk Colorfulness ([0, 255] scale)
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rgyb = np.sqrt(np.var(rg) + np.var(yb))
    mean_rgyb = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    cci = float(std_rgyb + 0.3 * mean_rgyb)

    # CIELAB Chroma
    lab = rgb2lab(img_rgb / 255.0)
    chroma = np.sqrt(lab[:, :, 1]**2 + lab[:, :, 2]**2)
    mean_chroma = float(np.mean(chroma))

    # Mean Luminance
    lum = float(np.mean(0.2126 * (R / 255.0) + 0.7152 * (G / 255.0) + 0.0722 * (B / 255.0)))
    return cci, mean_chroma, lum

# -------------------------------------------------------------------------
# 4. Perceptual No-Reference Baselines (NIQE, BRISQUE, PIQE)
# -------------------------------------------------------------------------
mat_path = join(dirname(skvideo.measure.__file__), 'data', 'niqe_image_params.mat')
niqe_params = scipy.io.loadmat(mat_path)
pop_mu = np.ravel(niqe_params["pop_mu"])
pop_cov = niqe_params["pop_cov"]

def extract_on_patches_fixed(img, patch_size):
    h, w = img.shape
    patch_size = int(patch_size)
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)
    patches = np.array(patches)
    return np.array([_niqe_extract_subband_feats(p) for p in patches])

def calculate_niqe(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    patch_size = 96
    h, w = gray.shape
    hoffset = h % patch_size
    woffset = w % patch_size
    if hoffset > 0: gray = gray[:-hoffset, :]
    if woffset > 0: gray = gray[:, :-woffset]
    img2 = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    mscn1, _, _ = compute_image_mscn_transform(gray)
    mscn2, _, _ = compute_image_mscn_transform(img2)
    feats_lvl1 = extract_on_patches_fixed(mscn1.astype(np.float32), patch_size)
    feats_lvl2 = extract_on_patches_fixed(mscn2.astype(np.float32), int(patch_size / 2))
    if len(feats_lvl1) == 0 or len(feats_lvl2) == 0:
        return float("nan")
    feats = np.hstack((feats_lvl1, feats_lvl2))
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)
    X = sample_mu - pop_mu
    covmat = (pop_cov + sample_cov) / 2.0
    pinvmat = scipy.linalg.pinv(covmat)
    return float(np.sqrt(np.dot(np.dot(X, pinvmat), X)))

def calculate_brisque(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return float(piq.brisque(t, data_range=1.0).item())

def calculate_piqe(img_bgr):
    score, _, _, _ = pypiqe.piqe(img_bgr)
    return float(score)
