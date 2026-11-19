import os
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import cv2

class LOLv2Dataset(data.Dataset):
    """
    LOL-v2 Real Dataset Loader for training and evaluation.
    Loads low-light image, Gaussian-smoothed image (for reverse degradation),
    and optionally the corresponding ground truth image.
    Optimized for CPU/GPU efficiency by cropping patches prior to spatial filtering.
    """
    def __init__(self, low_dir, gt_dir=None, patch_size=256, is_training=True):
        super(LOLv2Dataset, self).__init__()
        self.low_dir = low_dir
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.is_training = is_training

        self.low_files = sorted(glob.glob(os.path.join(low_dir, "*.png")) + glob.glob(os.path.join(low_dir, "*.jpg")))
        if len(self.low_files) == 0:
            raise ValueError(f"No image files found in {low_dir}")

        print(f"[LOLv2Dataset] Loaded {len(self.low_files)} images from {low_dir} (is_training={is_training})")

    def __len__(self):
        return len(self.low_files)

    def __getitem__(self, idx):
        low_path = self.low_files[idx]
        img_low = Image.open(low_path).convert('RGB')
        low_np = np.asarray(img_low, dtype=np.float32) / 255.0

        gt_np = None
        if self.gt_dir is not None:
            fname = os.path.basename(low_path)
            gt_name = fname.replace("low", "normal")
            gt_path = os.path.join(self.gt_dir, gt_name)
            if not os.path.exists(gt_path):
                gt_path = os.path.join(self.gt_dir, fname)
            if os.path.exists(gt_path):
                img_gt = Image.open(gt_path).convert('RGB')
                gt_np = np.asarray(img_gt, dtype=np.float32) / 255.0

        # Efficient patch crop during training prior to filtering
        if self.is_training and self.patch_size is not None:
            h, w, _ = low_np.shape
            ps = self.patch_size
            if h > ps and w > ps:
                top = random.randint(0, h - ps)
                left = random.randint(0, w - ps)
                low_np = low_np[top:top+ps, left:left+ps, :]
                if gt_np is not None:
                    gt_np = gt_np[top:top+ps, left:left+ps, :]

                # Fast data augmentations
                if random.random() > 0.5:
                    low_np = np.fliplr(low_np)
                    if gt_np is not None:
                        gt_np = np.fliplr(gt_np)
                if random.random() > 0.5:
                    low_np = np.flipud(low_np)
                    if gt_np is not None:
                        gt_np = np.flipud(gt_np)

        # Fast Gaussian blur on cropped patch
        img_gauss = cv2.GaussianBlur((low_np * 255.0).astype(np.uint8), (9, 9), sigmaX=1.7, sigmaY=1.7)
        gauss_np = img_gauss.astype(np.float32) / 255.0

        # Convert to torch tensors [C, H, W]
        t_low = torch.from_numpy(low_np.copy()).permute(2, 0, 1).float()
        t_gauss = torch.from_numpy(gauss_np.copy()).permute(2, 0, 1).float()

        if gt_np is not None:
            t_gt = torch.from_numpy(gt_np.copy()).permute(2, 0, 1).float()
            return t_low, t_gauss, t_gt, low_path
        else:
            return t_low, t_gauss, low_path
