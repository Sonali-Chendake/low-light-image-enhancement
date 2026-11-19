import torch
import torch.nn as nn
import torch.nn.functional as F
from piq import ssim
import lpips

class PhysRetinexV5Losses(nn.Module):
    """
    Supervised and Physics-Guided Multi-Component Loss Suite for PhysRetinex-v5.
    Components:
      1. L_rec:       Retinex decomposition consistency on low-light input
      2. L_enh_L1:    Direct L1 pixel distance to paired Ground Truth Normal
      3. L_enh_SSIM:  Structural dissimilarity (1 - SSIM) to Ground Truth Normal
      4. L_enh_LPIPS: Perceptual feature distance to Ground Truth Normal
      5. L_color:     Cosine angular color distance to Ground Truth Normal
      6. L_lum:       Luminance consistency to Ground Truth Normal (prevents overshoot)
      7. L_noise:     Conservative dark-region TV + Zero-mean residual constraint
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    @staticmethod
    def rgb_to_luminance(img):
        """Standard ITU-R BT.709 relative luminance."""
        return 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]

    def loss_reconstruction(self, I_low, R_clean, noise_res, L_low):
        """Retinex decomposition consistency: I_low approx (R_clean + Delta_R) * L_low"""
        return torch.abs(I_low - (R_clean + noise_res) * L_low).mean()

    def loss_enh_l1(self, I_enh, I_gt):
        """Direct L1 reconstruction loss to Ground Truth Normal"""
        return torch.abs(I_enh - I_gt).mean()

    def loss_enh_ssim(self, I_enh, I_gt):
        """Structural dissimilarity loss: 1 - SSIM(I_enh, I_gt)"""
        return 1.0 - ssim(I_enh, I_gt, data_range=1.0)

    def loss_enh_lpips(self, I_enh, I_gt):
        """
        Perceptual feature distance (AlexNet LPIPS).
        Evaluated on downsampled 0.5x resolution for 2.5x speedup on CPU.
        """
        # Downsample for fast execution on CPU while preserving feature gradients
        e_down = F.interpolate(I_enh * 2.0 - 1.0, scale_factor=0.5, mode='bilinear', align_corners=False)
        g_down = F.interpolate(I_gt * 2.0 - 1.0, scale_factor=0.5, mode='bilinear', align_corners=False)
        return self.lpips_fn(e_down, g_down).mean()

    def loss_color(self, I_enh, I_gt, eps=1e-7):
        """
        Cosine distance between enhanced output and Ground Truth Normal RGB vectors.
        Supervises toward GT rather than low-light input to remove low-light color cast.
        """
        dot = (I_enh * I_gt).sum(dim=1, keepdim=True)
        norm_e = torch.norm(I_enh, p=2, dim=1, keepdim=True)
        norm_g = torch.norm(I_gt, p=2, dim=1, keepdim=True)
        cos_sim = dot / (norm_e * norm_g + eps)
        return (1.0 - cos_sim).mean()

    def loss_luminance(self, I_enh, I_gt):
        """
        Weak luminance consistency loss to prevent brightness overshooting.
        L_lum = mean(|Y(I_enh) - Y(I_gt)|)
        """
        y_enh = self.rgb_to_luminance(I_enh)
        y_gt = self.rgb_to_luminance(I_gt)
        return torch.abs(y_enh - y_gt).mean()

    def loss_noise_regularization(self, R_clean, noise_res, I_low, sigma_lum=0.20, sigma_edge=0.08):
        """
        Conservative noise regularization:
        1. Dark-and-flat texture-preserving TV on R_clean.
        2. Zero-mean penalty on noise_res to prohibit additive DC bias.
        """
        lum = I_low.mean(dim=1, keepdim=True)
        gx_in = torch.abs(I_low[:, :, :, :-1] - I_low[:, :, :, 1:])
        gy_in = torch.abs(I_low[:, :, :-1, :] - I_low[:, :, 1:, :])
        gx_in = F.pad(gx_in, (0, 1, 0, 0))
        gy_in = F.pad(gy_in, (0, 0, 0, 1))
        edge_mag = (gx_in + gy_in).mean(dim=1, keepdim=True)

        weight_dark = torch.exp(-lum / sigma_lum)
        weight_flat = torch.exp(-edge_mag / sigma_edge)
        noise_weight = (weight_dark * weight_flat).detach()

        gx_r = torch.abs(R_clean[:, :, :, :-1] - R_clean[:, :, :, 1:])
        gy_r = torch.abs(R_clean[:, :, :-1, :] - R_clean[:, :, 1:, :])
        gx_r = F.pad(gx_r, (0, 1, 0, 0))
        gy_r = F.pad(gy_r, (0, 0, 0, 1))

        loss_tv = (noise_weight * (gx_r + gy_r)).mean()
        # Zero-mean constraint: prevents Delta_R from shifting average brightness
        loss_zm = torch.pow(noise_res.mean(), 2)

        return loss_tv + 5.0 * loss_zm

    def forward(self, I_low, I_gt, out_dict, weights):
        """
        Computes total loss and returns dictionary of individual loss terms.
        weights: dict with keys: rec, enh, ssim, lpips, color, lum, noise
        """
        I_enh = out_dict['enhanced']
        L_low = out_dict['L_l']
        R_clean = out_dict['R']
        noise_res = out_dict['noise_map']

        l_rec = self.loss_reconstruction(I_low, R_clean, noise_res, L_low)
        l_enh = self.loss_enh_l1(I_enh, I_gt)
        l_ssim = self.loss_enh_ssim(I_enh, I_gt)
        l_lpips = self.loss_enh_lpips(I_enh, I_gt)
        l_color = self.loss_color(I_enh, I_gt)
        l_lum = self.loss_luminance(I_enh, I_gt)
        l_noise = self.loss_noise_regularization(R_clean, noise_res, I_low)

        total_loss = (
            weights['rec'] * l_rec
            + weights['enh'] * l_enh
            + weights['ssim'] * l_ssim
            + weights['lpips'] * l_lpips
            + weights['color'] * l_color
            + weights['lum'] * l_lum
            + weights['noise'] * l_noise
        )

        return total_loss, {
            'total': total_loss.item(),
            'rec': l_rec.item(),
            'enh_l1': l_enh.item(),
            'ssim': l_ssim.item(),
            'lpips': l_lpips.item(),
            'color': l_color.item(),
            'lum': l_lum.item(),
            'noise': l_noise.item()
        }
