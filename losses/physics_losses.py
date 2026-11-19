import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsGuidedLosses(nn.Module):
    """
    Modular suite of physics-guided loss functions for Retinex-based low-light enhancement.
    Each loss is isolated and individually toggleable for controlled ablation experiments.
    """
    def __init__(self, gamma=2.2, target_exposure=0.5):
        super(PhysicsGuidedLosses, self).__init__()
        self.gamma = gamma
        self.target_exposure = target_exposure

    # =========================================================================
    # 1. BASELINE LOSSES (Directly inherited from DI-Retinex)
    # =========================================================================
    def loss_reverse_degradation_baseline(self, enhanced_img, img_gauss, img_lowlight):
        """
        Original DI-Retinex Reverse Degradation Loss in Gamma space.
        Inherited directly from DI-Retinex (Sun et al., IJCV 2025).
        """
        mask = (enhanced_img < 1.0).float()
        # Exposure ratio based on scalar global mean
        ratio = self.target_exposure / (img_lowlight.detach().mean() + 1e-8)
        # Power of gamma = 2.2
        loss_rd = (mask * torch.abs(img_gauss**self.gamma - (1.0 / ratio)**self.gamma * (enhanced_img)**self.gamma)).mean()
        return loss_rd, ratio

    def loss_variance_suppression_baseline(self, k, b):
        """
        Original DI-Retinex Variance Suppression Loss on the offset term.
        Inherited directly from DI-Retinex (Sun et al., IJCV 2025).
        """
        offset_val = k * b - k + b + 1.0
        # Compute spatial variance along H and W (dims 2 and 3), average over channels & batch
        loss_vs = offset_val.var(dim=(2, 3)).mean()
        return loss_vs

    # =========================================================================
    # 2. HYPOTHESIS 1: SPATIALLY ADAPTIVE BRIGHTNESS CORRECTION
    # =========================================================================
    def loss_adaptive_brightness(self, enhanced_img, img_gauss, img_lowlight, r_max=15.0, theta=0.40):
        """
        Modified from DI-Retinex.
        Hypothesis: Replacing the uniform scalar ratio with a spatially-adaptive,
        luminance-bounded ratio prevents highlight blowouts in already bright regions.
        """
        mask = (enhanced_img < 1.0).float()
        lum_local = img_gauss.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Spatially adaptive illumination ratio: smooth decay for brighter regions
        # When lum_local is low -> ratio approaches r_max
        # When lum_local >= theta -> ratio gracefully decays to 1.0 (no over-amplification)
        ratio_map = 1.0 + (r_max - 1.0) / (1.0 + torch.exp(15.0 * (lum_local - theta)))
        ratio_map = ratio_map.expand_as(enhanced_img)

        loss_adapt = (mask * torch.abs(img_gauss**self.gamma - (1.0 / ratio_map)**self.gamma * (enhanced_img + 1e-6)**self.gamma)).mean()
        return loss_adapt, ratio_map

    # =========================================================================
    # 3. HYPOTHESIS 2: ILLUMINATION-DEPENDENT NOISE REGULARIZATION
    # =========================================================================
    def loss_noise_regularization(self, enhanced_img, img_lowlight, sigma_lum=0.20, sigma_edge=0.08):
        """
        Newly formulated combination based on Poisson-Gaussian sensor noise physics.
        Hypothesis: Low-light sensor noise is most severe in low-illumination, flat regions.
        Penalizing high-frequency gradients in dark, non-edge areas suppresses amplified noise
        without softening true structural textures.
        """
        lum = img_lowlight.mean(dim=1, keepdim=True)
        # Compute horizontal and vertical gradients of input
        grad_x_in = torch.abs(img_lowlight[:, :, :, :-1] - img_lowlight[:, :, :, 1:])
        grad_y_in = torch.abs(img_lowlight[:, :, :-1, :] - img_lowlight[:, :, 1:, :])
        
        # Pad back to original dimensions
        grad_x_in = F.pad(grad_x_in, (0, 1, 0, 0))
        grad_y_in = F.pad(grad_y_in, (0, 0, 0, 1))
        edge_mag = (grad_x_in + grad_y_in).mean(dim=1, keepdim=True)

        # Weighting: High only in dark AND flat regions
        weight_dark = torch.exp(-lum / sigma_lum)
        weight_flat = torch.exp(-edge_mag / sigma_edge)
        noise_weight = (weight_dark * weight_flat).detach()

        # Compute gradients of enhanced image
        grad_x_out = torch.abs(enhanced_img[:, :, :, :-1] - enhanced_img[:, :, :, 1:])
        grad_y_out = torch.abs(enhanced_img[:, :, :-1, :] - enhanced_img[:, :, 1:, :])
        grad_x_out = F.pad(grad_x_out, (0, 1, 0, 0))
        grad_y_out = F.pad(grad_y_out, (0, 0, 0, 1))

        loss_noise = (noise_weight * (grad_x_out + grad_y_out)).mean()
        return loss_noise

    # =========================================================================
    # 4. HYPOTHESIS 3: GRADIENT-DOMAIN CONTRAST PRESERVATION
    # =========================================================================
    def loss_contrast_preservation(self, enhanced_img, img_lowlight, ratio_map=None):
        """
        Standard image-processing gradient constraint adapted to Retinex enhancement.
        Hypothesis: Constraining edge gradient magnitude between enhanced image and input
        prevents washed-out, milky mid-tones and preserves sharpness.
        """
        # Horizontal & vertical gradients of input and output
        gx_in = torch.abs(img_lowlight[:, :, :, :-1] - img_lowlight[:, :, :, 1:])
        gy_in = torch.abs(img_lowlight[:, :, :-1, :] - img_lowlight[:, :, 1:, :])
        gx_out = torch.abs(enhanced_img[:, :, :, :-1] - enhanced_img[:, :, :, 1:])
        gy_out = torch.abs(enhanced_img[:, :, :-1, :] - enhanced_img[:, :, 1:, :])

        r_clamped = torch.clamp(ratio_map, 1.0, 3.5) if ratio_map is not None else None
        if r_clamped is not None and r_clamped.ndim == 4:
            rx = r_clamped[:, :, :, :-1]
            ry = r_clamped[:, :, :-1, :]
            loss_c = F.l1_loss(gx_out, gx_in * rx) + F.l1_loss(gy_out, gy_in * ry)
        elif r_clamped is not None and r_clamped.ndim == 0:
            loss_c = F.l1_loss(gx_out, gx_in * r_clamped) + F.l1_loss(gy_out, gy_in * r_clamped)
        else:
            loss_c = F.l1_loss(gx_out, gx_in * 2.0) + F.l1_loss(gy_out, gy_in * 2.0)
        return loss_c

    # =========================================================================
    # 5. HYPOTHESIS 4: COLOR CONSTANCY & COSINE ANGLE LOSS
    # =========================================================================
    def loss_color_constancy(self, enhanced_img, img_lowlight, eps=1e-7):
        """
        Standard vector-space color angle conservation.
        Hypothesis: Constraining the cosine angle between input and output RGB vectors
        prevents hue shifting and color tinting while leaving luminance free.
        """
        dot_product = torch.sum(enhanced_img * img_lowlight, dim=1, keepdim=True)
        norm_enh = torch.norm(enhanced_img, p=2, dim=1, keepdim=True)
        norm_low = torch.norm(img_lowlight, p=2, dim=1, keepdim=True)
        
        cos_sim = dot_product / (norm_enh * norm_low + eps)
        # Cosine distance: 1 - cos(theta)
        loss_col = (1.0 - cos_sim).mean()
        return loss_col

    # =========================================================================
    # 6. ILLUMINATION SMOOTHNESS WITH EDGE-AWARE WEIGHTING
    # =========================================================================
    def loss_illumination_smoothness(self, L_l, img_lowlight, alpha=10.0):
        """
        Edge-aware Total Variation on illumination field.
        Penalizes gradients in L_l except across strong edges in input I_low.
        """
        lum = img_lowlight.mean(dim=1, keepdim=True)
        gx_in = torch.abs(lum[:, :, :, :-1] - lum[:, :, :, 1:])
        gy_in = torch.abs(lum[:, :, :-1, :] - lum[:, :, 1:, :])
        wx = torch.exp(-alpha * gx_in).detach()
        wy = torch.exp(-alpha * gy_in).detach()
        gx_L = torch.abs(L_l[:, :, :, :-1] - L_l[:, :, :, 1:])
        gy_L = torch.abs(L_l[:, :, :-1, :] - L_l[:, :, 1:, :])
        loss_s = (wx * gx_L).mean() + (wy * gy_L).mean()
        return loss_s

