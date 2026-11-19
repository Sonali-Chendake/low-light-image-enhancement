import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual Block with LeakyReLU and Conv layers."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    def forward(self, x):
        return x + self.conv(x)

class IlluminationEstimatorV2(nn.Module):
    """
    Refined Illumination Estimation Subnet with Residual Dense Guidance.
    Anchored to MaxRGB Retinex prior with smooth residual refinement.
    """
    def __init__(self, in_channels=4, hidden_dim=24):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        max_c, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([x, max_c], dim=1)
        h = self.relu(self.in_conv(feat))
        h = self.res1(h)
        h = self.res2(h)
        illum = max_c + 0.08 * self.tanh(self.out_conv(h))
        return torch.clamp(illum, min=1e-4, max=1.0)

class AdaptiveIlluminationCorrectionV3(nn.Module):
    """
    Calibrated Adaptive Illumination Correction with Expanded Dynamic Range.
    Gamma field in [0.10, 0.40] (mean 0.25 +- 0.15) to elevate low-light illumination
    into the natural GT target range while preserving highlight protection.
    """
    def __init__(self, hidden_dim=24):
        super().__init__()
        self.in_conv = nn.Conv2d(1, hidden_dim, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.res = ResidualBlock(hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, L_l):
        h = self.relu(self.in_conv(L_l))
        h = self.res(h)
        gamma_map = 0.25 + 0.15 * self.tanh(self.out_conv(h))
        L_h = torch.pow(L_l, gamma_map)
        return torch.clamp(L_h, min=1e-4, max=1.0)

class NoiseAwareReflectanceEstimatorV5(nn.Module):
    """
    Controlled Noise-Aware Reflectance Estimator with Strictly Bounded Headroom.
    delta_max parameter controls the maximum allowed noise residual.
    Prevents Delta_R from becoming an additive DC brightness booster.
    """
    def __init__(self, hidden_dim=24, delta_max=0.05):
        super().__init__()
        self.delta_max = delta_max
        self.in_conv = nn.Conv2d(4, hidden_dim, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.res = ResidualBlock(hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, 3, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, I_l, L_l):
        R_raw = torch.clamp(I_l / (L_l + 1e-4), 0.0, 1.0)
        feat = torch.cat([R_raw, L_l], dim=1)
        h = self.relu(self.in_conv(feat))
        h = self.res(h)
        noise_residual = self.tanh(self.out_conv(h)) * self.delta_max
        R_clean = torch.clamp(R_raw - noise_residual, 0.0, 1.0)
        return R_clean, R_raw, noise_residual

class PhysRetinexNetV5(nn.Module):
    """
    Proposed PhysRetinex-v5 Framework.
    Model capacity: Exactly 44,765 parameters (strictly matched across all experiments).
    Formulation:
      I_low approx (R_clean + Delta_R) * L_low
      I_high = R_clean * L_high
      I_enh = clamp(I_high, 0.0, 1.0)
    """
    def __init__(self, hidden_dim=24, delta_max=0.05):
        super().__init__()
        self.delta_max = delta_max
        self.illum_estimator = IlluminationEstimatorV2(in_channels=4, hidden_dim=hidden_dim)
        self.illum_correction = AdaptiveIlluminationCorrectionV3(hidden_dim=hidden_dim)
        self.reflectance_estimator = NoiseAwareReflectanceEstimatorV5(hidden_dim=hidden_dim, delta_max=delta_max)

    def forward(self, I_l):
        L_l = self.illum_estimator(I_l)
        L_h = self.illum_correction(L_l)
        R_clean, R_raw, noise_res = self.reflectance_estimator(I_l, L_l)
        I_phys = R_clean * L_h
        I_out = torch.clamp(I_phys, 0.0, 1.0)
        return {
            'enhanced': I_out,
            'I_phys': I_phys,
            'L_l': L_l,
            'L_h': L_h,
            'R': R_clean,
            'R_raw': R_raw,
            'noise_map': noise_res
        }

PhysRetinexNet = PhysRetinexNetV5

