import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaselineDIRetinexNet(nn.Module):
    """
    Exact architectural replica of original DI-Retinex enhance_net_nopool (Sun et al., IJCV 2025).
    Keeps original_DI_Retinex untouched while enabling isolated ablation experiments.
    """
    def __init__(self, ch=6, number_f=64):
        super(BaselineDIRetinexNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.e_conv1 = nn.Conv2d(3, number_f, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, ch, kernel_size=3, stride=1, padding=1, bias=True)

    def brightness_contrast(self, x, c, b):
        """
        Original DI-Retinex contrast-brightness transfer equation:
        k = tan((45 - 44.8 * c) / 180 * pi)
        y = k * (x + 0.5 * (b - 1)) + 0.5 * (b + 1)
        """
        k = torch.tan((45.0 - 44.8 * c) / 180.0 * np.pi)
        mean_val = 0.5
        y = k * (x + mean_val * (b - 1.0)) + mean_val * (b + 1.0)
        y = torch.clamp(y, 0.0, 1.0)
        return y, k

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.e_conv3(x2)

        half_ch = x3.shape[1] // 2
        c_raw = x3[:, :half_ch]
        b_raw = x3[:, half_ch:]

        c = self.tanh(c_raw)
        b = self.sigmoid(b_raw)

        enhanced_image, k = self.brightness_contrast(x, c, b)
        return enhanced_image, c, b, k
