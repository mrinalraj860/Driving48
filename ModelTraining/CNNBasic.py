import torch
import torch.nn as nn
import torch.nn.functional as F

# === Squeeze-and-Excitation Block ===
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

# === Gated Attention Block ===
class GatingAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)

# === Final Model ===
class PointCNNPlusPlus(nn.Module):
    def __init__(self, num_classes=47, max_frames=64):
        super().__init__()
        self.max_frames = max_frames

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.tpdc = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, d), dilation=(1, d))
            for d in [1, 2, 4]
        ])

        self.fuse_tpdc = nn.Sequential(
            nn.Conv2d(64 * 3, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.attn = GatingAttention(256)

        self.pos_embed = nn.Parameter(torch.randn(1, 256, 1, max_frames))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, N, T, 3] -> [B, 3, N, T]
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)  # [B, 64, N, T]

        # TPDC Multi-scale
        x = torch.cat([F.relu(conv(x)) for conv in self.tpdc], dim=1)  # [B, 192, N, T]
        x = self.fuse_tpdc(x)  # [B, 128, N, T]

        x = self.conv3(x)  # [B, 256, N, T]
        x = self.attn(x)   # [B, 256, N, T]

        # Positional Encoding (interpolate if needed)
        B, C, N, T = x.shape
        if T <= self.max_frames:
            pos = self.pos_embed[:, :, :, :T]
        else:
            pos = F.interpolate(self.pos_embed, size=T, mode='bilinear', align_corners=False)

        x = x + pos.expand(B, -1, N, -1)  # [B, 256, N, T]

        x = self.global_pool(x)  # [B, 256, 1, 1]
        x = x.view(B, -1)        # [B, 256]

        return self.classifier(x)  # [B, num_classes]