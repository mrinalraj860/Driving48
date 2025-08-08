import torch
import torch.nn as nn
import torch.optim as optim
class MotionTransformer(nn.Module):
    def __init__(self, num_classes=47, embed_dim=128, num_heads=4, depth=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(2, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B, N, T, C = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()     # [B, T, 1000, 2]
        x = self.embed(x)                          # [B, T, 1000, D]
        x = x.mean(dim=2)                          # [B, T, D] → global point average per frame

        cls = self.cls_token.repeat(B, 1, 1)       # [B, 1, D]
        x = torch.cat([cls, x], dim=1)             # [B, T+1, D]

        out = self.transformer(x)                  # [B, T+1, D]
        return self.fc(out[:, 0])                  # CLS token output