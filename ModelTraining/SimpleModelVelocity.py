import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionTransformer(nn.Module):
    def __init__(self, num_classes=47, point_dim=64, frame_dim=128, 
                 num_heads=4, depth=2, dropout=0.2, T_max=150, num_points=1000):
        super().__init__()
        self.num_points = num_points
        self.T_max = T_max
        
        # Step 1: Point-level encoder (shared across time)
        self.point_encoder = nn.Sequential(
            nn.Linear(4, point_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(point_dim, point_dim),
            nn.ReLU()
        )

        # Step 2: Frame-wise aggregation (mean over points)
        # No spatial transformer – simplification

        # Step 3: Temporal modeling
        self.temporal_proj = nn.Linear(point_dim, frame_dim)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, T_max, frame_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=frame_dim, nhead=num_heads,
            dim_feedforward=frame_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Step 4: Classification head (mean over time)
        self.fc = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, frame_dim // 2),
            nn.ReLU(),
            nn.Linear(frame_dim // 2, num_classes)
        )

    def forward(self, x, mask=None):
        # x: [B, N, T, 4]
        B, N, T, C = x.shape
        x = self.point_encoder(x)  # [B, N, T, point_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, N, point_dim]
        x = x.mean(dim=2)  # [B, T, point_dim]

        x = self.temporal_proj(x)  # [B, T, frame_dim]
        x = x + self.temporal_pos_embed[:, :T, :]  # [B, T, frame_dim]

        if mask is not None:
            x = self.temporal_transformer(x, src_key_padding_mask=mask)
        else:
            x = self.temporal_transformer(x)

        # Aggregate over time (mean)
        x = x.mean(dim=1)  # [B, frame_dim]
        return self.fc(x)  # [B, num_classes]