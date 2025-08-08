import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionTransformer(nn.Module):
    def __init__(self, num_classes=47, point_dim=64, frame_dim=128, 
                 num_heads=8, depth=3, dropout=0.3, T_max=150, num_points=1000):
        super().__init__()
        self.num_points = num_points
        self.T_max = T_max
        self.point_dim = point_dim
        self.frame_dim = frame_dim
        
        # Point-level feature extraction
        self.point_encoder = nn.Sequential(
            nn.Linear(4, point_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(point_dim),
            nn.Linear(point_dim, point_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Spatial transformer to model relationships between points
        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=point_dim,
                nhead=num_heads//2,
                dim_feedforward=point_dim*4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=1
        )
        
        # Learnable spatial aggregation tokens
        self.spatial_tokens = nn.Parameter(torch.randn(1, 16, point_dim))
        
        # Temporal transformer
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, T_max, frame_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, frame_dim))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=frame_dim,
            nhead=num_heads,
            dim_feedforward=frame_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=depth)
        self.temporal_proj = nn.Linear(self.point_dim, self.frame_dim)

        # Output head
        self.fc = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Dropout(dropout),
            nn.Linear(frame_dim, frame_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(frame_dim//2, num_classes)
        )

    def forward(self, x, mask=None):
        B, N, T, C = x.shape
        if N != self.num_points:
            raise ValueError(f"Expected {self.num_points} points, got {N}")
        
        # Point feature extraction
        x = self.point_encoder(x)  # [B, N, T, point_dim]
        
        # Process each frame independently
        frame_features = []
        for t in range(T):
            frame = x[:, :, t, :]  # [B, N, point_dim]
            
            # Add spatial tokens
            spatial_tokens = self.spatial_tokens.expand(B, -1, -1)
            combined = torch.cat([spatial_tokens, frame], dim=1)  # [B, 16+N, point_dim]
            
            # Spatial transformer
            spatial_out = self.spatial_transformer(combined)  # [B, 16+N, point_dim]
            
            # Aggregate spatial tokens
            frame_feature = spatial_out[:, :16].mean(dim=1)  # [B, point_dim]
            frame_features.append(frame_feature)
        
        # Stack frame features
        temporal_seq = torch.stack(frame_features, dim=1)  # [B, T, point_dim]
        
        # Project to temporal dimension
        temporal_seq = F.relu(self.temporal_proj(temporal_seq))
        
        # Add positional encoding
        temporal_seq = temporal_seq + self.temporal_pos_embed[:, :T, :]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        temporal_seq = torch.cat([cls_tokens, temporal_seq], dim=1)  # [B, T+1, frame_dim]
        
        # Temporal transformer
        if mask is not None:
            # Extend mask for CLS token
            extended_mask = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
                mask
            ], dim=1)
            temporal_out = self.temporal_transformer(temporal_seq, src_key_padding_mask=extended_mask)
        else:
            temporal_out = self.temporal_transformer(temporal_seq)
        
        # Classify using CLS token
        return self.fc(temporal_out[:, 0, :])