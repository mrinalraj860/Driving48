import torch
import torch.nn as nn

class MotionPointTransformer(nn.Module):
    def __init__(self, dim=128, depth=4, heads=8, mlp_dim=256, num_classes=47, dropout=0.1):
        super().__init__()
        
        self.temporal_encoder = nn.GRU(
            input_size=3,       # (x, y, visibility)
            hidden_size=dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.point_pos_emb = nn.Parameter(torch.zeros(1, 1000 + 1, dim))  # 1000 points + cls
        nn.init.trunc_normal_(self.point_pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: [B, N=1000, T, 3]
        B, N, T, C = x.shape

        # Flatten batch and point dim for GRU
        x = x.reshape(B * N, T, C)  # [B*N, T, 3]
        x, _ = self.temporal_encoder(x)  # [B*N, T, dim]
        x = x[:, -1, :]  # take final timestep → [B*N, dim]
        x = x.reshape(B, N, -1)  # [B, N, dim]

        # Add cls token and point positional embeddings
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, dim]
        x = torch.cat([cls_token, x], dim=1)          # [B, N+1, dim]
        x = x + self.point_pos_emb[:, :x.size(1), :]

        # Transformer across points (not time)
        x = self.transformer(x)  # [B, N+1, dim]

        return self.fc(x[:, 0])  # [B, num_classes]