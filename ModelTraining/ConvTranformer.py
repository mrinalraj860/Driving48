import torch
import torch.nn as nn

class ConvTransformer(nn.Module):
    def __init__(self, num_classes=47, embed_dim=256, num_heads=8, depth=4, dropout=0.2):
        super().__init__()

        # Define spatial conv once
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Define transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # Input shape: [B, N, T, 2]
        B, N, T, C = x.shape

        # Rearrange to: [B*T, C, N]
        x = x.permute(0, 2, 3, 1).reshape(B * T, C, N)

        # Spatial conv: [B*T, embed_dim, N]
        x = self.spatial_conv(x)

        # Global average over points: [B*T, embed_dim]
        x = x.mean(dim=2)

        # Restore temporal batch: [B, T, embed_dim]
        x = x.view(B, T, -1)

        # Append CLS token: [B, T+1, embed_dim]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Transformer: [B, T+1, embed_dim]
        x = self.transformer(x)

        # Final classification using CLS
        return self.classifier(x[:, 0])