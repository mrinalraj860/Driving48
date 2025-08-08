import torch
import torch.nn as nn

class MotionPointTransformer(nn.Module):
    def __init__(self, num_classes=47, embed_dim=128, num_heads=4, depth=4, dropout=0.1):
        super().__init__()

        # GRU directly on motion (x, y, vis = 3)
        self.temporal_encoder = nn.GRU(
            input_size=3, hidden_size=embed_dim, batch_first=True
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional embedding (likely for 1000 points + CLS)
        self.point_pos_emb = nn.Parameter(torch.randn(1, 1001, embed_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,  # your checkpoint uses 256 here
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Final classifier
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Input: [B, N, T, 3]
        B, N, T, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * T, N, C)  # [B*T, N, 3]

        # GRU across points
        x, _ = self.temporal_encoder(x)                # [B*T, N, D]
        x = x.mean(dim=1)                              # [B*T, D]
        x = x.reshape(B, T, -1)                        # [B, T, D]

        # CLS token and positional encoding
        cls_token = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.point_pos_emb[:, :x.shape[1], :]

        # Transformer + classification
        x = self.transformer(x)                        # [B, T+1, D]
        return self.fc(x[:, 0])                        # [B, num_classes]