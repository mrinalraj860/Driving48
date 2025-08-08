import torch
import torch.nn as nn

class LocalPointTransformer(nn.Module):
    def __init__(self, num_classes=47, num_points=1000, group_size=20, embed_dim=128, num_heads=4, depth=4, dropout=0.1):
        super().__init__()
        self.group_size = group_size
        self.num_groups = num_points // group_size

        # 1. Local MLP on group of 20 points
        self.local_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 2. Positional Encoding for group-level points
        self.group_pos_embed = nn.Parameter(torch.randn(1, self.num_groups, embed_dim))

        # 3. Global transformer across groups & time
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=embed_dim * 2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):  # x: [B, 1000, T, 3]
        B, N, T, _ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * T, N, 3)  # [B*T, 1000, 3]

        # Reshape into [B*T, num_groups, group_size, 3]
        x = x.view(B * T, self.num_groups, self.group_size, 3)

        # Apply local MLP on each group
        x = self.local_embed(x)                        # [B*T, num_groups, group_size, embed_dim]
        x = x.mean(dim=2)                              # [B*T, num_groups, embed_dim] → group-level embedding

        # Add position embedding
        x = x + self.group_pos_embed                   # [B*T, num_groups, embed_dim]

        # Transformer expects [B, Seq, Dim]
        x = self.transformer(x)                        # [B*T, num_groups, embed_dim]

        # Temporal pooling
        x = x.view(B, T, self.num_groups, -1)
        x = x.mean(dim=2)                              # [B, T, embed_dim]
        x = x.mean(dim=1)                              # [B, embed_dim]

        return self.classifier(x)