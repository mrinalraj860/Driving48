import torch
import torch.nn as nn

class TemporalPointTransformer(nn.Module):
    def __init__(self, input_dim=3, model_dim=128, num_heads=4, num_layers=2, num_classes=47, dropout=0.1):
        super().__init__()
        
        self.model_dim = model_dim
        
        # Input projection from (C) to (model_dim)
        self.input_proj = nn.Linear(input_dim, model_dim)

        # Positional encoding for T
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, 300, model_dim))  # [1, 1, T, D] (max T = 300)

        # Transformer Encoder (shared across all N points)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Temporal pooling: reduce over T
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(model_dim * 1000, 512),  # assuming N = 1000 points
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Input shape: [B, T, N, C]
        B, T, N, C = x.shape

        # Permute to: [B, N, T, C]
        x = x.permute(0, 2, 1, 3)  # [B, N, T, C]

        # Reshape for input projection: [B*N, T, C]
        x = x.reshape(B * N, T, C)

        # Project to model_dim: [B*N, T, D]
        x = self.input_proj(x)

        # Add temporal positional embeddings (broadcasted across N)
        if T > self.pos_embedding.shape[2]:
            raise ValueError(f"Increase max T in pos_embedding, got T={T}")
        x = x + self.pos_embedding[:, :, :T, :].expand(B * N, -1, -1, -1).squeeze(1)

        # Apply transformer: [B*N, T, D]
        x = self.transformer(x)

        # Pool over time T → [B*N, D]
        x = x.permute(0, 2, 1)  # → [B*N, D, T]
        x = self.temporal_pool(x).squeeze(-1)

        # Reshape back: [B, N, D]
        x = x.view(B, N, self.model_dim)

        # Flatten across N points → [B, N*D]
        x = x.reshape(B, -1)

        # Classification
        out = self.classifier(x)
        return out