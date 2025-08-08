import torch
import torch.nn as nn

class EnhancedTemporalTransformerClassifier(nn.Module):
    def __init__(self, input_dim=3, num_classes=47, d_model=256, nhead=8,
                 num_encoder_layers=2, dim_feedforward=512, dropout=0.2):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, mask=None):
        # x: [B, T, N, 9] from dataloader
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # → [B, N, T, 9]
        x = x.view(B * N, T, C)                 # → [B*N, T, 9]
        x = x.permute(0, 2, 1)                  # → [B*N, 9, T]

        x = self.temporal_conv(x)              # → [B*N, d_model, T]
        x = x.permute(0, 2, 1)                  # → [B*N, T, d_model]

        if mask is not None:
            # mask: [B, T] → [B*N, T]
            mask_exp = mask.unsqueeze(1).expand(B, N, T).reshape(B * N, T)
            x = x.masked_fill(mask_exp.unsqueeze(-1), 0.0)  # zero out padded time steps

            valid_counts = (~mask_exp).sum(dim=1, keepdim=True).clamp(min=1)  # [B*N, 1]
            x = x.sum(dim=1) / valid_counts                                   # [B*N, d_model]
        else:
            x = x.mean(dim=1)  # fallback if no mask

        # Reshape for transformer: [B, N, d_model]
        x = x.view(B, N, -1)
        x = self.transformer(x)     # [B, N, d_model]
        x = x.mean(dim=1)           # global aggregation across points

        return self.classifier(x)   # [B, num_classes]