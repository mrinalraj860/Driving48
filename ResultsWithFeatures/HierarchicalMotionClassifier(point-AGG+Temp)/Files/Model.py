import torch
import torch.nn as nn
import torch.nn.functional as F

class PointTemporalEncoder(nn.Module):
    """Encodes each point's temporal dynamics via 1D CNN with masking"""
    def __init__(self, input_dim=9, hidden_dim=128):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, mask=None):  # x: [B, N, T, C], mask: [B, T]
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C).permute(0, 2, 1)  # → [B*N, C, T]
        out = self.temporal_conv(x)  # → [B*N, D, T]

        if mask is not None:
            # Expand mask for all points: [B, T] → [B*N, T]
            expanded_mask = mask.unsqueeze(1).expand(B, N, T).contiguous().view(B * N, T)
            # Set padded positions to zero: [B*N, D, T]
            out = out * (~expanded_mask).unsqueeze(1).float()

            # Masked average pooling
            valid_lengths = (~expanded_mask).sum(dim=1, keepdim=True).float()  # [B*N, 1]
            valid_lengths = torch.clamp(valid_lengths, min=1.0)  # Avoid division by zero
            out = out.sum(dim=2) / valid_lengths  # → [B*N, D]
        else:
            out = torch.mean(out, dim=2)  # → [B*N, D]

        return out.view(B, N, -1)  # → [B, N, D]

class PointAggregator(nn.Module):
    """Aggregates across points via transformer"""
    def __init__(self, point_dim=128, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=point_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):  # x: [B, N, D]
        # Note: The mask here is for temporal dimension, but we're aggregating across points (N)
        # The transformer operates on the point dimension, not temporal
        # So we don't need to pass the temporal mask to the transformer

        out = self.transformer(x)  # → [B, N, D]
        return out.mean(dim=1)  # Global average pooling → [B, D]

class HierarchicalMotionClassifier(nn.Module):
    def __init__(self, input_dim=8, temporal_dim=128, num_classes=47):
        super().__init__()
        self.temporal_encoder = PointTemporalEncoder(input_dim=input_dim, hidden_dim=temporal_dim)
        self.aggregator = PointAggregator(point_dim=temporal_dim)
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, mask=None):  # x: [B, T, N, C], mask: [B, T]
        x = x.permute(0, 2, 1, 3)  # → [B, N, T, C]
        point_features = self.temporal_encoder(x, mask)  # → [B, N, D]
        global_repr = self.aggregator(point_features)  # → [B, D]
        return self.classifier(global_repr)