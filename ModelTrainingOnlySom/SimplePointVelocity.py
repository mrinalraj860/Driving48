import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Simpler Model using 1D Convolutions ---

class TemporalCNNClassifier(nn.Module):
    def __init__(self, input_dim=5, num_filters=64, num_classes=47, dropout=0.2):
        """
        A simpler, faster model using 1D convolutions over the time dimension.

        Args:
            input_dim (int): Number of features per point (x,y,vis,vx,vy).
            num_filters (int): The number of output channels for the conv layers.
            num_classes (int): Number of output action classes.
            dropout (float): The dropout value.
        """
        super().__init__()
        
        # This block will process the time-series for each point.
        # We expect input of shape [Batch, Channels, Length], so [B*N, C, T]
        self.temporal_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
        )
        
        # The feature dimension after the convolutions
        feature_dim = num_filters * 2
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, x, mask=None):
        """
        x: Input tensor of shape [B, N, T, C]
            B = batch size, N = num_points, T = seq_len, C = features
        mask: Key padding mask from collate_fn, shape [B, T]. Used for pooling.
        """
        B, N, T, C = x.shape
        
        # Reshape to treat all points' trajectories as a single large batch.
        # Shape becomes: [B * N, T, C]
        x_reshaped = x.contiguous().view(B * N, T, C)
        
        # Permute to fit Conv1d expectation: [Batch, Channels, Length]
        # Shape becomes: [B * N, C, T]
        x_permuted = x_reshaped.permute(0, 2, 1)
        
        # Pass through the temporal CNN feature extractor.
        # Shape: [B * N, feature_dim, T]
        temporal_features = self.temporal_feature_extractor(x_permuted)
        
        # Aggregate over the time dimension (T).
        # We use the mask to perform a masked average pooling.
        if mask is not None:
            # Repeat mask for each point: [B, T] -> [B*N, T]
            mask_repeated = mask.repeat_interleave(N, dim=0)
            # Invert mask for multiplication (1 for real, 0 for padding)
            # and reshape to [B*N, 1, T] for broadcasting
            inverted_mask = ~mask_repeated.unsqueeze(1)
            temporal_features = temporal_features * inverted_mask
            # Sum and divide by the number of non-padded elements
            pooled_out = temporal_features.sum(dim=2) / inverted_mask.sum(dim=2)
        else:
            # If no mask, use global average pooling
            pooled_out = temporal_features.mean(dim=2) # Shape: [B*N, feature_dim]
        
        # Reshape back to separate points for aggregation.
        # Shape: [B, N, feature_dim]
        point_features = pooled_out.view(B, N, -1)
        
        # Max-pooling across the points (N) dimension to get a global feature vector.
        # This is the "PointNet" aggregation step.
        # Shape: [B, feature_dim]
        global_feature = torch.max(point_features, dim=1)[0]
        
        # Classify the global feature.
        # Shape: [B, num_classes]
        logits = self.classifier(global_feature)
        
        return logits