import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Add positional encoding to temporal sequences"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class EnhancedTemporalEncoder(nn.Module):
    """Enhanced temporal encoder with deeper architecture and residual connections"""
    def __init__(self, input_dim=8, hidden_dim=256, num_layers=4):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # Multi-layer temporal convolution with residual connections
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim//4),  # Depthwise
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),  # Pointwise
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ))

        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):  # x: [B, N, T, C], mask: [B, T]
        B, N, T, C = x.shape

        # Project input features
        x = x.reshape(B * N, T, C)
        x = self.input_projection(x)  # [B*N, T, D]

        # Add positional encoding
        x = x.transpose(0, 1)  # [T, B*N, D]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [B*N, T, D]

        # Apply temporal convolutions with residual connections
        x_conv = x.transpose(1, 2)  # [B*N, D, T]
        for conv_layer in self.conv_layers:
            residual = x_conv
            x_conv = conv_layer(x_conv) + residual  # Residual connection

        x_conv = x_conv.transpose(1, 2)  # [B*N, T, D]

        # Apply temporal attention
        if mask is not None:
            # Create attention mask for padding
            expanded_mask = mask.unsqueeze(1).expand(B, N, T).contiguous().view(B * N, T)
            attn_mask = expanded_mask  # True for padded positions
        else:
            attn_mask = None

        x_attn, _ = self.temporal_attention(x_conv, x_conv, x_conv, key_padding_mask=attn_mask)

        # Residual connection and layer norm
        x = self.layer_norm(x_conv + x_attn)

        # Masked pooling
        if mask is not None:
            expanded_mask = mask.unsqueeze(1).expand(B, N, T).contiguous().view(B * N, T)
            x = x * (~expanded_mask).unsqueeze(-1).float()
            valid_lengths = (~expanded_mask).sum(dim=1, keepdim=True).float()
            valid_lengths = torch.clamp(valid_lengths, min=1.0)
            x = x.sum(dim=1) / valid_lengths  # [B*N, D]
        else:
            x = x.mean(dim=1)  # [B*N, D]

        return x.view(B, N, -1)  # [B, N, D]

class SpatialPointNet(nn.Module):
    """Enhanced spatial feature extraction using PointNet-style architecture"""
    def __init__(self, point_dim=256, output_dim=512):
        super().__init__()
        self.point_conv = nn.Sequential(
            nn.Conv1d(point_dim, point_dim * 2, 1),
            nn.BatchNorm1d(point_dim * 2),
            nn.GELU(),
            nn.Conv1d(point_dim * 2, point_dim * 2, 1),
            nn.BatchNorm1d(point_dim * 2),
            nn.GELU(),
            nn.Conv1d(point_dim * 2, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
        )

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: [B, N, D]
        x = x.transpose(1, 2)  # [B, D, N]

        # Point-wise convolutions
        x = self.point_conv(x)  # [B, output_dim, N]

        # Global feature aggregation
        max_feat = self.max_pool(x).squeeze(-1)  # [B, output_dim]
        avg_feat = self.avg_pool(x).squeeze(-1)  # [B, output_dim]

        # Combine max and average pooling
        return max_feat + avg_feat  # [B, output_dim]

class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple temporal scales"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.scales = [1, 2, 4]  # Different temporal scales
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3*scale, padding=3*scale//2, dilation=scale),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1)
            ) for scale in self.scales
        ])

    def forward(self, x):  # x: [B*N, C, T]
        scale_features = []
        for encoder in self.scale_encoders:
            feat = encoder(x).squeeze(-1)  # [B*N, hidden_dim]
            scale_features.append(feat)

        return torch.cat(scale_features, dim=1)  # [B*N, hidden_dim * num_scales]

class ImprovedMotionClassifier(nn.Module):
    def __init__(self, input_dim=8, temporal_dim=256, spatial_dim=512, num_classes=47):
        super().__init__()

        # Enhanced temporal encoding
        self.temporal_encoder = EnhancedTemporalEncoder(
            input_dim=input_dim,
            hidden_dim=temporal_dim,
            num_layers=4
        )

        # Multi-scale feature extraction
        self.multiscale_extractor = MultiScaleFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=64
        )

        # Spatial aggregation
        self.spatial_encoder = SpatialPointNet(
            point_dim=temporal_dim,
            output_dim=spatial_dim
        )

        # Feature fusion
        multiscale_dim = 64 * 3  # 3 scales
        fused_dim = spatial_dim + multiscale_dim

        # Enhanced classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None):  # x: [B, T, N, C], mask: [B, T]
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3)  # → [B, N, T, C]

        # Extract temporal features for each point
        temporal_features = self.temporal_encoder(x, mask)  # [B, N, temporal_dim]

        # Extract multi-scale features
        x_flat = x.reshape(B * N, T, C).permute(0, 2, 1)  # [B*N, C, T]
        multiscale_features = self.multiscale_extractor(x_flat)  # [B*N, multiscale_dim]
        multiscale_features = multiscale_features.view(B, N, -1)  # [B, N, multiscale_dim]
        multiscale_global = multiscale_features.mean(dim=1)  # [B, multiscale_dim]

        # Spatial aggregation of temporal features
        spatial_features = self.spatial_encoder(temporal_features)  # [B, spatial_dim]

        # Fuse all features
        fused_features = torch.cat([spatial_features, multiscale_global], dim=1)

        # Final classification
        return self.classifier(fused_features)

class FocalLoss(nn.Module):
    """Focal loss to handle class imbalance"""
    def __init__(self, alpha=1, gamma=2, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing with optional class weights"""
    def __init__(self, smoothing=0.1, class_weights=None):
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, pred, target):
        n_class = pred.size(1)
        with torch.no_grad():
            one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
            one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)

        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb)

        if self.class_weights is not None:
            weights = self.class_weights[target].unsqueeze(1)  # shape: [B, 1]
            loss = loss * weights

        return loss.sum(dim=1).mean()

# Improved model instantiation
def create_improved_model(input_dim=8, num_classes=47):
    """Factory function to create the improved model"""
    return ImprovedMotionClassifier(
        input_dim=input_dim,
        temporal_dim=256,
        spatial_dim=512,
        num_classes=num_classes
    )