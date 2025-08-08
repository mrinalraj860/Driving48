# Model.py - Improved Motion Transformer Classifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiScaleSpatialAggregator(nn.Module):
    """Multi-scale spatial attention with residual connections"""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Multi-head attention for spatial aggregation
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=4, 
            dropout=0.1,
            batch_first=True
        )
        
        # Global average pooling branch
        self.global_pool_proj = nn.Linear(feature_dim, feature_dim)
        
        # Point-wise attention branch
        self.point_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, T, N, C]
        B, T, N, C = x.shape
        x_reshaped = x.view(B * T, N, C)
        
        # Branch 1: Self-attention over points
        attn_out, _ = self.spatial_attention(x_reshaped, x_reshaped, x_reshaped)
        spatial_attn_pooled = attn_out.mean(dim=1)  # [B*T, C]
        
        # Branch 2: Weighted global pooling
        point_weights = self.point_attention(x_reshaped)  # [B*T, N, 1]
        weighted_pooled = (x_reshaped * point_weights).sum(dim=1)  # [B*T, C]
        
        # Combine both branches
        combined = torch.cat([spatial_attn_pooled, weighted_pooled], dim=-1)
        fused = self.fusion(combined)  # [B*T, C]
        
        return fused.view(B, T, C)

class ImprovedTemporalPooling(nn.Module):
    """Improved temporal pooling with multiple strategies"""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Learnable pooling weights for different strategies
        self.pool_weights = nn.Parameter(torch.ones(3))
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, D], mask shape: [B, T]
        
        # Strategy 1: Attention pooling
        attn_logits = self.temporal_attention(x).squeeze(-1)  # [B, T]
        attn_logits.masked_fill_(mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)  # [B, T, 1]
        attn_pooled = (x * attn_weights).sum(dim=1)  # [B, D]
        
        # Strategy 2: Max pooling (masked)
        x_masked = x.clone()
        x_masked[mask] = -1e9
        max_pooled, _ = x_masked.max(dim=1)  # [B, D]
        
        # Strategy 3: Average pooling (masked)
        x_masked_avg = x.clone()
        x_masked_avg[mask] = 0
        seq_lengths = (~mask).sum(dim=1, keepdim=True).float()  # [B, 1]
        avg_pooled = x_masked_avg.sum(dim=1) / seq_lengths.clamp(min=1)  # [B, D]
        
        # Combine strategies with learnable weights
        pool_weights_norm = F.softmax(self.pool_weights, dim=0)
        combined = (pool_weights_norm[0] * attn_pooled + 
                   pool_weights_norm[1] * max_pooled + 
                   pool_weights_norm[2] * avg_pooled)
        
        return combined

class MotionTransformerClassifier(nn.Module):
    """Improved Motion Transformer Classifier with better stability and performance"""
    
    def __init__(self, num_classes=48, input_dim=9, d_model=256, n_head=8, n_layers=4, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input feature processing
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Spatial aggregation
        self.spatial_aggregator = MultiScaleSpatialAggregator(d_model // 2)
        
        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout * 0.5)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout * 0.5)
        
        # Transformer encoder with gradient checkpointing support
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 2,  # Reduced from 4x to 2x
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Temporal pooling
        self.temporal_pooling = ImprovedTemporalPooling(d_model)
        
        # Classification head with residual connection
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller gain for stability
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, N, C], mask shape: [B, T]
        
        # Project input features
        B, T, N, C = x.shape
        x_proj = self.input_proj(x.view(B * T * N, C)).view(B, T, N, -1)
        
        # Spatial aggregation
        x_spatial = self.spatial_aggregator(x_proj)  # [B, T, d_model//2]
        
        # Feature embedding
        x_embed = self.embedding(x_spatial)  # [B, T, d_model]
        
        # Add positional encoding
        x_pos = self.pos_encoder(x_embed)
        
        # Transformer encoding
        x_transformer = self.transformer_encoder(x_pos, src_key_padding_mask=mask)
        
        # Temporal pooling
        x_pooled = self.temporal_pooling(x_transformer, mask)  # [B, d_model]
        
        # Classification
        logits = self.classifier_head(x_pooled)  # [B, num_classes]
        
        return logits

# Alternative simpler but effective model
class SimpleMotionClassifier(nn.Module):
    """Simpler but effective baseline model"""
    
    def __init__(self, num_classes=48, input_dim=9, hidden_dim=128, dropout=0.3):
        super().__init__()
        
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Simple attention over points
        self.point_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Temporal processing with LSTM
        self.temporal_encoder = nn.LSTM(
            hidden_dim, hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, N, C]
        B, T, N, C = x.shape
        
        # Encode each point
        x_flat = x.view(B * T, N, C)
        point_features = self.point_encoder(x_flat)  # [B*T, N, hidden_dim]
        
        # Attention over points
        attn_weights = F.softmax(self.point_attention(point_features), dim=1)
        spatial_features = (point_features * attn_weights).sum(dim=1)  # [B*T, hidden_dim]
        spatial_features = spatial_features.view(B, T, -1)  # [B, T, hidden_dim]
        
        # Temporal encoding
        # Pack padded sequence for LSTM
        seq_lengths = (~mask).sum(dim=1).cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(
            spatial_features, seq_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, _) = self.temporal_encoder(packed_input)
        
        # Use final hidden state (from both directions)
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, hidden_dim*2]
        
        # Classification
        logits = self.classifier(final_hidden)
        
        return logits