import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- You can keep your existing PositionalEncoding class ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# --- NEW: Robust Transformer Classifier ---
class RobustPointTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=128, nhead=8, num_encoder_layers=3, num_classes=47, dropout=0.2):
        super().__init__()
        
        # 1. Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        # The transformer itself doesn't know sequence order, so we add this.
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout, 
            batch_first=True # Expects [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), # Normalize features before final layers
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.d_model = d_model

    def forward(self, x, mask=None):
        """
        x: Input tensor of shape [B, N, T, C]
        mask: Key padding mask from collate_fn, shape [B, T]
              (True for padded values, False for real values)
        """
        B, N, T, C = x.shape
        
        # Reshape to treat all points' trajectories as a single large batch
        # Shape becomes: [B * N, T, C]
        x_reshaped = x.permute(0, 1, 3, 2).contiguous().view(B * N, T, C)
        
        # Project input features to model dimension
        # Shape: [B * N, T, d_model]
        x_proj = self.input_proj(x_reshaped)
        
        # Apply positional encoding
        # Transformer expects [T, B*N, d_model], so we permute
        x_pos = self.pos_encoder(x_proj.permute(1, 0, 2)).permute(1, 0, 2)
        
        # Repeat mask for each point's trajectory
        # mask shape is [B, T], we need [B*N, T]
        mask_repeated = mask.repeat_interleave(N, dim=0)

        # Pass through the transformer encoder
        # Shape: [B * N, T, d_model]
        transformer_out = self.transformer_encoder(x_pos, src_key_padding_mask=mask_repeated)
        
        # --- KEY CHANGE: MASKED AVERAGE POOLING ---
        # This is the correct way to handle padded sequences.
        
        # Invert mask for multiplication: real values are 1, padded values are 0
        output_mask = ~mask_repeated # Shape: [B*N, T]
        output_mask = output_mask.unsqueeze(-1).float() # Shape: [B*N, T, 1]
        
        # Zero out the padded values in the transformer output
        transformer_out = transformer_out * output_mask
        
        # Sum the features over the time dimension
        summed_features = torch.sum(transformer_out, dim=1) # Shape: [B*N, d_model]
        
        # Count the number of non-padded time steps for each sequence
        valid_timesteps = torch.sum(output_mask, dim=1) # Shape: [B*N, 1]
        valid_timesteps = torch.clamp(valid_timesteps, min=1.0) # Avoid division by zero
        
        # Calculate the mean by dividing by the number of valid time steps
        point_features_flat = summed_features / valid_timesteps # Shape: [B*N, d_model]
        
        # --- END OF KEY CHANGE ---

        # Reshape back to separate points
        # Shape: [B, N, d_model]
        point_features = point_features_flat.view(B, N, self.d_model)
        
        # Max-pooling across the points dimension to get a global feature vector
        # Shape: [B, d_model]
        global_feature = torch.max(point_features, dim=1)[0]
        
        # Classify the global feature
        logits = self.classifier(global_feature)
        
        return logits