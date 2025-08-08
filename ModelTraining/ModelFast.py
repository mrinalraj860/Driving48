import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# --- PositionalEncoding Class (No changes needed) ---
# This class is standard and works well.
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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# --- NEW: Fast and Efficient Transformer Classifier ---
# This model is designed to be significantly faster.
class FastPointTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=256, nhead=8, num_encoder_layers=4, num_classes=47, dropout=0.2):
        super().__init__()
        
        # 1. Point-wise Feature Extractor (MLP)
        # This layer processes each point at each timestep independently.
        # It learns to extract relevant features from the [pos, vis, vel] data.
        self.point_feature_extractor = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.ReLU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder
        # This now operates on a sequence of aggregated features, one per timestep.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout, 
            batch_first=True # We will use batch-first format [B, T, C]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
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
        
        # Reshape for point-wise processing: [B, T, N, C] -> [B*T, N, C]
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * T, N, C)
        
        # 1. Extract features from each point independently
        # Shape: [B*T, N, d_model]
        point_features = self.point_feature_extractor(x_reshaped)
        
        # 2. Aggregate point features at each timestep using max-pooling.
        # This creates a single representative feature vector for the entire point cloud at each time step.
        # Shape: [B*T, d_model]
        time_step_features = torch.max(point_features, dim=1)[0]
        
        # Reshape back to sequence format for the transformer
        # Shape: [B, T, d_model]
        sequence_features = time_step_features.view(B, T, self.d_model)
        
        # 3. Apply Positional Encoding
        # TransformerEncoder expects [T, B, C] if batch_first=False, but we use batch_first=True.
        # So we need to permute for pos_encoder and then permute back.
        sequence_features_pos = sequence_features.permute(1, 0, 2) # [T, B, C]
        sequence_features_pos = self.pos_encoder(sequence_features_pos)
        sequence_features_pos = sequence_features_pos.permute(1, 0, 2) # [B, T, C]
        
        # 4. Pass through the transformer encoder
        # The mask will correctly ignore padded time steps.
        # Shape: [B, T, d_model]
        transformer_out = self.transformer_encoder(sequence_features_pos, src_key_padding_mask=mask)
        
        # 5. Global Average Pooling over the time dimension
        # We must account for the padding mask to only average over valid timesteps.
        if mask is not None:
            # Invert mask: True for real values, False for padded values
            output_mask = ~mask # Shape: [B, T]
            output_mask = output_mask.unsqueeze(-1).float() # Shape: [B, T, 1]
            
            # Zero out the padded values
            transformer_out = transformer_out * output_mask
            
            # Sum the features and divide by the number of valid timesteps
            summed_features = torch.sum(transformer_out, dim=1) # Shape: [B, d_model]
            valid_timesteps = torch.sum(output_mask, dim=1) # Shape: [B, 1]
            valid_timesteps = torch.clamp(valid_timesteps, min=1.0) # Avoid division by zero
            global_feature = summed_features / valid_timesteps # Shape: [B, d_model]
        else:
            # If no mask, just do a simple mean
            global_feature = torch.mean(transformer_out, dim=1)

        # 6. Classify the final feature vector
        logits = self.classifier(global_feature)
        
        return logits