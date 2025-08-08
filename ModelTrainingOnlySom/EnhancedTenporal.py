import torch
import torch.nn as nn
import torch.nn.functional as F

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=500):
#         super().__init__()
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe = torch.zeros(1, max_len, d_model)
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):  # increase max_len
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TemporalAttentionPooling(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention_fc = nn.Linear(feature_dim, 1)

    def forward(self, x, mask=None):
        # x: [B*N, feature_dim, T]
        attn_weights = self.attention_fc(x.permute(0, 2, 1))  # [B*N, T, 1]
        if mask is not None:
            mask = (~mask.repeat_interleave(x.size(0)//mask.size(0), dim=0)).unsqueeze(-1)  # [B*N, T, 1]
            attn_weights.masked_fill_(~mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)  # [B*N, T, 1]
        pooled = torch.sum(x * attn_weights.permute(0, 2, 1), dim=2)  # [B*N, feature_dim]
        return pooled


class EnhancedTemporalCNNClassifier(nn.Module):
    def __init__(self, input_dim=5, num_filters=64, num_classes=47, dropout=0.4, max_seq_len=300):
        super().__init__()

        self.temporal_feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.positional_encoding = PositionalEncoding(num_filters * 2)
        self.temporal_attention_pooling = TemporalAttentionPooling(num_filters * 2)

        feature_dim = num_filters * 2

        self.pointwise_feature_fc = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, x, mask=None):
        B, N, T, C = x.shape

        x_reshaped = x.contiguous().view(B * N, T, C)
        x_permuted = x_reshaped.permute(0, 2, 1)

        temporal_features = self.temporal_feature_extractor(x_permuted)

        temporal_features = temporal_features.permute(0, 2, 1)
        temporal_features = self.positional_encoding(temporal_features)
        temporal_features = temporal_features.permute(0, 2, 1)

        pooled_out = self.temporal_attention_pooling(temporal_features, mask)

        point_features = pooled_out.view(B, N, -1)

        global_feature, _ = torch.max(point_features, dim=1)

        pointwise_feature = self.pointwise_feature_fc(global_feature)

        logits = self.classifier(pointwise_feature)

        return logits
