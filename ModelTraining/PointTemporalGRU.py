import torch
import torch.nn as nn


class PointTemporalGRU(nn.Module):
    def __init__(self, num_classes=47, input_dim=2, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x):
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C)                         # [B*N, T, 2]
        out, _ = self.gru(x)                            # [B*N, T, 2*hidden]
        feat = out[:, -1, :]                            # Use last timestep
        feat = feat.view(B, N, -1)                      # [B, N, 2*hidden]
        
        # Attention across points
        attn = self.attention(feat)                     # [B, N, 1]
        attn = torch.softmax(attn, dim=1)
        global_feat = torch.sum(attn * feat, dim=1)     # [B, 2*hidden]

        return self.fc(global_feat)