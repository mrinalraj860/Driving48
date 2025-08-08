import torch
import torch.nn as nn

class CNNGRUClassifier(nn.Module):
    def __init__(self, num_classes=47, input_dim=2, cnn_channels=64, gru_hidden=128):
        super().__init__()

        # CNN over points: [B, T, 2] for each point → [B, T, cnn_channels]
        self.point_cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )

        # GRU over time steps
        self.gru = nn.GRU(input_size=cnn_channels, hidden_size=gru_hidden,
                          batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Linear(gru_hidden * 2, num_classes)
        )

    def forward(self, x):
        # x: [B, 1000, T, 2]
        B, N, T, C = x.shape

        # Reshape for CNN: combine batch and point dim
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, 1000, 2]
        x = x.view(B * T, N, C)                 # [B*T, 1000, 2]
        x = x.permute(0, 2, 1)                  # [B*T, 2, 1000] → Conv1D over points
        x = self.point_cnn(x)                   # [B*T, cnn_channels, 1000]
        x = x.mean(dim=2)                       # [B*T, cnn_channels] → average over points

        x = x.view(B, T, -1)                    # [B, T, cnn_channels]

        # GRU over time
        out, _ = self.gru(x)                    # [B, T, 2*hidden]
        last_step = out[:, -1, :]               # [B, 2*hidden]
        return self.fc(last_step)               # [B, num_classes]