import torch
import torch.nn as nn

class CNNGRUClassifier(nn.Module):
    def __init__(self, num_classes=7, input_dim=2, cnn_channels=128, gru_hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=cnn_channels, hidden_size=gru_hidden,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Linear(gru_hidden * 2, num_classes)
        )

    def forward(self, x):
        B, N, T, C = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * T, N, C)  # [B*T, 1000, 2]
        x = x.permute(0, 2, 1)                                    # [B*T, 2, 1000]
        x = self.cnn(x).mean(dim=2)                               # [B*T, cnn_channels]
        x = x.view(B, T, -1)                                      # [B, T, cnn_channels]
        out, _ = self.gru(x)                                      # [B, T, 2*hidden]
        return self.fc(out[:, -1])