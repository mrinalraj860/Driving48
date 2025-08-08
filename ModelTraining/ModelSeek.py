# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # def knn_search(x, k=8):
# #     """
# #     Batch-wise k-NN search using PyTorch (MPS compatible)
# #     Input: x - [B, N, C] coordinates (x,y only)
# #     Output: idx - [B, N, k] neighbor indices
# #     """
# #     B, N, _ = x.shape
# #     device = x.device
    
# #     # Compute pairwise distances
# #     x_trans = x.transpose(1, 2)  # [B, C, N]
# #     xx = torch.bmm(x, x_trans)    # [B, N, N]
# #     x_norm = xx.diagonal(dim1=1, dim2=2).unsqueeze(-1)  # [B, N, 1]
# #     dist = x_norm - 2*xx + x_norm.transpose(1, 2)  # [B, N, N]
    
# #     # Find k-nearest neighbors (ignore self)
# #     _, topk = torch.topk(dist, k=k+1, dim=-1, largest=False)
# #     return topk[:, :, 1:]  # Remove self index

# # class EfficientGraphConv(nn.Module):
# #     def __init__(self, in_channels, out_channels, k=8):
# #         super().__init__()
# #         self.k = k
# #         self.conv = nn.Sequential(
# #             nn.Conv2d(in_channels, out_channels, kernel_size=1),
# #             nn.BatchNorm2d(out_channels),
# #             nn.ReLU()
# #         )
        
# #     def forward(self, x):
# #         # x: [B, C, P, T]
# #         B, C, P, T = x.shape
        
# #         # Process each frame independently
# #         all_features = []
# #         for t in range(T):
# #             frame_data = x[..., t]  # [B, C, P]
            
# #             # Get coordinates (first 2 channels)
# #             coords = frame_data[:, :2].permute(0, 2, 1)  # [B, P, 2]
            
# #             # Find k-NN indices for this frame
# #             with torch.no_grad():
# #                 idx = knn_search(coords, self.k)  # [B, P, k]
            
# #             # Gather neighbor features
# #             neighbors = frame_data.unsqueeze(2).expand(-1, -1, P, -1)  # [B, C, P, P]
# #             idx = idx.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, P, k]
# #             neighbor_features = torch.gather(neighbors, 3, idx)  # [B, C, P, k]
            
# #             # Central point
# #             central = frame_data.unsqueeze(3)  # [B, C, P, 1]
            
# #             # Concatenate and process
# #             grouped = torch.cat([central, neighbor_features], dim=3)  # [B, C, P, k+1]
# #             out = self.conv(grouped)  # [B, out_channels, P, k+1]
# #             out = out.max(dim=3)[0]  # Max pooling -> [B, out_channels, P]
# #             all_features.append(out)
        
# #         return torch.stack(all_features, dim=3)  # [B, out_channels, P, T]

# # class EfficientSTGCN(nn.Module):
# #     def __init__(self, num_classes, in_channels=3, base_dim=64, temporal_kernel=5):
# #         super().__init__()
# #         # Input: [B, P, T, C] -> permute to [B, C, P, T]
# #         self.embed = nn.Sequential(
# #             nn.Conv2d(in_channels, base_dim, 1),
# #             nn.BatchNorm2d(base_dim),
# #             nn.ReLU()
# #         )
        
# #         # Spatial processing blocks
# #         self.spatial_blocks = nn.ModuleList([
# #             EfficientGraphConv(base_dim, base_dim),
# #             EfficientGraphConv(base_dim, base_dim*2),
# #             EfficientGraphConv(base_dim*2, base_dim*4)
# #         ])
        
# #         # Temporal processing
# #         self.temp_convs = nn.ModuleList([
# #             nn.Sequential(
# #                 nn.Conv2d(base_dim, base_dim, (1, temporal_kernel), padding=(0, temporal_kernel//2)),
# #                 nn.BatchNorm2d(base_dim),
# #                 nn.ReLU()
# #             ),
# #             nn.Sequential(
# #                 nn.Conv2d(base_dim*2, base_dim*2, (1, temporal_kernel), padding=(0, temporal_kernel//2)),
# #                 nn.BatchNorm2d(base_dim*2),
# #                 nn.ReLU()
# #             )
# #         ])
        
# #         # Classifier
# #         self.pool = nn.AdaptiveAvgPool2d((1, 1))
# #         self.fc = nn.Sequential(
# #             nn.Linear(base_dim*4, base_dim*8),
# #             nn.ReLU(),
# #             nn.Dropout(0.5),
# #             nn.Linear(base_dim*8, num_classes)
# #         )
        
# #         # For Grad-CAM
# #         self.last_activations = None
        
# #     def forward(self, x):
# #         # x: [B, P, T, C] -> [B, C, P, T]
# #         x = x.permute(0, 3, 1, 2).contiguous()
# #         x = self.embed(x)
        
# #         # Spatial-temporal processing
# #         for i, block in enumerate(self.spatial_blocks):
# #             x = block(x)  # Spatial aggregation
# #             if i < len(self.temp_convs):
# #                 x = self.temp_convs[i](x)  # Temporal convolution
        
# #         # Save for Grad-CAM
# #         self.last_activations = x
        
# #         # Global pooling
# #         x = self.pool(x).flatten(1)
# #         return self.fc(x)


import torch
import torch.nn as nn
import torch.nn.functional as F

# def knn_search(x, k=8):
#     """
#     x: [B, N, 2] (coordinates only)
#     returns: [B, N, k] indices of nearest neighbors
#     """
#     B, N, _ = x.shape
#     xx = torch.bmm(x, x.transpose(1, 2))  # [B, N, N]
#     x_norm = xx.diagonal(dim1=1, dim2=2).unsqueeze(-1)
#     dist = x_norm - 2 * xx + x_norm.transpose(1, 2)
#     _, topk = torch.topk(dist, k=k+1, dim=-1, largest=False)
#     return topk[:, :, 1:]  # remove self

# class EfficientGraphConv(nn.Module):
#     def __init__(self, in_channels, out_channels, k=8):
#         super().__init__()
#         self.k = k
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )

#     def forward(self, x, visibility):
#         # x: [B, C, P, T]   visibility: [B, 1, P, T]
#         B, C, P, T = x.shape
#         vis_mask = visibility.expand(-1, C, -1, -1)

#         out_frames = []
#         for t in range(T):
#             feat = x[:, :, :, t]             # [B, C, P]
#             coords = feat[:, :2, :].permute(0, 2, 1)  # [B, P, 2]

#             with torch.no_grad():
#                 idx = knn_search(coords, self.k)  # [B, P, k]

#             neighbors = feat.unsqueeze(2).expand(-1, -1, P, -1)
#             idx = idx.unsqueeze(1).expand(-1, C, -1, -1)
#             neighbor_feat = torch.gather(neighbors, 3, idx)  # [B, C, P, k]

#             central = feat.unsqueeze(3)  # [B, C, P, 1]
#             grouped = torch.cat([central, neighbor_feat], dim=3)  # [B, C, P, k+1]

#             # Apply visibility mask to grouped features (if needed)
#             vis_t = vis_mask[:, :, :, t].unsqueeze(3)  # [B, C, P, 1]
#             grouped = grouped * vis_t  # soft attenuation

#             conv_out = self.conv(grouped)  # [B, out_C, P, k+1]
#             pooled = conv_out.max(dim=3)[0]  # max over neighbors → [B, out_C, P]
#             out_frames.append(pooled)

#         return torch.stack(out_frames, dim=3)  # [B, out_C, P, T]

# class EfficientSTGCN(nn.Module):
#     def __init__(self, num_classes, in_channels=3, base_dim=64, hidden_dim=128, k=8):
#         super().__init__()

#         self.embedding = nn.Sequential(
#             nn.Conv2d(in_channels, base_dim, kernel_size=1),
#             nn.BatchNorm2d(base_dim),
#             nn.ReLU()
#         )

#         self.gconv1 = EfficientGraphConv(base_dim, base_dim, k=k)
#         self.gconv2 = EfficientGraphConv(base_dim, base_dim * 2, k=k)

#         self.bi_gru = nn.GRU(input_size=base_dim * 2, hidden_size=hidden_dim,
#                              batch_first=True, bidirectional=True)

#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(hidden_dim, num_classes)
#         )

#         self.last_activations = None

#     def forward(self, x):
#         # x: [B, P, T, 3] (x, y, visibility)
#         coords = x[..., :2]         # [B, P, T, 2]
#         visibility = x[..., 2:]     # [B, P, T, 1]

#         x = x.permute(0, 3, 1, 2).contiguous()  # [B, 3, P, T]
#         vis = visibility.permute(0, 3, 1, 2).contiguous()  # [B, 1, P, T]

#         x = self.embedding(x)  # [B, base_dim, P, T]

#         x = self.gconv1(x, vis)
#         x = self.gconv2(x, vis)  # [B, C, P, T]

#         self.last_activations = x  # for Grad-CAM

#         # Prepare for GRU: [B, C, P, T] → [B, P, T, C] → reshape
#         x = x.permute(0, 2, 3, 1).contiguous()  # [B, P, T, C]
#         B, P, T, C = x.shape
#         x = x.view(B * P, T, C)  # [B*P, T, C]

#         out, _ = self.bi_gru(x)  # [B*P, T, 2*H]
#         out = out[:, -1, :]      # take last frame → [B*P, 2*H]
#         out = out.view(B, P, -1)  # [B, P, 2*H]

#         x = out.mean(dim=1)  # average over all points → [B, 2*H]
#         return self.classifier(x)


class FastSTGCN(nn.Module):
    def __init__(self, num_classes, in_channels=2, base_dim=64):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU()
        )

        self.spatial_temporal = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=(5, 1), padding=(2, 0)),  # spatial
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(),
            nn.Conv2d(base_dim * 2, base_dim * 2, kernel_size=(1, 5), padding=(0, 2)),  # temporal
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_dim * 2, base_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(base_dim * 4, num_classes)
        )

    def forward(self, x):
        # x: [B, P, T, 3]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, 3, P, T]
        x = self.embed(x)                       # [B, base_dim, P, T]
        x = self.spatial_temporal(x)            # [B, base_dim*2, P, T]
        return self.classifier(x)               # [B, num_classes]