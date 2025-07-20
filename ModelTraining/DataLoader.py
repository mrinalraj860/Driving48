import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class MotionDataset(Dataset):
    def __init__(self, df_exists, pt_folder):
        """
        df_exists: DataFrame with columns ['vid_name', 'label']
        pt_folder: Folder containing .pt files named as {vid_name}.pt
        """
        self.df = df_exists.reset_index(drop=True)
        self.pt_folder = pt_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_name = row['vid_name']
        label = row['label']
        
        pt_path = os.path.join(self.pt_folder, f"{vid_name}_tracking.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"{pt_path} not found.")

        data = torch.load(pt_path)

        seg_len, num_points = data['shape_info']
        
        pred_tracks = data['pred_tracks'].reshape(seg_len, num_points, 2) / 512
        pred_vis = data['pred_visibility'].reshape(seg_len, num_points, 1)
        
        input_tensor = torch.cat([pred_tracks, pred_vis], dim=-1)  # [T, N, 3]
        
        return input_tensor, torch.tensor(label).long(), vid_name



import torch.nn.functional as F

# def motion_collate_fn(batch):
#     features, labels, video_names = zip(*batch)  # each: [T, N, 3]

#     max_T = max([item.shape[0] for item in features])
    
#     padded_features = []
#     # print(f"Max temporal length (T): {max_T}")
#     for item in features:
#         T, N, C = item.shape
#         pad_len = max_T - T

#         # Pad temporal dimension (T) at the end
#         padded = F.pad(item, pad=(0, 0, 0, 0, 0, pad_len))  # [T, N, 3] → [max_T, N, 3]
#         padded_features.append(padded)

#     features_tensor = torch.stack(padded_features).permute(0, 2, 1, 3)  # [B, N, T, 3]
#     labels_tensor = torch.tensor(labels).long()

#     return features_tensor, labels_tensor, video_names

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def motion_collate_fn(batch):
    features, labels, video_names = zip(*batch)  # each: [T, N, 3]

    all_T = [item.shape[0] for item in features]
    T_avg = round(sum(all_T) / len(all_T))
    # print(f"All T sizes in batch: {all_T}")
    # print(f"Using T_avg = {T_avg}")

    processed = []
    for item in features:
        T, N, C = item.shape

        if T > T_avg:
            # Center crop
            start = (T - T_avg) // 2
            end = start + T_avg
            item = item[start:end, :, :]  # [T_avg, N, 3]

        elif T < T_avg:
            # Symmetric pad (front + back)
            pad_total = T_avg - T
            pad_front = pad_total // 2
            pad_back = pad_total - pad_front

            pad_tensor_front = item[0:1, :, :].expand(pad_front, -1, -1)  # repeat first frame
            pad_tensor_back = item[-1:, :, :].expand(pad_back, -1, -1)    # repeat last frame

            item = torch.cat([pad_tensor_front, item, pad_tensor_back], dim=0)  # [T_avg, N, 3]

        assert item.shape[0] == T_avg, "T mismatch after padding/cropping"
        processed.append(item)

    # Final stacking: [B, T_avg, N, 3] → [B, N, T_avg, 3]
    features_tensor = torch.stack(processed).permute(0, 2, 1, 3)  # [B, N, T, 3]
    labels_tensor = torch.tensor(labels).long()

    return features_tensor, labels_tensor, video_names