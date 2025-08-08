import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd


class MotionDataset(Dataset):
    def __init__(self, df_exists, pt_folder, som2idx, idx2som):
        self.df = df_exists.reset_index(drop=True)
        self.pt_folder = pt_folder
        self.som2idx = som2idx
        self.idx2som = idx2som

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_name = row['vid_name']
        som_label = row['1']

        pt_path = os.path.join(self.pt_folder, f"{vid_name}_tracking.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"{pt_path} not found.")

        data = torch.load(pt_path, map_location=torch.device('cpu'))
        seg_len, num_points = data['shape_info']
        pred_tracks = data['pred_tracks'].reshape(seg_len, num_points, 2) / 512

        label_idx = self.som2idx[som_label]
        return pred_tracks, torch.tensor(label_idx).long(), vid_name
    



import torch
import torch.nn.functional as F
def motion_collate_fn(batch):
    features, labels, video_names = zip(*batch)
    masks = []
    processed = []
    T_max = 150  # Fixed temporal length
    for seq in features:
        T, N, C = seq.shape
        
        # Create mask: True for padded positions
        mask = torch.zeros(T_max, dtype=torch.bool)
        
        if T > T_max:
            # Center crop to T_max
            start = (T - T_max) // 2
            end = start + T_max
            seq = seq[start:end]
        else:
            # Symmetric padding
            pad_front = (T_max - T) // 2
            pad_back = T_max - T - pad_front
            
            # Pad with repeated frames
            front_pad = seq[0].unsqueeze(0).expand(pad_front, N, C)
            back_pad = seq[-1].unsqueeze(0).expand(pad_back, N, C)
            seq = torch.cat([front_pad, seq, back_pad], dim=0)
            
            # Set mask for padded regions
            mask[:pad_front] = True
            mask[T_max - pad_back:] = True

        masks.append(mask)
        processed.append(seq)
    
    # Stack features and masks
    features_tensor = torch.stack(processed).permute(0, 2, 1, 3)  # [B, N, T, C]
    masks_tensor = torch.stack(masks)  # [B, T]
    labels_tensor = torch.stack([torch.tensor(label).long() if not isinstance(label, torch.Tensor) else label for label in labels])
    
    return features_tensor, labels_tensor, masks_tensor, video_names