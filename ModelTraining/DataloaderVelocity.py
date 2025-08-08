
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

class MotionDataset(Dataset):
    def __init__(self, df_exists, pt_folder, training=True, max_points=1000):
        self.df = df_exists.reset_index(drop=True)
        self.pt_folder = pt_folder
        self.training = training
        self.max_points = max_points

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_name = row['vid_name']
        label = row['label']
        
        pt_path = os.path.join(self.pt_folder, f"{vid_name}_tracking.pt")
        data = torch.load(pt_path, map_location='cpu') 
        
        seg_len, num_points = data['shape_info']
        
        pred_tracks = data['pred_tracks'].reshape(seg_len, num_points, 2) / 512.0
        pred_vis = data['pred_visibility'].reshape(seg_len, num_points, 1)

        velocity = torch.zeros_like(pred_tracks)
        if seg_len > 1:
            velocity[1:] = pred_tracks[1:] - pred_tracks[:-1]

        input_tensor = torch.cat([pred_tracks, pred_vis, velocity], dim=-1)
        
        # --- FIX: Normalize the number of points to self.max_points ---
        current_points = input_tensor.shape[1]
        if current_points > self.max_points:
            # If more points than max, randomly sample a subset
            indices = torch.randperm(current_points)[:self.max_points]
            input_tensor = input_tensor[:, indices, :]
        elif current_points < self.max_points:
            # If fewer points, pad with zeros along the points dimension
            padding_needed = self.max_points - current_points
            padding_shape = (input_tensor.shape[0], padding_needed, input_tensor.shape[2])
            padding = torch.zeros(padding_shape, dtype=input_tensor.dtype)
            input_tensor = torch.cat([input_tensor, padding], dim=1)
        
        # --- Augmentation (Optional but recommended for training) ---
        if self.training:
            # This augmentation is fine as it doesn't change tensor shapes
            if torch.rand(1) > 0.5:
                input_tensor[:, :, :2] += torch.randn_like(input_tensor[:, :, :2]) * 0.02

        return input_tensor, torch.tensor(label).long(), vid_name

# --- UPDATED: Collate Function for the new model ---
# No changes needed here, but it now works because the dataset provides fixed-size tensors.
def motion_collate_fn(batch):
    """
    Pads sequences, creates a key padding mask, and formats data for the model.
    """
    features, labels, video_names = zip(*batch)

    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    lengths = torch.tensor([s.shape[0] for s in features])
    max_len = features_padded.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

    features_tensor = features_padded.permute(0, 2, 1, 3)
    
    labels_tensor = torch.tensor(labels).long()

    return features_tensor, labels_tensor, mask, video_names
