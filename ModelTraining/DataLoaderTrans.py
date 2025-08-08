import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
# Augmentation function remains the same
def augment_points(tensor):
    if torch.rand(1) > 0.5:
        tensor += torch.randn(2) * 0.02
        
    if torch.rand(1) > 0.5:
        scale = 0.9 + 0.2 * torch.rand(1)
        tensor *= scale
        
    if torch.rand(1) > 0.5:
        mask = torch.rand_like(tensor[..., 0]) > 0.1
        tensor = tensor * mask.unsqueeze(-1)
    
    return tensor

class MotionDataset(Dataset):
    def __init__(self, df_exists, pt_folder, training=True, T_max=150):
        """
        df_exists: DataFrame with columns ['vid_name', 'label']
        pt_folder: Folder containing .pt files named as {vid_name}_tracking.pt
        training: Whether this is training data
        T_max: Fixed temporal length (frames) to pad/crop to
        """
        self.df = df_exists.reset_index(drop=True)
        self.pt_folder = pt_folder
        self.training = training
        self.T_max = T_max

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_name = row['vid_name']
        label = row['label']
        
        pt_path = os.path.join(self.pt_folder, f"{vid_name}_tracking.pt")
        data = torch.load(pt_path, map_location=torch.device('cpu'))
        seg_len, num_points = data['shape_info']
        pred_tracks = data['pred_tracks'].reshape(seg_len, num_points, 2) / 512.0
        pred_vis = data['pred_visibility'].reshape(seg_len, num_points, 1)

        # Calculate velocity (pos[t] - pos[t-1])
        # The first frame's velocity is zero
        velocity = torch.zeros_like(pred_tracks)
        velocity[1:] = pred_tracks[1:] - pred_tracks[:-1]
        
        # Concatenate features: [pos_x, pos_y, vis, vel_x, vel_y]
        # Shape: [T, N, 5]
        input_tensor = torch.cat([pred_tracks, pred_vis, velocity], dim=-1)

        return input_tensor, torch.tensor(label).long(), vid_name


def motion_collate_fn(batch):
    """
    Pads to max length in batch and creates a key padding mask.
    """
    features, labels, video_names = zip(*batch)

    # Get the feature dimension from the first item
    C = features[0].shape[-1] # Should be 5 now
    # print(f"Feature dimension: {C}")
    # Pad sequences to the length of the longest sequence in the batch
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    # Create a boolean mask: True for padded values, False for real values
    lengths = torch.tensor([s.shape[0] for s in features])
    max_len = features_padded.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

    # Permute features to be [B, N, T, C] for consistency with some models
    # features_padded is [B, T, N, C]
    features_tensor = features_padded.permute(0, 2, 1, 3) # [B, N, T, C]
    
    labels_tensor = torch.tensor(labels).long()

    # The mask is [B, T]. Depending on the model, you might need it.
    return features_tensor, labels_tensor, mask, video_names