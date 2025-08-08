import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd

class MotionDataset(Dataset):
    def __init__(self, df_exists, pt_folder, som2idx, training=True, T_max=150):
        """
        df_exists: DataFrame with at least ['vid_name', '1'] where '1' is somersault label.
        pt_folder: Path to .pt files.
        som2idx: Dictionary mapping somersault labels to integer indices.
        training: Whether the dataset is for training (can be used to apply augmentation).
        T_max: Fixed max temporal length.
        """
        self.df = df_exists.reset_index(drop=True)
        self.pt_folder = pt_folder
        self.som2idx = som2idx
        self.training = training
        self.T_max = T_max

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
        pred_tracks = data['pred_tracks'].reshape(seg_len, num_points, 2) / 512.0
        pred_vis = data['pred_visibility'].reshape(seg_len, num_points, 1)

        # Velocity calculation
        velocity = torch.zeros_like(pred_tracks)
        velocity[1:] = pred_tracks[1:] - pred_tracks[:-1]

        # Combine [x, y, vis, vx, vy] -> shape: [T, N, 5]
        input_tensor = torch.cat([pred_tracks, pred_vis, velocity], dim=-1)

        label_idx = self.som2idx[som_label]
        return input_tensor, torch.tensor(label_idx).long(), vid_name

    def augment(self, tensor):
        # Basic augmentation on position (x,y) only
        pos = tensor[..., :2]
        if torch.rand(1).item() > 0.5:
            pos += torch.randn_like(pos) * 0.02
        if torch.rand(1).item() > 0.5:
            scale = 0.9 + 0.2 * torch.rand(1).item()
            pos *= scale
        if torch.rand(1).item() > 0.5:
            mask = torch.rand_like(pos[..., 0]) > 0.1
            pos = pos * mask.unsqueeze(-1)
        tensor[..., :2] = pos
        return tensor

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