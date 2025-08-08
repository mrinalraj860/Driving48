import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
import numpy as np
import random


import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
import numpy as np
import random

class MotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, pt_folder: str, training: bool = True,
                 max_points: int = 1000, augment_prob: float = 0.7):
        self.df = df.reset_index(drop=True)
        self.pt_folder = pt_folder
        self.training = training
        self.max_points = max_points
        self.augment_prob = augment_prob
        print(f"Initialized MotionDataset with {len(self.df)} samples. Training: {self.training}")

    def __len__(self) -> int:
        return len(self.df)

    def _normalize_coordinates(self, coords: torch.Tensor, height: float = 384.0, width: float = 512.0) -> torch.Tensor:
        coords_norm = coords.clone()
        coords_norm[..., 0] = coords_norm[..., 0] / width   # x
        coords_norm[..., 1] = coords_norm[..., 1] / height  # y

        # Clamp to reasonable range to handle outliers
        coords_norm = torch.clamp(coords_norm, -0.1, 1.1)
        return coords_norm

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        vid_name, label = row['vid_name'], row['label']

        pt_path = os.path.join(self.pt_folder, f"{vid_name}_tracking.pt")

        try:
            data = torch.load(pt_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            # Return a dummy tensor for failed loads
            return (torch.zeros(10, self.max_points, 9),
                   torch.tensor(0).long(),
                   torch.ones(10, dtype=torch.bool),
                   "error_vid")

        seg_len, num_points = data['shape_info']

        # Load and normalize coordinates
        pred_tracks = data['pred_tracks'].view(seg_len, num_points, 2)
        coords = self._normalize_coordinates(pred_tracks)

        # Load visibility
        pred_vis = data['pred_visibility'].view(seg_len, num_points, 1)
        pred_vis = pred_vis.float()

        input_tensor = torch.cat([coords, pred_vis], dim=-1)

        return input_tensor, torch.tensor(label).long(), vid_name

def motion_collate_fn(batch: list) -> tuple:
    features, labels, video_names = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in features])
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    max_len = features_padded.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
    labels_tensor = torch.stack(labels)
    return features_padded, labels_tensor, mask, video_names
