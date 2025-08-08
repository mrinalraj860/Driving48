# rgb_dataloader.py
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

class RGBVideoDataset(Dataset):
    def __init__(self, df_exists, rgb_folder, image_size=112):
        self.df = df_exists.reset_index(drop=True)
        self.rgb_folder = rgb_folder
        self.resize = T.Resize((image_size, image_size))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_name = row['vid_name']
        label = int(row['label'])

        video_path = os.path.join(self.rgb_folder, f"{vid_name}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"{video_path} not found.")

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            frame = self.resize(frame)
            frames.append(frame)
        cap.release()

        video_tensor = torch.stack(frames)  # [T, 3, H, W]
        # print(video_tensor.shape)
        return video_tensor, torch.tensor(label).long(), vid_name

    
    