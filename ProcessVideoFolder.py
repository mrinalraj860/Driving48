
import pickle
from collections import defaultdict
from ultralytics import YOLO
import os
import torch
import numpy as np
from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor

from Helper import *
# Initialize global counter for failed person detections
Count = 0
ListWithManualAnnotationNeeded = []
DEVICE='cpu'
SAVE_DIR = "./videosTensors1000"  # Directory to save the tensor segments
folder_path = "Temp/frames"   # Your preprocessed frame folder
temp_video_path = "Temp/temp.mp4"
FRAME_SIZE = (512, 384)     # (width, height)
FPS = 30


def process_video_folder(folder_path, video_name, ListWithManualAnnotationNeeded):
    global Count
    cap = cv2.VideoCapture(os.path.join(folder_path, video_name ))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, frame = cap.read()
    video = read_video_from_path(os.path.join(folder_path, video_name), (512, 384))
    print(f"Processing video: {video_name} | Video shape: {video.shape}")
    
    frame_resized = cv2.resize(frame, FRAME_SIZE)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    print(f"Resized frame shape: {frame_rgb.shape}")
    frameLength = video.shape[0]

    if frameLength >180:
        print(f"❌ Video {video_name} has more than 200 frames. Manual annotation needed.")
        ListWithManualAnnotationNeeded.append(video_name)
        return
    
    best_box = get_best_person_box(frame_rgb)
    if best_box is None:
        print(f"❌ No person detected in the first frame of {video_name}. Manual annotation needed.")
        ListWithManualAnnotationNeeded.append(video_name)
        return
    query_metrics, annotated_frame = get_query_points(frame_rgb.copy(), best_box)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    pred_tracks, pred_visibility = cotracker(
        video_tensor, 
        queries=query_metrics[None] 
    )
    print(f"Shapes of each pred_tracks {pred_tracks.shape}, {pred_visibility.shape} pred_visibility")
    vis = Visualizer(save_dir='./FullAnnotated1000', linewidth=1, mode='cool', tracks_leave_trace=0)
    vis.visualize(video=video_tensor, tracks=pred_tracks, visibility=pred_visibility, filename=f'queries_{video_name.split(".")[0]}');
    seg_len = pred_tracks.shape[1]
    num_points = pred_tracks.shape[2]
    segment_data = {
        "pred_tracks": pred_tracks[0].reshape(-1),
        "pred_visibility": pred_visibility[0].reshape(-1),
        "shape_info" : (seg_len, num_points),
    }
    segment_name = video_name.split('.')[0]  # Use the video name without extension
    save_path = os.path.join(SAVE_DIR, f"{segment_name}_tracking.pt")
    torch.save(segment_data, save_path)
    
    print(f"✅ Saved tensor segment | seg_len {seg_len} | num_points {num_points} | File: {save_path}")



if __name__ == "__main__":
    ROOT = "/Users/mrinalraj/Downloads/WebDownload/Driving48/rgb"
    OUTPUT_DIR = "output_videos"
    CSV_PATH = os.path.join(OUTPUT_DIR, "video_labels.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    process_video_folder(ROOT, "LkluZoNfKu8_00071.mp4", [])
