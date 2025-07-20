# 
import os
import pandas as pd
from ProcessVideoFolder import process_video_folder
import pickle

ROOT = "/Users/mrinalraj/Downloads/WebDownload/Driving48/rgb"
OUTPUT_DIR = "output_videos"
CSV_PATH = os.path.join(OUTPUT_DIR, "video_labels.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH1 = os.path.join(OUTPUT_DIR, "ListAnnotationNeeded.csv")
video_list = []
for i in os.listdir("rgb/"):
    if i.endswith(".mp4"):
        video_list.append(i)
all_segments = []
ListAnnotationNeeded = []
for idx, video in enumerate(video_list):

    print_debug = True

    if print_debug:
        print(f"\n[{idx}] Processing folder: {ROOT} for video: {video}")

    try:
        process_video_folder(ROOT, video, ListAnnotationNeeded)
        all_segments.append((f"{video}", "segments_saved"))
    except Exception as e:
        print(f"Skipping ({video}) due to error: {e}")
        continue

# Save CSV of successfully processed base video names
df = pd.DataFrame(all_segments, columns=["video_name", "status"])
df.to_csv(CSV_PATH, index=False)
df = pd.DataFrame(ListAnnotationNeeded, columns=["ListAnnotationNeeded"])
df.to_csv(CSV_PATH1, index=False)
print(f"\nSaved CSV to {CSV_PATH}")
print(f"\nSaved CSV to {CSV_PATH1}")