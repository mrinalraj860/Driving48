import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from dataLoader import RGBVideoDataset
from model import Simple3DCNN
from tqdm import tqdm

# === CONFIG ===
TEST_DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/ProcessedTestCorrect.csv"
RGB_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/rgb"
MODEL_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/RGBVIDEOTrained/rgb_3dcnn_model.pt"
BATCH_SIZE = 8

def rgb_collate_fn(batch):
    videos, labels, names = zip(*batch)
    T_all = [v.shape[0] for v in videos]
    T_avg = round(sum(T_all) / len(T_all))

    processed_videos = []
    for v in videos:
        T = v.shape[0]
        if T > T_avg:
            start = (T - T_avg) // 2
            v = v[start:start + T_avg]
        elif T < T_avg:
            pad = T_avg - T
            pad_front = pad // 2
            pad_back = pad - pad_front
            front_pad = v[0:1].expand(pad_front, -1, -1, -1)
            back_pad = v[-1:].expand(pad_back, -1, -1, -1)
            v = torch.cat([front_pad, v, back_pad], dim=0)
        assert v.shape[0] == T_avg
        processed_videos.append(v)

    # Stack to shape [B, T, 3, H, W] → [B, 3, T, H, W]
    video_tensor = torch.stack(processed_videos).permute(0, 2, 1, 3, 4)
    labels_tensor = torch.tensor(labels)
    return video_tensor, labels_tensor, names


DEVICE = torch.device("cpu")
print("Using device:", DEVICE)

# === Load Test Data ===
df_test = pd.read_csv(TEST_DF_PATH)
df_test = df_test[df_test['vid_name'].apply(lambda x: os.path.exists(os.path.join(RGB_FOLDER, f"{x}.mp4")))]
df_test = df_test.reset_index(drop=True)

# Label remapping
labels = df_test['label'].tolist()
unique_labels = sorted(set(labels))
class_remap = {old: new for new, old in enumerate(unique_labels)}
df_test['label'] = df_test['label'].map(class_remap)
NUM_CLASSES = len(class_remap)

# === Dataset & Loader ===
test_dataset = RGBVideoDataset(df_test, RGB_FOLDER)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=rgb_collate_fn)

# === Load Model ===
model = Simple3DCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Evaluation ===
all_preds, all_labels, all_names = [], [], []

with torch.no_grad():
    for inputs, labels, names in tqdm(test_loader, desc="Evaluating"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_names.extend(names)

# === Save Confusion Matrix ===
cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
cm_df = pd.DataFrame(cm, index=[f"True_{i}" for i in range(NUM_CLASSES)],
                        columns=[f"Pred_{i}" for i in range(NUM_CLASSES)])
cm_df.to_csv("rgb_confusion_matrix.csv")

# === Per-Class Accuracy ===
# === Per-Class Metrics: TP, FP, FN, TN, Accuracy ===
per_class_metrics = []
total_samples = len(all_labels)

for class_id in range(NUM_CLASSES):
    TP = cm[class_id, class_id]
    FP = cm[:, class_id].sum() - TP
    FN = cm[class_id, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    support = cm[class_id, :].sum()
    pred_total = cm[:, class_id].sum()
    accuracy = TP / support * 100 if support > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    per_class_metrics.append({
        "Class": class_id,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Support": support,
        "Predicted": pred_total,
        "Accuracy (%)": round(accuracy, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
    })

metrics_df = pd.DataFrame(per_class_metrics)
metrics_df.to_csv("rgb_per_class_metrics.csv", index=False)

# === Classification Report (Optional to print) ===
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, zero_division=0, labels=list(range(NUM_CLASSES))))