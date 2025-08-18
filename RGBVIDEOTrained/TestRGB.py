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


'''Using device: cpu
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 418/418 [53:07<00:00,  7.63s/it]

Classification Report:

              precision    recall  f1-score   support

           0       0.00      0.00      0.00        60
           1       0.04      0.03      0.03        40
           2       0.00      0.00      0.00        15
           3       0.11      0.67      0.19        57
           4       0.00      0.00      0.00         5
           5       0.12      0.51      0.19       129
           6       0.15      0.13      0.14        45
           7       0.00      0.00      0.00       162
           8       0.12      0.05      0.07       113
           9       0.03      0.10      0.04        20
          10       0.00      0.00      0.00         9
          11       0.00      0.00      0.00        39
          12       0.13      0.03      0.05       106
          13       0.00      0.00      0.00         3
          14       0.02      0.27      0.03        15
          15       0.12      0.05      0.07       188
          16       0.00      0.00      0.00         9
          17       0.05      0.02      0.03       123
          18       0.15      0.30      0.20        20
          19       0.00      0.00      0.00       160
          20       0.11      0.22      0.14         9
          21       0.11      0.03      0.05       165
          22       0.10      0.02      0.03        55
          23       0.00      0.00      0.00        11
          24       0.03      0.04      0.03       127
          25       0.00      0.00      0.00        10
          26       0.20      0.04      0.06       209
          27       0.20      0.02      0.03        66
          28       0.67      0.03      0.06        63
          29       0.00      0.00      0.00        41
          30       0.09      0.44      0.15       145
          31       0.00      0.00      0.00         8
          32       0.03      0.09      0.05        76
          33       0.12      0.03      0.05       135
          34       0.00      0.00      0.00       114
          35       0.07      0.01      0.02        93
          36       0.00      0.00      0.00        15
          37       0.01      0.04      0.02        28
          38       0.00      0.00      0.00        13
          39       0.00      0.00      0.00         8
          40       0.07      0.20      0.10        25
          41       0.00      0.00      0.00        15
          42       0.04      0.01      0.02        67
          43       0.07      0.04      0.05       163
          44       0.12      0.03      0.04       113
          45       0.16      0.08      0.11       239
          46       0.00      0.00      0.00         6

    accuracy                           0.08      3337
   macro avg       0.07      0.08      0.04      3337
weighted avg       0.10      0.08      0.06      3337'''