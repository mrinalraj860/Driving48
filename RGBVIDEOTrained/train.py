# train_rgb.py
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# collate_rgb.py
import torch

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




from dataLoader import RGBVideoDataset
from model import Simple3DCNN

# === CONFIG ===
DF_PATH = "/notebooks/Driving48/df_exsists.csv"
RGB_FOLDER = "/notebooks/videotensors/rgb"
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", DEVICE)

# === Load DataFrame & Dataset ===
df_exists = pd.read_csv(DF_PATH)

# Only include existing video files
df_exists = df_exists[df_exists['vid_name'].apply(lambda x: os.path.exists(os.path.join(RGB_FOLDER, f"{x}.mp4")))]
df_exists = df_exists.reset_index(drop=True)

labels = df_exists['label'].tolist()
print(df_exists["label"].value_counts())
class_remap = {old: new for new, old in enumerate(sorted(set(labels)))}
df_exists['label'] = df_exists['label'].map(class_remap)
print(df_exists["label"].value_counts())
NUM_CLASSES = len(class_remap)

dataset = RGBVideoDataset(df_exists, RGB_FOLDER)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=rgb_collate_fn)


# === Model, Loss, Optimizer ===
model = Simple3DCNN(num_classes=NUM_CLASSES).to(DEVICE)

weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=df_exists['label'])
weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Training ===
train_losses, train_accuracies = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    preds, targets = [], []
    count = 0
    for inputs, labels, _ in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        print(f"count is : {count}")
        count = count + 8
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        preds.extend(predicted.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    train_losses.append(total_loss)
    train_accuracies.append(acc)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f} Accuracy: {acc:.2f}%")

# === Save Model + Plot ===
torch.save(model.state_dict(), "rgb_3dcnn_model.pt")
pd.DataFrame({'Epoch': range(1, EPOCHS+1), 'Loss': train_losses, 'Accuracy': train_accuracies}).to_csv("rgb_training_metrics.csv", index=False)

cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Confusion Matrix (Final Epoch)")
plt.tight_layout()
plt.savefig("rgb_confusion_matrix.png")
plt.close()
