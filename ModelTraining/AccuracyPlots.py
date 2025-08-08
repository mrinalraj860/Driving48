import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from DataLoader import MotionDataset, motion_collate_fn
# from PointTemporalGRU import PointTemporalGRU  # replace with your model
from CNNGRUClassifier import CNNGRUClassifier
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight

# === Configuration ===
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "CNNGRUClassifier"
SAVE_DIR = "TrainCNNGRU"
PLOT_DIR = f"plots_{MODEL_NAME}"
os.makedirs(PLOT_DIR, exist_ok=True)

TRAIN_PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/videosTensors1000"
TRAIN_DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/df_exsists.csv"
BATCH_SIZE = 8

# === Load Dataset ===
train_df = pd.read_csv(TRAIN_DF_PATH)
train_dataset = MotionDataset(train_df, TRAIN_PT_FOLDER)
train_labels = [int(label) for _, label, _ in train_dataset]
present_classes = sorted(set(train_labels))
class_remap = {old: new for new, old in enumerate(present_classes)}

class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, remap):
        self.dataset = dataset
        self.remap = remap

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, label, name = self.dataset[idx]
        return x, self.remap[int(label)], name

remapped_train_dataset = RemappedDataset(train_dataset, class_remap)
train_loader = DataLoader(remapped_train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=motion_collate_fn)
NUM_CLASSES = len(class_remap)

# === Loss Function with Class Weights ===
weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES),
                               y=[label for _, label, _ in remapped_train_dataset])
class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
criterion = CrossEntropyLoss(weight=class_weights_tensor)

# === Evaluate All Saved Models ===
records = []

epochs_found = sorted([int(d.split('_')[-1]) for d in os.listdir(SAVE_DIR) if d.startswith('Epoch_')])
for epoch in epochs_found:
    model_path = os.path.join(SAVE_DIR, f"Epoch_{epoch}", f"model_epoch_{epoch}.pt")
    if not os.path.exists(model_path):
        continue

    model = CNNGRUClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    correct, total, total_loss = 0, 0, 0
    with torch.no_grad():
        for inputs, labels, _ in tqdm(train_loader, desc=f"Evaluating Epoch {epoch}"):
            inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    train_acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    records.append({'epoch': epoch, 'accuracy': train_acc, 'loss': avg_loss})
    print(f"Epoch {epoch}: Accuracy={train_acc:.2f}%, Loss={avg_loss:.4f}")

# === Save to CSV ===
df = pd.DataFrame(records)
df.to_csv(os.path.join(PLOT_DIR, "metrics.csv"), index=False)

# === Plot Accuracy ===
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['accuracy'], marker='o', label='Train Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Recovered Train Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "train_accuracy.png"))
plt.close()

# === Plot Loss ===
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['loss'], marker='o', color='red', label='Train Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Recovered Train Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "train_loss.png"))
plt.close()

print(f"\n✅ Train accuracy/loss recovered and saved in: {PLOT_DIR}")