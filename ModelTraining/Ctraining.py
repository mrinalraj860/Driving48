import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary

from DataLoaderTrans import MotionDataset, motion_collate_fn
from TemporalCnnClasifier import TemporalCNNClassifier

# === Configuration ===
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")
LR = 1e-4

TRAIN_PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/videosTensors1000"
TRAIN_DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/df_exsists.csv"
TEST_PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/Test"
TEST_DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/ProcessedTestCorrect.csv"

PREV_SAVE_DIR = "TrainingVelocityTemporalCNNClassifier"
SAVE_DIR = "TrainingAgain"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Dataset Preparation ===
train_df = pd.read_csv(TRAIN_DF_PATH)
train_dataset = MotionDataset(train_df, TRAIN_PT_FOLDER)
train_labels = [int(label) for _, label, _ in train_dataset]
present_classes = sorted(set(train_labels))
class_remap = {old: new for new, old in enumerate(present_classes)}

class RemappedDataset(Dataset):
    def __init__(self, dataset, remap):
        self.dataset = dataset
        self.remap = remap

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, label, name = self.dataset[idx]
        return x, self.remap[int(label)], name

remapped_train_dataset = RemappedDataset(train_dataset, class_remap)
NUM_CLASSES = len(class_remap)

weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES),
                               y=[label for _, label, _ in remapped_train_dataset])
class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

train_loader = DataLoader(remapped_train_dataset, batch_size=8, shuffle=True, collate_fn=motion_collate_fn, batch_size=1, num_workers=0)

test_df = pd.read_csv(TEST_DF_PATH)
test_df = test_df[test_df["label"].isin(present_classes)].reset_index(drop=True)
test_df["label"] = test_df["label"].map(class_remap)
test_dataset = MotionDataset(test_df, TEST_PT_FOLDER)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=motion_collate_fn, batch_size=1, num_workers=0)

# === Load Most Recent Model ===
def get_latest_model_path(base_dir):
    epoch_folders = [d for d in os.listdir(base_dir) if d.startswith("Epoch_")]
    epoch_numbers = sorted([int(d.split("_")[1]) for d in epoch_folders])
    if not epoch_numbers:
        raise FileNotFoundError("No previous model found to resume.")
    latest_epoch = epoch_numbers[-1]
    model_path = os.path.join(base_dir, f"Epoch_{latest_epoch}", f"model_epoch_{latest_epoch}.pt")
    print(f"Latest model path: {model_path}")
    return model_path, latest_epoch

model = TemporalCNNClassifier(
    num_classes=NUM_CLASSES
).to(DEVICE)
# summary(model, input_size=(1, 1000, 8, 2), device=DEVICE)

latest_model_path, last_epoch = get_latest_model_path(PREV_SAVE_DIR)
model.load_state_dict(torch.load(latest_model_path))
print(f"Resumed from: {latest_model_path}")

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor,
    label_smoothing=0.1
)

# optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4) # TemporalCNNClassifier
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) # RobustPointTransformer

# OneCycle learning rate scheduler
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=20,
    steps_per_epoch=len(train_loader),
    pct_start=0.3
)

# === Continue Training for 20 Epochs ===
for epoch_offset in range(1, 21):
    epoch = last_epoch + epoch_offset
    print(f"\nEpoch [{epoch}]")
    epoch_dir = os.path.join(SAVE_DIR, f"Epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels, masks, _ in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        # print(f"Input shape: {inputs.shape}")
        # print(f"Labels shape: {labels.shape}")
        # print(f"Masks shape: {masks.shape}")
        inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE)
        masks = masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs, mask=masks)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        # --- Scheduler Step ---
        scheduler.step()

    train_acc = 100 * correct / total
    train_loss = total_loss / len(train_loader)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

    # --- Testing ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels, masks, _ in tqdm(test_loader, desc=f"Testing Epoch {epoch}"):
            inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE)
            masks = masks.to(DEVICE)
            # print(f"Test Input shape: {inputs.shape}")
            # print(f"Test Labels shape: {labels.shape}")
            # print(f"Test Masks shape: {masks.shape}")
            outputs = model(inputs, mask=masks)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {test_acc:.2f}%")

    # --- Save Classification Metrics ---
    report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(epoch_dir, "classification_report.csv"))


    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(xticks_rotation=45, cmap='Blues', ax=ax)
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.savefig(os.path.join(epoch_dir, "confusion_matrix.png"))
    plt.close()

    per_class_correct = np.zeros(NUM_CLASSES)
    per_class_total = np.zeros(NUM_CLASSES)
    for label, pred in zip(all_labels, all_preds):
        per_class_total[label] += 1
        if label == pred:
            per_class_correct[label] += 1

    per_class_accuracy = per_class_correct / np.maximum(per_class_total, 1)
    acc_df = pd.DataFrame({
        "Class": [f"Class_{i}" for i in range(NUM_CLASSES)],
        "Accuracy": per_class_accuracy,
        "Total_Samples": per_class_total.astype(int),
        "Correct_Predictions": per_class_correct.astype(int)
    })
    acc_df.to_csv(os.path.join(epoch_dir, "per_class_accuracy.csv"), index=False)

    print("\nPer-class Accuracy:")
    for idx, acc in enumerate(per_class_accuracy):
        print(f"Class {idx}: {acc*100:.2f}% ({int(per_class_correct[idx])}/{int(per_class_total[idx])})")

    # Save model
    torch.save(model.state_dict(), os.path.join(epoch_dir, f"model_epoch_{epoch}.pt"))
    scheduler.step(train_loss)
    print(f"Epoch {epoch} completed. Model saved to {epoch_dir}")

print("\n✅ Continued Training Complete.")