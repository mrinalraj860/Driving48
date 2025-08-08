import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from DataloaderVelocity import MotionDataset, motion_collate_fn
# from Model import MotionTransformer   # your model
# from SimplePointVelocity import TemporalCNNClassifier  # your model
from EnhancedTenporal import EnhancedTemporalCNNClassifier  # your model

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# === Config ===
TRAIN_DF_PATH = "df_exists_with_som.csv"
PT_FOLDER = "videosTensors1000"
SAVE_DIR = "Training_SOM"
TEST_PT_FOLDER = "Test"
TEST_DF_PATH = "df_exists_with_som_test.csv"
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-4



# === Load and Map SOM Labels Globally ===
train_df = pd.read_csv(TRAIN_DF_PATH)

# === Train/Test Split ===
test_df = pd.read_csv(TEST_DF_PATH)

# === Load and Map SOM Labels Globally ===
full_df = pd.concat([train_df, test_df])
unique_soms = sorted(full_df["1"].unique())
som2idx = {som: idx for idx, som in enumerate(unique_soms)}
idx2som = {idx: som for som, idx in som2idx.items()}

# === Create Datasets ===
train_dataset = MotionDataset(train_df, PT_FOLDER, som2idx)
test_dataset = MotionDataset(test_df, TEST_PT_FOLDER, som2idx, training=False)

# === Compute Class Weights ===
train_labels = [int(label) for _, label, _ in train_dataset]
test_labels = [int(label) for _, label, _ in test_dataset]
NUM_CLASSES = len(som2idx)

print(f"Number of classes: {NUM_CLASSES}")

classes_in_y = np.unique(train_labels)
print(f"Classes in y: {classes_in_y}")
weights = compute_class_weight('balanced', classes=classes_in_y, y=train_labels)

# Create full weight tensor with default = 1.0
weight_tensor = torch.ones(NUM_CLASSES, dtype=torch.float32)
weights = np.clip(weights, a_min=0.1, a_max=50)
# Set the valid weights
for cls, w in zip(classes_in_y, weights):
    weight_tensor[cls] = w

class_weights_tensor = weight_tensor.to(DEVICE)
print(f"Clipped Class weights: {class_weights_tensor}")

# === Dataloaders ===
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=motion_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=motion_collate_fn)

# === Model, Loss, Optimizer, Scheduler ===
model = EnhancedTemporalCNNClassifier(
    num_classes=NUM_CLASSES,
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader))

# === Training Loop ===
train_losses, train_accuracies, test_accuracies = [], [], []

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch [{epoch}/{EPOCHS}]")

    # --- Always run the training part ---
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels, masks, _ in tqdm(train_loader, desc="Training"):
        inputs, labels, masks = inputs.to(DEVICE).float(), labels.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs, mask=masks)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_loss = total_loss / len(train_loader)
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # Always track training stats
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Use a placeholder for test accuracy. It will be overwritten if testing runs.
    current_test_acc = np.nan
    
    # --- Conditionally run testing and saving ---
    if epoch % 5 == 0 or epoch == EPOCHS:
        print(f"--- Running evaluation for epoch {epoch} ---")
        epoch_dir = os.path.join(SAVE_DIR, f"Epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels, masks, _ in tqdm(test_loader, desc="Testing"):
                inputs, labels, masks = inputs.to(DEVICE).float(), labels.to(DEVICE), masks.to(DEVICE)
                preds = model(inputs).argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        current_test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"Test Accuracy: {current_test_acc:.2f}%")

        # === Save Metrics & Model ===
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(os.path.join(epoch_dir, "classification_report.csv"))
        
        df_report = pd.DataFrame(report).transpose()
        class_rows = [c for c in df_report.index if c.isdigit()]
        df_report = df_report.loc[class_rows]
        df_report.index = df_report.index.astype(int)
        df_report.sort_index(inplace=True)
        df_report["TP"] = (df_report["recall"] * df_report["support"]).round().astype(int)
        df_report = df_report.rename(columns={
            "precision": "Precision", "recall": "Recall",
            "f1-score": "F1", "support": "Support"
        })

        print("Per-Class Test Metrics:")
        print(df_report[["TP", "Support", "Precision", "Recall", "F1"]])

        df_report.to_csv(os.path.join(epoch_dir, "per_class_metrics.csv"))
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(xticks_rotation=45, cmap='Blues')
        plt.title(f"Confusion Matrix - Epoch {epoch}")
        plt.savefig(os.path.join(epoch_dir, "confusion_matrix.png"))
        plt.close()

        torch.save(model.state_dict(), os.path.join(epoch_dir, f"model_epoch_{epoch}_acc_{current_test_acc:.2f}.pt"))

    # Always track test accuracy (will be NaN if not an evaluation epoch)
    test_accuracies.append(current_test_acc)

# === Final Metrics ===
pd.DataFrame({
    "Epoch": list(range(1, EPOCHS + 1)),
    "Train_Loss": train_losses,
    "Train_Accuracy": train_accuracies,
    "Test_Accuracy": test_accuracies
}).to_csv(os.path.join(SAVE_DIR, "training_metrics.csv"), index=False)

print("Training complete.")