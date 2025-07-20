import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torchinfo import summary
import numpy as np

from DataLoader import MotionDataset, motion_collate_fn
from Model import MotionPointTransformer


# === Config ===
PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/videosTensors1000"
DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/df_exsists.csv"
EPOCHS = 200
BATCH_SIZE = 8
LR = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# === Step 1: Load df and build base dataset ===
df_exists = pd.read_csv(DF_PATH)
base_dataset = MotionDataset(df_exists, PT_FOLDER)

# === Step 2: Detect present classes ===
all_labels = [int(label) for _, label, _ in base_dataset]
present_classes = sorted(set(all_labels))  # e.g., [0, 1, ..., 29, 31, ..., 47]
print("Present classes:", present_classes)

# === Step 3: Build label remap dict ===
class_remap = {old: new for new, old in enumerate(present_classes)}
print("Remap Dict:", class_remap)


# === Step 4: Wrap with RemappedDataset ===
class RemappedDataset(Dataset):
    def __init__(self, original_dataset, remap_dict):
        self.dataset = original_dataset
        self.remap = remap_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, label, name = self.dataset[idx]
        return x, self.remap[int(label)], name

remapped_dataset = RemappedDataset(base_dataset, class_remap)
NUM_CLASSES = len(class_remap)

# === Step 5: Class Weights ===
remapped_labels = [label for _, label, _ in remapped_dataset]
print("Unique labels:", set(remapped_labels))
weights = compute_class_weight(class_weight='balanced', classes=np.arange(NUM_CLASSES), y=remapped_labels)
class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

# === Step 6: Prepare DataLoader ===
dataloader = DataLoader(remapped_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=motion_collate_fn)


model = MotionPointTransformer(num_classes=NUM_CLASSES)
summary(model, input_size=(1, 1000, 32, 3))
model = model.to(DEVICE)
class_weights_tensor = class_weights_tensor.to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

# LR Scheduler: Reduce learning rate on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Model Summary

# === Step 8: Training Loop ===
os.makedirs("plots", exist_ok=True)
# === Training Logging Structures ===
train_losses, train_accuracies, all_preds, all_targets = [], [], [], []
per_class_metrics = []
tp_tn_fp_fn_metrics = []

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    epoch_preds, epoch_targets = [], []
    count = 0
    for inputs, labels, _ in dataloader:
        inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE)
        count += inputs.size(0)  # Count number of samples processed in this batch
        print(f"Processing batch {epoch+1}, count: {count} | Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
        # NaN and label range checks (safety)
        assert not torch.isnan(inputs).any(), "NaNs detected in inputs!"
        assert labels.min() >= 0 and labels.max() < NUM_CLASSES, "Labels out of bounds!"

        optimizer.zero_grad()
        assert inputs.device == model.cls_token.device, f"Input and model not on same device! Input: {inputs.device}, Model: {model.cls_token.device}"
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        epoch_preds.extend(predicted.cpu().numpy())
        epoch_targets.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    train_losses.append(total_loss)
    train_accuracies.append(acc)
    all_preds.extend(epoch_preds)
    all_targets.extend(epoch_targets)

    # Metrics computation
    precision, recall, f1, _ = precision_recall_fscore_support(
        epoch_targets, epoch_preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    per_class_metrics.append({
        'Epoch': epoch + 1,
        **{f'Precision_Class_{i}': p for i, p in enumerate(precision)},
        **{f'Recall_Class_{i}': r for i, r in enumerate(recall)},
        **{f'F1_Class_{i}': f for i, f in enumerate(f1)}
    })

    cm_epoch = confusion_matrix(epoch_targets, epoch_preds, labels=list(range(NUM_CLASSES)))
    for i in range(NUM_CLASSES):
        TP = cm_epoch[i, i]
        FP = cm_epoch[:, i].sum() - TP
        FN = cm_epoch[i, :].sum() - TP
        TN = cm_epoch.sum() - (TP + FP + FN)

        tp_tn_fp_fn_metrics.append({
            'Epoch': epoch + 1,
            'Class': i,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        })

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}  Accuracy: {acc:.2f}%")

    # Step the scheduler using epoch loss
    scheduler.step(total_loss)

# Save Model
torch.save(model.state_dict(), "motion_transformer_optimized.pt")

# Save training metrics
pd.DataFrame({
    'Epoch': list(range(1, EPOCHS + 1)),
    'Loss': train_losses,
    'Accuracy (%)': train_accuracies
}).to_csv("plots/training_metrics.csv", index=False)

# Save precision, recall, F1
pd.DataFrame(per_class_metrics).to_csv("plots/per_class_metrics.csv", index=False)

# Save TP/TN/FP/FN metrics
pd.DataFrame(tp_tn_fp_fn_metrics).to_csv("plots/tp_tn_fp_fn_per_epoch.csv", index=False)

# Final confusion matrix visualization
cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Confusion Matrix (Final Epoch)")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.close()

print("✅ Training complete. Metrics, plots, and confusion matrix saved in 'plots/' folder.")