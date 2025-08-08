import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
from tqdm import tqdm
import numpy as np

from DataLoaderTest import MotionDataset, motion_collate_fn
from Tranformer2 import MotionPointTransformer

# === Paths and Device Config ===
PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/Test"
PROCESSED_TEST_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/ProcessedTestCorrect.csv"
MODEL_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/saved_models_model1/model_epoch_96_acc_90.79.pt"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# === Load and Filter Test Data ===
df_test = pd.read_csv(PROCESSED_TEST_PATH)
print(f"Loaded {len(df_test)} test samples.")

# === Dataset & DataLoader ===
test_dataset = MotionDataset(df_test, PT_FOLDER)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=motion_collate_fn)

# === Class Mapping (Assume same as train-time) ===
unique_labels = sorted(df_test["label"].unique())
class_remap = {old: new for new, old in enumerate(unique_labels)}
inv_class_remap = {v: k for k, v in class_remap.items()}
NUM_CLASSES = len(class_remap)
print("Class remap:", class_remap)
print("Number of classes:", NUM_CLASSES)

# === Remap test labels
df_test["label"] = df_test["label"].map(class_remap)

# === Load Model ===
model = MotionPointTransformer(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully. on Device:", DEVICE)

# === Run Inference ===
all_preds, all_labels, all_names = [], [], []

with torch.no_grad():
    for inputs, labels, names in tqdm(test_loader, desc="Testing", total=len(test_loader)):
        inputs = inputs.to(DEVICE).float()
        # print(f"Input shape: {inputs.shape}")
        labels = labels.to(DEVICE)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_names.extend(names)

# === Save Results ===
df_test["pred_mapped"] = all_preds
df_test["pred_original"] = df_test["pred_mapped"].map(inv_class_remap)
df_test.to_csv("test_results.csv", index=False)

# === Evaluation ===
cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("test_classification_report.csv")

# === Print Accuracy and Metrics ===
acc = accuracy_score(all_labels, all_preds)
print(f"\n✅ Overall Accuracy: {acc:.4f}")

# Print overall metrics safely
print("\n=== Overall Evaluation Metrics ===")
available_keys = df_report.index.tolist()
keys_to_print = [k for k in ["accuracy", "macro avg", "weighted avg", "micro avg"] if k in available_keys]
print(df_report.loc[keys_to_print][["precision", "recall", "f1-score", "support"]])

# Print per-class metrics
print("\n=== Per-Class Evaluation Metrics ===")
exclude_keys = keys_to_print  # accuracy, macro avg, etc.
per_class_metrics = df_report.drop(index=exclude_keys, errors="ignore")
print(per_class_metrics[["precision", "recall", "f1-score", "support"]])

# === Plotting ===
os.makedirs("test_plots", exist_ok=True)

# Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("test_plots/confusion_matrix.png")
plt.close()

# Per-Class Precision
plt.figure(figsize=(12, 5))
per_class_metrics["precision"].plot(kind="bar", title="Precision per Class")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig("test_plots/precision_per_class.png")
plt.close()

# Per-Class Recall
plt.figure(figsize=(12, 5))
per_class_metrics["recall"].plot(kind="bar", title="Recall per Class", color="green")
plt.ylabel("Recall")
plt.tight_layout()
plt.savefig("test_plots/recall_per_class.png")
plt.close()

# Per-Class F1 Score
plt.figure(figsize=(12, 5))
per_class_metrics["f1-score"].plot(kind="bar", title="F1 Score per Class", color="red")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig("test_plots/f1_score_per_class.png")
plt.close()

print("✅ Test complete: Results in `test_results.csv` and plots in `test_plots/`.")