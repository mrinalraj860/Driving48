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
MODEL_DIR = "/Users/mrinalraj/Downloads/WebDownload/Driving48/saved_models_model1"
RESULT_DIR = "/Users/mrinalraj/Downloads/WebDownload/Driving48/saved_model_result1"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# === Load Test Data ===
df_test = pd.read_csv(PROCESSED_TEST_PATH)
unique_labels = sorted(df_test["label"].unique())
class_remap = {old: new for new, old in enumerate(unique_labels)}
inv_class_remap = {v: k for k, v in class_remap.items()}
NUM_CLASSES = len(class_remap)
df_test["label"] = df_test["label"].map(class_remap)
print(f"Loaded {len(df_test)} test samples. Classes: {NUM_CLASSES}")

# === Dataset & DataLoader ===
test_dataset = MotionDataset(df_test, PT_FOLDER)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=motion_collate_fn)

# === Iterate through saved models ===
os.makedirs(RESULT_DIR, exist_ok=True)
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]

for model_file in model_files:
    print(f"\n🔍 Evaluating model: {model_file}")
    model_path = os.path.join(MODEL_DIR, model_file)
    result_subdir = os.path.join(RESULT_DIR, os.path.splitext(model_file)[0])
    os.makedirs(result_subdir, exist_ok=True)

    # === Load Model ===
    model = MotionPointTransformer(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # === Inference ===
    all_preds, all_labels, all_names = [], [], []

    with torch.no_grad():
        for inputs, labels, names in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(DEVICE).float()
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_names.extend(names)

    # === Save Predictions ===
    df_output = df_test.copy()
    df_output["pred_mapped"] = all_preds
    df_output["pred_original"] = df_output["pred_mapped"].map(inv_class_remap)
    df_output.to_csv(os.path.join(result_subdir, "test_results.csv"), index=False)

    # === Metrics ===
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(result_subdir, "classification_report.csv"))

    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ Accuracy: {acc:.4f}")

    # === Plots ===
    plot_dir = os.path.join(result_subdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
    plt.close()

    # Per-class plots
    keys_to_skip = ["accuracy", "macro avg", "weighted avg"]
    per_class_metrics = df_report.drop(index=keys_to_skip, errors="ignore")

    for metric, color in zip(["precision", "recall", "f1-score"], ["blue", "green", "red"]):
        plt.figure(figsize=(12, 5))
        per_class_metrics[metric].plot(kind="bar", title=f"{metric.title()} per Class", color=color)
        plt.ylabel(metric.title())
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{metric}_per_class.png"))
        plt.close()

print("\n✅ All models evaluated. Results saved in:", RESULT_DIR)