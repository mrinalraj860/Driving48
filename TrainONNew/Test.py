import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

# --- Required Dependencies ---
# This script assumes you have two other files in your project:
# 1. Dataloader.py: Contains the MotionDataset class and motion_collate_fn function.
# 2. Model.py: Contains the MotionTransformerClassifier model class.
# Make sure these files are present in the same directory or your Python path.
from Dataloader import MotionDataset, motion_collate_fn
from Model import MotionTransformerClassifier

def test_model(test_loader, model, device, num_classes):
    """
    Evaluates the model on the test set and prints accuracy metrics.
    
    Args:
        test_loader (DataLoader): DataLoader for the test set.
        model (nn.Module): The trained PyTorch model to be evaluated.
        device (torch.device): The device to run the model on.
        num_classes (int): The total number of classes for classification.
    """
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)

    print("Starting evaluation on the test set...")
    with torch.no_grad():
        for features, labels, mask, _ in tqdm(test_loader, desc="Evaluating"):
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            outputs = model(features, mask)
            _, predicted = torch.max(outputs.data, 1)

            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    # --- Calculate and Display Metrics ---
    total_correct = confusion_matrix.diag().sum().item()
    total_samples = confusion_matrix.sum().item()
    accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0

    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")

    # Calculate and display per-class accuracy
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    per_class_accuracy = torch.nan_to_num(per_class_accuracy) # Handle classes with no samples in the test set

    print("Per-class Test Accuracy:")
    for i, acc in enumerate(per_class_accuracy):
        print(f"  Class {i:2d}: {acc.item() * 100:6.2f}%", end='\t' if (i+1) % 5 != 0 else '\n')
    print("\n")


if __name__ == '__main__':
    # --- Argument Parser ---
    # This allows you to specify the epoch to test from the command line.
    # Example usage: python test.py --epoch 1
    parser = argparse.ArgumentParser(description="Test a trained Motion Transformer model.")
    parser.add_argument('--epoch', type=int, required=True, help='The epoch number of the model to test.')
    args = parser.parse_args()

    # ==================== Configuration ====================
    # --- Model Architecture (must match the trained model) ---
    D_MODEL = 256
    N_HEAD = 8
    N_LAYERS = 4
    DROPOUT = 0.3
    
    # --- Data and Path Configuration ---
    MAX_POINTS = 1000
    BATCH_SIZE = 32
    # IMPORTANT: Update these paths to point to your data files.
    # The training DF is needed to ensure the label mapping is identical.
    TRAIN_PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/videosTensors1000"
    TRAIN_DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/df_exists1.csv"
    TEST_PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/Test"
    TEST_DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/ProcessedTestCorrect.csv"
    BASE_MODEL_DIR = 'trainingOnTheModel' # The directory where models were saved

    # Set device
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # ==================== Data Loading and Preparation ====================
    print("Loading data...")
    try:
        train_df = pd.read_csv(TRAIN_DF_PATH)
        test_df = pd.read_csv(TEST_DF_PATH)
    except FileNotFoundError as e:
        print(f"Error: Data file not found at {e.filename}.")
        print("Please update the data paths in the script.")
        exit()

    # --- Recreate the Exact Same Label Remapping ---
    # This is critical for the model to understand the labels correctly.
    original_labels = sorted(train_df['label'].unique())
    label_mapping = {orig_label: new_label for new_label, orig_label in enumerate(original_labels)}
    
    # Apply mapping to the test dataframe
    test_df['label'] = test_df['label'].map(label_mapping)
    # Drop rows where the label might not have been in the training set
    test_df.dropna(subset=['label'], inplace=True)
    test_df['label'] = test_df['label'].astype(int)

    NUM_CLASSES = len(label_mapping)
    print(f"Labels remapped based on training data. Total classes: {NUM_CLASSES}")

    # --- Create Test Dataset and DataLoader ---
    test_dataset = MotionDataset(test_df, TEST_PT_FOLDER, training=False, max_points=MAX_POINTS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=motion_collate_fn, num_workers=4, pin_memory=True)

    # ==================== Model Loading ====================
    # --- Construct the path to the saved model ---
    model_path = os.path.join(BASE_MODEL_DIR, f'epoch_{args.epoch}', f'model_epoch_{args.epoch}.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print("Please ensure the epoch number is correct and the model directory exists.")
        exit()
        
    print(f"Loading model from: {model_path}")

    # --- Initialize and load the model ---
    model = MotionTransformerClassifier(
        num_classes=NUM_CLASSES, d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS, dropout=DROPOUT
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))

    # ==================== Run Testing Process ====================
    test_model(test_loader, model, device, NUM_CLASSES)
