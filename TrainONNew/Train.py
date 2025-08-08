import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# --- Required Dependencies ---
# This script assumes you have two other files in your project:
# 1. Dataloader.py: Contains the MotionDataset class and motion_collate_fn function.
# 2. Model.py: Contains the MotionTransformerClassifier model class.
# Make sure these files are present in the same directory or your Python path.
from Dataloader import MotionDataset, motion_collate_fn
from Model import MotionTransformerClassifier



def train_only(train_loader, model, optimizer, criterion, device, scheduler, num_epochs=100, base_save_dir='./TrainONNew/trainingOnTheModel'):
    """
    Trains the model, saving the model and training loss for each epoch into a structured directory.
    
    Args:
        train_loader (DataLoader): DataLoader for the training set.
        model (nn.Module): The PyTorch model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run the model on (e.g., 'cuda', 'mps', 'cpu').
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): The number of epochs to train for.
        base_save_dir (str): The root directory where epoch folders will be saved.
    """
    # Create the base directory for all training outputs if it doesn't exist
    os.makedirs(base_save_dir, exist_ok=True)
    print(f"Starting training (without validation). All outputs will be saved in the '{base_save_dir}' directory.")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        total_loss = 0
        for features, labels, mask, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            # Move data to the selected device
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features, mask)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step() # OneCycleLR scheduler is updated after each batch

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}], "
              f"Training Loss: {avg_train_loss:.4f}")

        # --- Create epoch-specific directory ---
        epoch_dir = os.path.join(base_save_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_dir, exist_ok=True)

        # --- Save training metrics for the current epoch ---
        metrics_path = os.path.join(epoch_dir, 'training_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"Average Training Loss: {avg_train_loss:.4f}\n")

        # --- Save the model for the current epoch ---
        model_path = os.path.join(epoch_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        
        print(f"Model and metrics for epoch {epoch+1} saved in '{epoch_dir}'")

    # --- Finalization ---
    print(f"\nTraining finished.")


if __name__ == '__main__':
    # ==================== Hyperparameters ====================
    MAX_POINTS = 1000
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 60
    WEIGHT_DECAY = 1e-4

    # Model architecture parameters
    D_MODEL = 256
    N_HEAD = 8
    N_LAYERS = 4
    DROPOUT = 0.3

    # Set device (MPS for Apple Silicon, otherwise CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==================== Data Loading ====================
    # IMPORTANT: Update this path to point to your training data.
    TRAIN_PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/videosTensors1000"
    TRAIN_DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/df_exists1.csv"
    
    print("Loading training data from specified path...")
    try:
        train_df = pd.read_csv(TRAIN_DF_PATH)
    except FileNotFoundError as e:
        print(f"Error: Data file not found at {e.filename}.")
        print("Please update the TRAIN_DF_PATH variable in the script.")
        exit()

    # --- Label Remapping ---
    # Convert string labels to integer indices (0, 1, 2, ...)
    original_labels = sorted(train_df['label'].unique())
    label_mapping = {orig_label: new_label for new_label, orig_label in enumerate(original_labels)}
    train_df['label'] = train_df['label'].map(label_mapping)
    NUM_CLASSES = len(label_mapping)
    print(f"Labels remapped. Total number of classes: {NUM_CLASSES}")
    
    # --- Create Dataset and DataLoader ---
    train_dataset = MotionDataset(train_df, TRAIN_PT_FOLDER, training=True, max_points=MAX_POINTS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=motion_collate_fn, num_workers=4, pin_memory=True)

    # ==================== Model, Optimizer, Loss ====================
    model = MotionTransformerClassifier(
        num_classes=NUM_CLASSES, d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS, dropout=DROPOUT
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                           steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS)

    # --- Class Weights for Imbalanced Datasets ---
    class_counts = train_df['label'].value_counts().sort_index()
    class_counts = class_counts.reindex(range(NUM_CLASSES), fill_value=1) # Ensure all classes are present
    class_weights = len(train_df) / (NUM_CLASSES * class_counts.values)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    print("Criterion created with class weights and label smoothing.")

    # ==================== Run Training Process ====================
    train_only(train_loader, model, optimizer, criterion, device, scheduler, num_epochs=NUM_EPOCHS)

    print("\n--- Next Steps ---")
    print("1. The script has finished training.")
    print("2. Your final trained model is saved as 'final_motion_model.pth'.")

