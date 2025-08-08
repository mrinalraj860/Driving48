# improved_training.py - Improved training script with better hyperparameters and techniques

import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import warnings


warnings.filterwarnings('ignore')

# Import your existing dataloader and the improved model
from Dataloader import MotionDataset, motion_collate_fn
from Model import MotionTransformerClassifier, SimpleMotionClassifier

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def calculate_class_weights(train_df, num_classes, power=0.5):
    """Calculate class weights with smoothing"""
    class_counts = train_df['label'].value_counts().sort_index()
    total_samples = len(train_df)
    
    # Calculate weights with smoothing
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # Use 1 if class doesn't exist
        weight = (total_samples / (num_classes * count)) ** power
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights / weights.sum() * num_classes  # Normalize

def train_and_evaluate_improved(train_loader, val_loader, model, optimizer, criterion, 
                               device, scheduler, num_classes, num_epochs, 
                               accumulation_steps=1, early_stopping=None):
    
    print("Starting improved training...")
    best_accuracy = 0.0
    best_f1 = 0.0
    train_losses = []
    val_accuracies = []
    
    os.makedirs("saved_models", exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_correct = 0
        num_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (features, labels, mask, _) in enumerate(pbar):
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            # Forward pass
            outputs = model(features, mask)
            loss = criterion(outputs, labels)
            
            # Backward pass with gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            num_samples += labels.size(0)
            num_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100 * num_correct / num_samples
            pbar.set_postfix({
                'Loss': f'{total_loss/(i+1):.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * num_correct / num_samples
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels, mask, _ in tqdm(val_loader, desc="Validating"):
                features = features.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                
                outputs = model(features, mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_samples += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_samples
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
        
        # Log results
        print(f"\nEpoch [{epoch+1}/{num_epochs}]:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_f1 > best_f1):
            best_accuracy = val_accuracy
            best_f1 = val_f1
            save_path = os.path.join("saved_models", 'best_motion_classifier_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_accuracy': best_accuracy,
                'best_f1': best_f1,
            }, save_path)
            print(f"🎉 New best model saved! Acc: {best_accuracy:.2f}%, F1: {best_f1:.2f}%")
        
        # Early stopping
        if early_stopping and early_stopping(avg_val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)
        print("="*80)
    
    print(f"Training finished. Best validation accuracy: {best_accuracy:.2f}%, Best F1: {best_f1:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return best_accuracy, best_f1

if __name__ == '__main__':
    # Improved hyperparameters
    BATCH_SIZE = 4  # Reduced batch size for better gradient estimates
    ACCUMULATION_STEPS = 8  # Effective batch size = 16 * 8 = 128
    LEARNING_RATE = 3e-4  # Slightly higher learning rate
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-5  # Reduced weight decay
    
    # Model hyperparameters (smaller for stability)
    D_MODEL = 256  # Reduced from 512
    N_HEAD = 8
    N_LAYERS = 4  # Reduced from 6
    DROPOUT = 0.2  # Reduced dropout
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths (update these to your paths)
    TRAIN_PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/videosTensors1000"
    TRAIN_DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/df_exists1.csv"
    TEST_PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/Test"
    TEST_DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/ProcessedTestCorrect.csv"
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_DF_PATH)
    val_df = pd.read_csv(TEST_DF_PATH)
    
    # Label mapping
    original_labels = sorted(pd.concat([train_df['label'], val_df['label']]).unique())
    label_mapping = {orig_label: i for i, orig_label in enumerate(original_labels)}
    train_df['label'] = train_df['label'].map(label_mapping)
    val_df['label'] = val_df['label'].map(label_mapping).dropna().astype(int)
    NUM_CLASSES = len(label_mapping)
    print(f"Number of classes: {NUM_CLASSES}")
    
    # Check class distribution
    print("Class distribution in training set:")
    print(train_df['label'].value_counts().sort_index())
    
    # Create weighted sampler with less aggressive resampling
    print("Creating balanced sampler...")
    class_counts = train_df['label'].value_counts().sort_index()
    class_weights = calculate_class_weights(train_df, NUM_CLASSES, power=0.3)  # Less aggressive
    print(train_df['label'].map(lambda x: class_weights[x]).unique())
    print(train_df['label'].map(lambda x: class_weights[x]).apply(type).unique())
    sample_weights = train_df['label'].map(lambda x: class_weights[x]).astype(float).to_numpy()
    sample_weights = torch.from_numpy(sample_weights).float()
    train_sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    # Create datasets and dataloaders
    train_dataset = MotionDataset(train_df, TRAIN_PT_FOLDER, training=True)
    val_dataset = MotionDataset(val_df, TEST_PT_FOLDER, training=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,
        collate_fn=motion_collate_fn, 
        num_workers=10,  # Reduced for stability
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE * 2,  # Larger batch size for validation
        shuffle=False, 
        collate_fn=motion_collate_fn, 
        num_workers=10,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Try both models - start with the simpler one
    print("Testing Simple Model first...")
    
    # Simple model
    # simple_model = SimpleMotionClassifier(
    #     num_classes=NUM_CLASSES,
    #     input_dim=9,
    #     hidden_dim=256,
    #     dropout=DROPOUT
    # ).to(device)
    
    # Complex model (as alternative)
    complex_model = MotionTransformerClassifier(
        num_classes=NUM_CLASSES,
        input_dim=9,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Choose model (start with simple)
    model = complex_model
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Use CosineAnnealingWarmRestarts for better convergence
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,
        eta_min=LEARNING_RATE * 0.01
    )
    
    # Loss function with class weights
    class_weights_loss = calculate_class_weights(train_df, NUM_CLASSES, power=0.25)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_loss.to(device),
        label_smoothing=0.05  # Reduced label smoothing
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    # Train the model
    print("Starting training...")
    best_acc, best_f1 = train_and_evaluate_improved(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        num_classes=NUM_CLASSES,
        num_epochs=NUM_EPOCHS,
        accumulation_steps=ACCUMULATION_STEPS,
        early_stopping=early_stopping
    )
    
    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Best Validation F1 Score: {best_f1:.2f}%")