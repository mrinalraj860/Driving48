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
                               device, scheduler, num_classes, num_epochs, early_stopping=None):

    print("Starting improved training...")
    best_accuracy = 0.0
    best_f1 = 0.0
    train_losses = []
    val_accuracies = []

    os.makedirs("saved_models", exist_ok=True)
    print(f"Starting Epochs")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_correct = 0
        num_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device inside is {device}")
        for i, (features, labels, mask, _) in enumerate(pbar):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            # print(f"Feature shape: {features.shape}, Labels: {labels.shape}, maskL {mask.shape}")
            # print(f"Shape {features.shape}, {labels.shape}, {mask.shape}")
            # Forward pass
            outputs = model(features, mask)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
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
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

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
        print(device)
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

