from torch.utils.data import DataLoader, Dataset
class RemappedDataset(Dataset):
    def __init__(self, dataset, remap):
        self.dataset = dataset
        self.remap = remap

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, label, name = self.dataset[idx]
        return x, self.remap[int(label)], name

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import os
    import pandas as pd
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.utils.class_weight import compute_class_weight
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from torchinfo import summary

    # from DataLoader import MotionDataset, motion_collate_fn
    from DataLoaderTrans import MotionDataset, motion_collate_fn
    # from CNNGRUClassifier import CNNGRUClassifier
    # from CNNBasic import PointCNNPlusPlus
    # from ModelSeek import FastSTGCN
    # from Tranformer2 import MotionPointTransformer
    # from PointTemporalGRU import PointTemporalGRU
    # from SpatioTemoral import MotionTransformer
    # from MotionPointTranformer import MotionTransformer
    # from SimpleModelVelocity import MotionTransformer
    from PointLstm import RobustPointTransformer
    # from TemporalCnnClasifier import TemporalCNNClassifier
    # from ConvTranformer import ConvTransformer Not used in this context

    # === Configuration ===
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")

    TRAIN_PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/videosTensors1000"
    TRAIN_DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/df_exsists.csv"
    TEST_PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/Test"
    TEST_DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/ProcessedTestCorrect.csv"

    EPOCHS = 50
    BATCH_SIZE = 1
    LR = 1e-3 # CNNGRUClassifier
    # LR = 1e-3 PointTemporalGRU
    # LR = 1e-4  # For PointTransformerClassifier
    # LR = 5e-4  # For RobustPointTransformer

    SAVE_DIR = "Training"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # === Prepare Training Dataset ===
    train_df = pd.read_csv(TRAIN_DF_PATH)
    # train_dataset = MotionDataset(train_df, TRAIN_PT_FOLDER, training=True)
    train_dataset = MotionDataset(train_df, TRAIN_PT_FOLDER)
    train_labels = [int(label) for _, label, _ in train_dataset]
    present_classes = sorted(set(train_labels))
    class_remap = {old: new for new, old in enumerate(present_classes)}
    print(f"The Test df label is {train_df['label'].unique()}, {train_df['label'].value_counts()}")

    remapped_train_dataset = RemappedDataset(train_dataset, class_remap)
    NUM_CLASSES = len(class_remap)
    print(f"Number of classes after remapping: {NUM_CLASSES}")

    weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES),
                                y=[label for _, label, _ in remapped_train_dataset])
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(remapped_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=motion_collate_fn)

    # === Prepare Test Dataset ===
    test_df = pd.read_csv(TEST_DF_PATH)
    test_df = test_df[test_df["label"].isin(present_classes)].reset_index(drop=True)
    test_df["label"] = test_df["label"].map(class_remap)
    # test_dataset = MotionDataset(test_df, TEST_PT_FOLDER, training=False)
    test_dataset = MotionDataset(test_df, TEST_PT_FOLDER)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=motion_collate_fn)

    print(f"The Test df label is {test_df['label'].unique()}, {test_df['label'].value_counts()}")


    model = RobustPointTransformer(
        input_dim=5,
        d_model=128,         # Dimension of the model
        nhead=8,             # Number of attention heads
        num_encoder_layers=1,# Number of transformer layers
        num_classes=NUM_CLASSES,
        dropout=0.2         # Dropout rate
    ).to(DEVICE)
    # model = CNNGRUClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    print("Model initialized:")
    print(model)

    # summary(model, input_size=(1, 1000, 8, 4), device=DEVICE)

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
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )


    # optimizer = optim.AdamW(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5) For CNN and Other Model
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2) # For MotionTransformer

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # === Training & Testing Loop ===
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")
        epoch_dir = os.path.join(SAVE_DIR, f"Epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        # --- Training ---
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels, masks, _ in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            # print(f"Input shape: {inputs.shape}")
            # print(f"Labels shape: {labels.shape}")
            # print(f"Masks shape: {masks.shape}")
            inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer.zero_grad()
            # outputs = model(inputs)
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
                # outputs = model(inputs)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())


        test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"Test Accuracy: {test_acc:.2f}%")

        # --- Save Classification Metrics ---
        report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(os.path.join(epoch_dir, "classification_report.csv"))

        # --- Save Confusion Matrix ---
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots(figsize=(12, 10))
        disp.plot(xticks_rotation=45, cmap='Blues', ax=ax)
        plt.title(f"Confusion Matrix - Epoch {epoch}")
        plt.savefig(os.path.join(epoch_dir, "confusion_matrix.png"))
        plt.close()

        # --- Compute & Save Per-Class Accuracy ---
        per_class_correct = np.zeros(NUM_CLASSES)
        per_class_total = np.zeros(NUM_CLASSES)

        for label, pred in zip(all_labels, all_preds):
            per_class_total[label] += 1
            if label == pred:
                per_class_correct[label] += 1

        per_class_accuracy = per_class_correct / np.maximum(per_class_total, 1)
        per_class_acc_df = pd.DataFrame({
            "Class": [f"Class_{i}" for i in range(NUM_CLASSES)],
            "Accuracy": per_class_accuracy,
            "Total_Samples": per_class_total.astype(int),
            "Correct_Predictions": per_class_correct.astype(int)
        })
        per_class_acc_df.to_csv(os.path.join(epoch_dir, "per_class_accuracy.csv"), index=False)

        print("\nPer-class Accuracy:")
        for idx, acc in enumerate(per_class_accuracy):
            print(f"Class {idx}: {acc*100:.2f}% ({int(per_class_correct[idx])}/{int(per_class_total[idx])})")

        # --- Save Model Checkpoint ---
        torch.save(model.state_dict(), os.path.join(epoch_dir, f"model_epoch_{epoch}.pt"))


        # --- Store Metrics ---
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(f"Epoch {epoch} completed. Model saved to {epoch_dir}")

    # === Final Metrics Saving ===
    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, EPOCHS + 1)),
        "Train_Loss": train_losses,
        "Train_Accuracy": train_accuracies,
        "Test_Accuracy": test_accuracies
    })
    metrics_df.to_csv(os.path.join(SAVE_DIR, "training_metrics.csv"), index=False)

    # === Final Loss Plot ===
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
    plt.close()

    # === Final Accuracy Plot ===
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, EPOCHS + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train/Test Accuracy per Epoch')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "accuracy_curve.png"))
    plt.close()

    # === Final Metrics Plot ===
    print("\n Training Complete.")