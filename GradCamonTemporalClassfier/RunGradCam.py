import torch
import numpy as np
from Dataloader import MotionDataset, motion_collate_fn
from torch.utils.data import DataLoader, Dataset
from Model import EnhancedTemporalTransformerClassifier
from Model import *
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Create sample data
    BATCH_SIZE = 2
    TEST_PT_FOLDER = "/Users/mrinalraj/Downloads/WebDownload/Driving48/Test"
    TEST_DF_PATH = "/Users/mrinalraj/Downloads/WebDownload/Driving48/ProcessedTestCorrect.csv"
    val_df = pd.read_csv(TEST_DF_PATH)
    val_dataset = MotionDataset(val_df, TEST_PT_FOLDER, training=False)

    print("Creating sample data...")
    data_tensor = val_dataset[0]  # Get first sample
    motion_tensor, label, video_name = val_dataset[0]
    print(f"Sample data shape: {motion_tensor.shape}, Label: {label}, Video: {video_name}")
    motion_tensor = motion_tensor.unsqueeze(0)
    # print(f"Sample data shape: {motion_tensor}")
    # print(f"Data shape: {motion_tensor.shape}")
    
    # Initialize model
    print("Initializing model...")
    model = EnhancedTemporalTransformerClassifier()

    # Load the checkpoint
    checkpoint = torch.load("/Users/mrinalraj/Downloads/WebDownload/Driving48/GradCamonTemporalClassfier/best_motion_classifier_model.pth", map_location=torch.device('cpu'))

    # Load the model weights from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    # Put model in eval mode
    model.eval()
    
    print("\n" + "="*50)
    print("METHOD 1: Motion-based Point Selection")
    print("="*50)
    
    # Method 1: Select points based on motion characteristics
    best_indices_motion, anim_motion = analyze_video_with_gradcam(
        model, motion_tensor, method='motion', n_select=100
    )
    
    print(f"Selected point indices shape: {best_indices_motion.shape}")
    print(f"First few selected indices: {best_indices_motion[0][:10]}")
    
    print("\n" + "="*50)
    print("METHOD 2: GradCAM-based Point Selection")
    print("="*50)
    
    # Method 2: Select points based on GradCAM importance
    # Note: This requires a forward pass through the model
    with torch.no_grad():
        # First do a forward pass to get the model predictions
        output = model(motion_tensor)
        print(f"Model output shape: {output.shape}")
    
    # Now use GradCAM for point selection
    best_indices_gradcam, anim_gradcam = analyze_video_with_gradcam(
        model, motion_tensor, method='gradcam', n_select=100
    )
    
    print(f"GradCAM selected point indices shape: {best_indices_gradcam.shape}")
    print(f"First few GradCAM selected indices: {best_indices_gradcam[0][:10]}")
    
    # Compare the two methods
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    
    # Check overlap between methods
    overlap = torch.isin(best_indices_motion[0], best_indices_gradcam[0]).sum()
    print(f"Overlap between motion and GradCAM selection: {overlap}/100 points")
    
    # Get GradCAM scores for all points in first video
    print("\nGetting detailed GradCAM analysis...")
    single_video = motion_tensor[0:1]  # [1, T, N, 8]
    gradcam_scores = model.get_gradcam(single_video)
    print(f"GradCAM scores shape: {gradcam_scores.shape}")
    print(f"GradCAM score range: {gradcam_scores.min():.4f} to {gradcam_scores.max():.4f}")
    
    # Show statistics about selected points
    motion_selected_scores = gradcam_scores[best_indices_motion[0]]
    gradcam_selected_scores = gradcam_scores[best_indices_gradcam[0]]
    
    print(f"\nMotion-selected points GradCAM scores:")
    print(f"  Mean: {motion_selected_scores.mean():.4f}")
    print(f"  Std:  {motion_selected_scores.std():.4f}")
    
    print(f"\nGradCAM-selected points GradCAM scores:")
    print(f"  Mean: {gradcam_selected_scores.mean():.4f}")
    print(f"  Std:  {gradcam_selected_scores.std():.4f}")

    return model, motion_tensor, best_indices_motion, best_indices_gradcam

# Additional utility functions for analysis
def analyze_point_importance(model, data_tensor, class_idx=0):
    """
    Analyze which points are most important for a specific class prediction
    """
    model.eval()
    
    # Get GradCAM for specific class
    gradcam_scores = model.get_gradcam(data_tensor[0:1], 
                                      target_class=torch.tensor([class_idx]))
    
    # Sort points by importance
    sorted_indices = np.argsort(gradcam_scores)[::-1]  # Descending order
    
    print(f"Top 10 most important points for class {class_idx}:")
    for i in range(10):
        point_idx = sorted_indices[i]
        score = gradcam_scores[point_idx]
        print(f"  Point {point_idx:4d}: {score:.4f}")
    
    return sorted_indices, gradcam_scores

def visualize_gradcam_heatmap(gradcam_scores, video_width=384, video_height=512):
    """
    Create a heatmap visualization of GradCAM scores
    """
    # This is a simplified version - you'd need actual point coordinates
    # for a proper spatial heatmap
    
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Score distribution
    plt.subplot(1, 2, 1)
    plt.hist(gradcam_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('GradCAM Score')
    plt.ylabel('Number of Points')
    plt.title('Distribution of GradCAM Scores')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Top points ranking
    plt.subplot(1, 2, 2)
    sorted_scores = np.sort(gradcam_scores)[::-1][:100]  # Top 100
    plt.plot(sorted_scores, 'o-', markersize=3)
    plt.xlabel('Rank')
    plt.ylabel('GradCAM Score')
    plt.title('Top 100 Points by GradCAM Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the main analysis
    model, data, motion_indices, gradcam_indices = main()
    
    # Additional analysis
    print("\n" + "="*50)
    print("DETAILED ANALYSIS")
    print("="*50)
    print(model)
    # Analyze specific class importance
    analyze_point_importance(model, data, class_idx=0)
    
    # Create GradCAM heatmap
    gradcam_scores = model.get_gradcam(data[0:1])
    visualize_gradcam_heatmap(gradcam_scores)