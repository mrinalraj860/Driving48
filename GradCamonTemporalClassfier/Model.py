import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import cv2

class EnhancedTemporalTransformerClassifier(nn.Module):
    def __init__(self, input_dim=8, num_classes=47, d_model=256, nhead=8,
                 num_encoder_layers=2, dim_feedforward=512, dropout=0.2):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
        self.features = None
        self.gradients = None
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x, mask=None, return_features=False):
        # x: [B, T, N, 8] from dataloader
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # → [B, N, T, 8]
        x = x.view(B * N, T, C)  # → [B*N, T, 8]
        x = x.permute(0, 2, 1)  # → [B*N, 8, T]
        x = self.temporal_conv(x)  # → [B*N, d_model, T]
        x = x.permute(0, 2, 1)  # → [B*N, T, d_model]
        
        if mask is not None:
            # mask: [B, T] → [B*N, T]
            mask_exp = mask.unsqueeze(1).expand(B, N, T).reshape(B * N, T)
            x = x.masked_fill(mask_exp.unsqueeze(-1), 0.0)
            valid_counts = (~mask_exp).sum(dim=1, keepdim=True).clamp(min=1)
            x = x.sum(dim=1) / valid_counts  # [B*N, d_model]
        else:
            x = x.mean(dim=1)
            
        # Reshape for transformer: [B, N, d_model]
        x = x.view(B, N, -1)
        
        # Store features and register hook for GradCAM
        self.features = x
        if x.requires_grad:
            x.register_hook(self.activations_hook)
            
        x = self.transformer(x)  # [B, N, d_model]
        
        if return_features:
            point_features = x.clone()  # [B, N, d_model]
        
        x = x.mean(dim=1)  # global aggregation across points
        output = self.classifier(x)  # [B, num_classes]
        
        if return_features:
            return output, point_features
        return output
    
    @torch.no_grad()
    def normalize_per_sample(self, scores):
        # scores: [N], normalize to [0,1] robustly
        if scores.numel() == 0:
            return scores
        smin, smax = scores.min(), scores.max()
        return (scores - smin) / (smax - smin + 1e-12)

    def get_gradcam(self, input_tensor, target_class=None):
        """
        Per-point Grad-CAM-like saliency on the transformer features.
        Returns: np.array of shape [N] in [0,1].
        """
        self.eval()
        device = next(self.parameters()).device
        x = input_tensor.to(device)
        x.requires_grad_(True)

        # forward
        logits = self.forward(x)             # sets self.features and hook to fill self.gradients
        if target_class is None:
            target_class = logits.argmax(dim=1)  # [B]
        # backward for class score (first sample only, matching your usage)
        self.zero_grad(set_to_none=True)
        class_score = logits[0, target_class[0]]
        class_score.backward(retain_graph=False)

        # gradients/features shapes: [B, N, d_model]
        grads = self.gradients[0]    # [N, d_model]
        feats = self.features[0]     # [N, d_model]

        # Standard CAM weighting per point = <grad, feat> across feature dim
        # (could also use |grad|, or grad.mean(dim=1) if you prefer)
        point_scores = (grads * feats).sum(dim=1)          # [N]
        point_scores = F.relu(point_scores)                # non-negative importance
        point_scores = self.normalize_per_sample(point_scores).detach().cpu().numpy()
        return point_scores

class PointSelector:
    """Class to select best points based on various criteria"""
    
    @staticmethod
    def select_best_points_motion(data, n_select=100):
        """
        Select points based on motion characteristics (NO visibility filtering)
        data: [B, T, N, 8] tensor
        Returns indices of best points
        """
        B, T, N, C = data.shape
        scores = torch.zeros(B, N)
        
        for b in range(B):
            # Extract motion features for each point
            velocities = data[b, :, :, 3:5]  # vel_x, vel_y
            accelerations = data[b, :, :, 5:7]  # acc_x, acc_y
            direction_change = data[b, :, :, 7]  # direction_change
            
            # Calculate motion magnitude
            vel_magnitude = torch.norm(velocities, dim=2)  # [T, N]
            acc_magnitude = torch.norm(accelerations, dim=2)  # [T, N]
            
            # Score based on motion only (NO visibility component)
            motion_score = vel_magnitude.mean(dim=0)  # Average velocity magnitude
            dynamic_score = direction_change.std(dim=0)  # Direction change variability
            
            # Combine scores (only motion-based)
            scores[b] = (motion_score * 0.7 + dynamic_score * 0.3)
        
        # Select top N points for each batch
        _, top_indices = torch.topk(scores, n_select, dim=1)
        return top_indices
    
    @staticmethod
    def select_best_points_gradcam(model, data, n_select=100):
        """
        Select points based on GradCAM importance
        """
        B = data.shape[0]
        all_indices = []
        
        for b in range(B):
            single_data = data[b:b+1]  # Keep batch dimension
            cam = model.get_gradcam(single_data)
            
            # Select top points based on GradCAM scores
            top_indices = np.argsort(cam)[-n_select:]
            all_indices.append(torch.tensor(top_indices))
            
        return torch.stack(all_indices)

class PointTrackingVisualizer:
    """Visualize point tracking and movement"""
    
    def __init__(self, video_width=640, video_height=480):
        self.video_width = video_width
        self.video_height = video_height
        
    def visualize_points_movement(self, data, point_indices, save_path=None, 
                                show_trails=True, trail_length=10, title_suffix=""):
        """
        Visualize movement of selected points across frames (NO visibility filtering)
        data: [T, N, 8] - single video sequence 
        point_indices: [n_select] - indices of points to visualize
        title_suffix: string to add to title
        """
        T, N, C = data.shape
        n_select = len(point_indices)
        
        # Extract coordinates for selected points
        positions = data[:, point_indices, :2]  # [T, n_select, 2]
        
        # Convert normalized coordinates to pixel coordinates
        positions[:, :, 0] *= self.video_width   # x coordinates
        positions[:, :, 1] *= self.video_height  # y coordinates
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, self.video_width)
        ax.set_ylim(0, self.video_height)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert y-axis to match image coordinates
        
        # Color map for different points
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_select, 20)))
        if n_select > 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            colors = np.tile(colors, (n_select // 20 + 1, 1))[:n_select]
        
        def animate(frame):
            ax.clear()
            ax.set_xlim(0, self.video_width)
            ax.set_ylim(0, self.video_height)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(f'Point Tracking{title_suffix} - Frame {frame}/{T-1}')
            
            for i in range(n_select):
                x, y = positions[frame, i]
                
                # Plot current point (always visible, no visibility check)
                ax.plot(x, y, 'o', color=colors[i], markersize=8, alpha=0.8)
                
                # Plot trail
                if show_trails and frame > 0:
                    start_frame = max(0, frame - trail_length)
                    trail_x = positions[start_frame:frame+1, i, 0]
                    trail_y = positions[start_frame:frame+1, i, 1]
                    
                    # Plot full trail (no visibility filtering)
                    ax.plot(trail_x, trail_y, '-', color=colors[i], 
                           alpha=0.6, linewidth=2)
            
            ax.grid(True, alpha=0.3)
            return []
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=T, 
                                     interval=100, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to {save_path}")
        
        plt.show()
        return anim
    
    def visualize_velocity_vectors(self, data, point_indices, save_path=None, 
                                  trail_length=10, scale_factor=50):
        """
        Visualize velocity vectors of selected points
        """
        T, N, C = data.shape
        n_select = len(point_indices)
        
        # Extract data for selected points
        positions = data[:, point_indices, :2]  # [T, n_select, 2]
        velocities = data[:, point_indices, 3:5]  # [T, n_select, 2]
        
        # Convert normalized coordinates to pixel coordinates
        positions[:, :, 0] *= self.video_width
        positions[:, :, 1] *= self.video_height
        
        # Color map
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_select, 20)))
        if n_select > 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            colors = np.tile(colors, (n_select // 20 + 1, 1))[:n_select]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def animate(frame):
            ax.clear()
            ax.set_xlim(0, self.video_width)
            ax.set_ylim(0, self.video_height)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(f'Velocity Vectors - Frame {frame}/{T-1}')
            
            for i in range(n_select):
                x, y = positions[frame, i]
                vx, vy = velocities[frame, i]
                
                # Scale velocity for visualization
                vx_scaled = vx * scale_factor
                vy_scaled = vy * scale_factor
                
                # Plot point
                ax.plot(x, y, 'o', color=colors[i], markersize=6, alpha=0.8)
                
                # Plot velocity vector
                ax.arrow(x, y, vx_scaled, vy_scaled, head_width=10, head_length=15,
                        fc=colors[i], ec=colors[i], alpha=0.7)
                
                # Plot trail
                if frame > 0:
                    start_frame = max(0, frame - trail_length)
                    trail_x = positions[start_frame:frame+1, i, 0]
                    trail_y = positions[start_frame:frame+1, i, 1]
                    ax.plot(trail_x, trail_y, '-', color=colors[i], alpha=0.4, linewidth=1)
            
            ax.grid(True, alpha=0.3)
            return []
        
        anim = animation.FuncAnimation(fig, animate, frames=T, 
                                     interval=100, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Velocity animation saved to {save_path}")
        
        plt.show()
        return anim
    
    def visualize_acceleration_vectors(self, data, point_indices, save_path=None, 
                                     trail_length=10, scale_factor=200):
        """
        Visualize acceleration vectors of selected points
        """
        T, N, C = data.shape
        n_select = len(point_indices)
        
        # Extract data for selected points
        positions = data[:, point_indices, :2]  # [T, n_select, 2]
        accelerations = data[:, point_indices, 5:7]  # [T, n_select, 2]
        
        # Convert normalized coordinates to pixel coordinates
        positions[:, :, 0] *= self.video_width
        positions[:, :, 1] *= self.video_height
        
        # Color map
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_select, 20)))
        if n_select > 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            colors = np.tile(colors, (n_select // 20 + 1, 1))[:n_select]
        
        fig, axis = plt.subplots(figsize=(12, 8))
        
        def animate(frame):
            axis.clear()
            axis.set_xlim(0, self.video_width)
            axis.set_ylim(0, self.video_height)
            axis.set_aspect('equal')
            axis.invert_yaxis()
            axis.set_title(f'Acceleration Vectors - Frame {frame}/{T-1}')
            
            for i in range(n_select):
                x, y = positions[frame, i]
                acc_x, acc_y = accelerations[frame, i]
                
                # Scale acceleration for visualization
                acc_x_scaled = acc_x * scale_factor
                acc_y_scaled = acc_y * scale_factor
                
                # Plot point
                axis.plot(x, y, 'o', color=colors[i], markersize=6, alpha=0.8)
                
                # Plot acceleration vector (only if magnitude is significant)
                acc_magnitude = np.sqrt(acc_x_scaled**2 + acc_y_scaled**2)
                if acc_magnitude > 5:  # threshold to avoid cluttering with tiny vectors
                    axis.arrow(x, y, acc_x_scaled, acc_y_scaled, head_width=8, head_length=12,
                              fc=colors[i], ec=colors[i], alpha=0.7)
                
                # Plot trail
                if frame > 0:
                    start_frame = max(0, frame - trail_length)
                    trail_x = positions[start_frame:frame+1, i, 0]
                    trail_y = positions[start_frame:frame+1, i, 1]
                    axis.plot(trail_x, trail_y, '-', color=colors[i], alpha=0.4, linewidth=1)
            
            axis.grid(True, alpha=0.3)
            return []
        
        anim = animation.FuncAnimation(fig, animate, frames=T, 
                                     interval=100, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Acceleration animation saved to {save_path}")
        
        plt.show()
        return anim
    
    def plot_motion_statistics(self, data, point_indices):
        """
        Plot motion statistics for selected points
        """
        T, N, C = data.shape
        selected_data = data[:, point_indices, :]  # [T, n_select, 8]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Velocity magnitude over time
        velocities = selected_data[:, :, 3:5]  # [T, n_select, 2]
        vel_magnitude = torch.norm(velocities, dim=2)  # [T, n_select]
        
        axes[0, 0].plot(vel_magnitude.mean(dim=1))
        axes[0, 0].fill_between(range(T), 
                               vel_magnitude.mean(dim=1) - vel_magnitude.std(dim=1),
                               vel_magnitude.mean(dim=1) + vel_magnitude.std(dim=1),
                               alpha=0.3)
        axes[0, 0].set_title('Average Velocity Magnitude')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Velocity')
        
        # Acceleration magnitude over time
        accelerations = selected_data[:, :, 5:7]  # [T, n_select, 2]
        acc_magnitude = torch.norm(accelerations, dim=2)  # [T, n_select]
        
        axes[0, 1].plot(acc_magnitude.mean(dim=1))
        axes[0, 1].fill_between(range(T),
                               acc_magnitude.mean(dim=1) - acc_magnitude.std(dim=1),
                               acc_magnitude.mean(dim=1) + acc_magnitude.std(dim=1),
                               alpha=0.3)
        axes[0, 1].set_title('Average Acceleration Magnitude')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Acceleration')
        
        # Visibility over time (for information only)
        visibility = selected_data[:, :, 2]  # [T, n_select]
        axes[1, 0].plot(visibility.mean(dim=1))
        axes[1, 0].set_title('Average Visibility (Not Used in Selection)')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Visibility')
        
        # Direction change over time
        direction_change = selected_data[:, :, 7]  # [T, n_select]
        axes[1, 1].plot(direction_change.mean(dim=1))
        axes[1, 1].set_title('Average Direction Change')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Direction Change')
        
        plt.tight_layout()
        plt.show()


def analyze_video_with_comprehensive_visualization(model, data_tensor, method='motion', n_select=100,
                                                 trail_length=15):
    """
    Comprehensive analysis with multiple visualizations:
    1. Basic point tracking
    2. Velocity vectors
    3. Acceleration vectors
    """
    print(f"Input data shape: {data_tensor.shape}")
    B, T, N, C = data_tensor.shape

    selector = PointSelector()
    visualizer = PointTrackingVisualizer()

    # Select best points (no visibility filtering)
    if method == 'motion':
        print("Selecting points based on motion characteristics (no visibility filter)...")
        best_indices = selector.select_best_points_motion(data_tensor, n_select)
    elif method == 'gradcam':
        print("Selecting points based on GradCAM importance...")
        best_indices = selector.select_best_points_gradcam(model, data_tensor, n_select)
    else:
        raise ValueError("Method must be 'motion' or 'gradcam'")

    print(f"Selected {n_select} best points for each video")

    video_data = data_tensor[0]        # [T, N, 8]
    video_indices = best_indices[0]    # [n_select]

    print("Creating comprehensive visualizations...")

    # 1. Motion statistics
    visualizer.plot_motion_statistics(video_data, video_indices)

    # 2. Basic point tracking
    print("Creating basic point tracking animation...")
    anim_basic = visualizer.visualize_points_movement(
        video_data, video_indices,
        save_path='point_tracking_basic.gif',
        show_trails=True,
        trail_length=trail_length,
        title_suffix=" (Basic)"
    )

    # 3. Velocity vectors
    print("Creating velocity vector animation...")
    anim_velocity = visualizer.visualize_velocity_vectors(
        video_data, video_indices,
        save_path='point_tracking_velocity.gif',
        trail_length=trail_length,
        scale_factor=50
    )

    # 4. Acceleration vectors
    print("Creating acceleration vector animation...")
    anim_acceleration = visualizer.visualize_acceleration_vectors(
        video_data, video_indices,
        save_path='point_tracking_acceleration.gif',
        trail_length=trail_length,
        scale_factor=200
    )

    return best_indices, (anim_basic, anim_velocity, anim_acceleration)