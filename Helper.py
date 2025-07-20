import pickle
from collections import defaultdict
from ultralytics import YOLO
import os
import torch
import numpy as np
from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor



# Constants
MODEL_PATH = "yolo11x.pt"
FRAME_SIZE = (512, 384)
SAVE_DIR = "./videosTensors100"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load models
yolo_model = YOLO(MODEL_PATH)
cotracker = CoTrackerPredictor(
    checkpoint='/Users/mrinalraj/Downloads/WebDownload/Driving48/checkpoints/scaled_offline.pth'
)

ListWithManualAnnotationNeeded = []

def get_best_person_box(frame):
    results = yolo_model(frame)
    result = results[0]
    boxes = result.boxes
    cls_ids = boxes.cls.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    class_names = result.names
    # print(f"Detected classes: {class_names}")
    if boxes is None or boxes.cls is None or len(boxes.cls) == 0:
        print("No detections found")
        return None
    person_indices = [i for i, cls_id in enumerate(cls_ids) if class_names[int(cls_id)] == 'person']
    person_boxes = [xyxy[i] for i in person_indices]

    max_area = 0    
    # print(f"Number of person boxes detected: {len(person_boxes)}")
    # print("Finding area")
    best_box = None
    for box in person_boxes:
        print(f"Box: {box}")
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_box = [x1, y1, x2, y2]

    return best_box



def get_query_points(frame, best_box, num_points=1000):
    print("Inside Query points generation function:")
    # print("Frame shape:", frame.shape)

    query_metrics = []
    x1, y1, x2, y2 = best_box
    print(f"Best box coordinates: {x1}, {y1}, {x2}, {y2}")
    # Generate uniformly spaced (x, y) points inside the box
    xs = np.random.uniform(x1, x2, size=num_points).astype(int)
    ys = np.random.uniform(y1, y2, size=num_points).astype(int)

    # Clip coordinates to stay within frame bounds (just in case)
    xs = np.clip(xs, 0, frame.shape[1] - 1)
    ys = np.clip(ys, 0, frame.shape[0] - 1)
    # print(f"Sampled {len(xs)} points within the bounding box.")
    # Visual confirmation: draw circles on the frame
    for x, y in zip(xs, ys):
        cv2.circle(frame, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
        query_metrics.append([0, x, y])

    # Show the annotated frame
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title(f"Uniformly Sampled {num_points} Points")
    # plt.show()

    return torch.tensor(query_metrics, dtype=torch.float32), frame


# import cv2
# import numpy as np
# import torch
# from typing import Tuple, Optional


# def get_query_points(
#     frame: np.ndarray,
#     best_box: Optional[Tuple[int, int, int, int]],
#     num_points: int = 100,
#     canny_thresh: Tuple[int, int] = (50, 150),
#     debug: bool = False,
# ):
#     """Generate query points inside the largest contour found within *best_box*.

#     The function attempts to sample points that lie on the foreground silhouette
#     (largest contour) of the object inside the provided bounding box. If no
#     contour is found, it gracefully falls back to uniformly sampling inside the
#     bounding box.

#     Parameters
#     ----------
#     frame : np.ndarray
#         Input BGR frame.
#     best_box : tuple[int, int, int, int] | None
#         Bounding box in *(x1, y1, x2, y2)* format. If *None* or empty, an empty
#         tensor is returned.
#     num_points : int, default 1000
#         Desired number of query points.
#     canny_thresh : tuple[int, int], default (50, 150)
#         *(low, high)* thresholds for Canny edge detection.
#     debug : bool, default False
#         If *True*, the function draws the intermediate contour mask to the
#         screen for inspection.

#     Returns
#     -------
#     torch.Tensor
#         Tensor of shape *(N, 3)* with rows ``[0, x, y]`` where 0 is a dummy
#         label to remain compatible with downstream code.
#     np.ndarray
#         A copy of *frame* annotated with the bounding box and sampled points.
#     """

#     # Make a writable copy – never scribble on the caller's `frame`.
#     annotated = frame.copy()
#     query_metrics = []

#     # -----------------------------
#     # Validation & early-return
#     # -----------------------------
#     if not best_box:
#         # Gracefully handle missing detection.
#         return torch.empty((0, 3), dtype=torch.float32), annotated

#     x1, y1, x2, y2 = map(int, best_box)

#     # Draw bounding box & label for visual confirmation.
#     cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(
#         annotated,
#         "object",
#         (x1, max(y1 - 10, 0)),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.6,
#         (0, 255, 0),
#         2,
#     )

#     # -----------------------------
#     # Contour-guided sampling
#     # -----------------------------
#     cropped = frame[y1:y2, x1:x2]

#     # Handle degenerate crop.
#     if cropped.size == 0:
#         return torch.empty((0, 3), dtype=torch.float32), annotated

#     gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, canny_thresh[0], canny_thresh[1])

#     # Dilate to close small gaps so contours are more contiguous.
#     kernel = np.ones((3, 3), np.uint8)
#     dilated = cv2.dilate(edges, kernel, iterations=2)

#     contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     xs_sampled: np.ndarray | None = None
#     ys_sampled: np.ndarray | None = None

#     if contours:
#         # Choose the largest contour (heuristic for main silhouette).
#         largest = max(contours, key=cv2.contourArea)

#         # Rasterise the contour into a mask.
#         mask = np.zeros_like(gray)
#         cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

#         # Pixel coordinates inside the mask.
#         ys, xs = np.where(mask == 255)

#         if len(xs):
#             k = min(num_points, len(xs))
#             idx = np.random.choice(len(xs), size=k, replace=False)
#             xs_sampled = xs[idx] + x1  # Map back to frame coords
#             ys_sampled = ys[idx] + y1

#             # Optional debug visualisation.
#             if debug:
#                 vis = np.zeros_like(mask)
#                 vis[ys[idx], xs[idx]] = 255
#                 cv2.imshow("Contour mask", mask)
#                 cv2.imshow("Sampled mask points", vis)
#                 cv2.waitKey(1)

#     # -----------------------------
#     # Fallback uniform sampling if contour path failed.
#     # -----------------------------
#     if xs_sampled is None or ys_sampled is None:
#         xs_sampled = np.random.uniform(x1, x2, size=num_points).astype(int)
#         ys_sampled = np.random.uniform(y1, y2, size=num_points).astype(int)

#     # Clip to valid frame bounds.
#     xs_sampled = np.clip(xs_sampled, 0, frame.shape[1] - 1)
#     ys_sampled = np.clip(ys_sampled, 0, frame.shape[0] - 1)

#     # Mark points on the annotated image and build the output array.
#     for x, y in zip(xs_sampled, ys_sampled):
#         cv2.circle(annotated, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
#         query_metrics.append([0, x, y])

#     return torch.tensor(query_metrics, dtype=torch.float32), annotated
