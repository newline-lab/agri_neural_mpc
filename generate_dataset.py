import csv
import os
import cv2
import torch
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================
# ROS / YOLO IMPORTS
# =========================

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.getcwd(), script_dir + '/ros/src/yolov7-ros/src'))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from utils.datasets import letterbox


# =========================
# CONFIGURATION
# =========================

is_ripe = True
polar = False
fixedView = False


# =========================
# TREE SCORE FUNCTION
# =========================

def weight_value(n_elements, mean_score, midpoint=5, steepness=10.):
    return np.ceil(
        100 * (mean_score - 0.5) *
        (0.5 + 0.5 * np.tanh(steepness * (n_elements - midpoint)))
    ) / 100


# =========================
# UTILITIES
# =========================

def find_latest_folder(base_dir):
    subdirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    return os.path.join(
        base_dir,
        max(subdirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    )


# =========================
# HEATMAP PLOTTING FUNCTION
# =========================

def plot_3d_scatter(X_np, Y_1, class_ripe, output_dir, file_name="3d_scatter.png"):
    """
    Plots a 3D scatter plot with the tree scores and the (x, y) coordinates.

    Args:
        X_np (ndarray): Array of shape (N, 2) with (x, y) coordinates.
        Y_1 (list): List of tree scores.
        class_ripe (bool): Whether the class is ripe (True) or unripe (False).
        output_dir (str): Directory to save the scatter plot image.
        file_name (str): Name of the scatter plot image file.

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(1, 2)

    # 3D scatter plot
    ax1 = fig.add_subplot(gs[0, 1], projection='3d')
    sc1 = ax1.scatter(X_np[:, 0], X_np[:, 1], Y_1, c=Y_1, cmap='viridis', s=15)
    
    ax1.set_xlabel(r'$(\mathbf{t} - \mathbf{p})_x$')
    ax1.set_ylabel(r'$(\mathbf{t} - \mathbf{p})_y$')
    ax1.set_zlabel(r'[$g(\mathbf{x})]_\mathrm{ripe}$' if class_ripe else r'[$g(\mathbf{x})]_\mathrm{unripe}$')
    
    ax1.view_init(elev=30, azim=45)  # Adjust the view angle
    ax1.zaxis.set_rotate_label(False)
    ax1.zaxis.label.set_rotation(90)
    
    # Optional: Add some extra points (e.g., in orange)
    add_orange_scatter(ax1)
    
    # Add a color bar to indicate the value of tree scores
    fig.colorbar(sc1, ax=ax1, shrink=0.5, aspect=10)

    # Save the figure
    scatter_plot_path = os.path.join(output_dir, file_name)
    plt.savefig(scatter_plot_path)
    plt.close()

    print(f"3D scatter plot saved to: {scatter_plot_path}")


# Helper function to add orange scatter points if needed
def add_orange_scatter(ax):
    # For example, adding orange points to the scatter plot (you can customize this part)
    ax.scatter(0, 0, 0, c='orange', marker='o', s=50)  # Dummy example point


# =========================
# DETECTION
# =========================

def run_detection():

    device = select_device('0' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join(
        script_dir,
        'ros/src/yolov7-ros/weights/apples_weights.pt'
    )

    model = attempt_load(model_path, map_location=device).eval().to(device)
    class_names = model.names

    print(f"Model loaded with {len(class_names)} classes: {class_names}")

    # Dataset path
    base_dir = os.path.join(script_dir, 'datasets')
    base_dir = os.path.join(base_dir, "polar" if polar else "cartesian")
    base_dir = os.path.join(base_dir, "fixed_view" if fixedView else "variable_view")
    base_dir = os.path.join(base_dir, "ripe" if is_ripe else "raw")

    latest_folder = find_latest_folder(base_dir)
    print(f"Processing images in: {latest_folder}")

    input_csv_path = glob.glob(os.path.join(latest_folder, "*.csv"))[0]

    output_image_dir = os.path.join(latest_folder, "detections")
    os.makedirs(output_image_dir, exist_ok=True)

    output_csv_path = os.path.join(latest_folder, 'TreeDatasetCNN.csv')

    rows = []
    image_paths = []

    with open(input_csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append(row)
            image_paths.append(row[-1])


    def process_batch(image_paths, batch_size=8):

        all_detections = []
        all_original_images = []

        for i in range(0, len(image_paths), batch_size):

            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            original_images = []
            ratio_pads = []

            for path in batch_paths:

                img = cv2.imread(path)
                if img is None:
                    all_detections.append(None)
                    all_original_images.append(None)
                    continue

                original_img = img.copy()

                img_resized, ratio, pad = letterbox(img, 640, auto=False)
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                img_tensor = torch.from_numpy(img_resized).float().to(device) / 255.0

                batch_images.append(img_tensor)
                original_images.append(original_img)
                ratio_pads.append((ratio, pad))

            if not batch_images:
                continue

            with torch.no_grad():
                pred = model(torch.stack(batch_images, 0))[0]

            detections = non_max_suppression(pred, 0.6, 0.7)

            for j, det in enumerate(detections):

                orig_img = original_images[j]

                if det is not None and len(det):
                    det[:, :4] = scale_coords(
                        batch_images[0].shape[1:],
                        det[:, :4],
                        orig_img.shape,
                        ratio_pads[j]
                    )

                all_detections.append(det)
                all_original_images.append(orig_img)

        return all_detections, all_original_images


    detections, original_images = process_batch(image_paths)


    # =========================
    # WRITE FINAL CSV
    # =========================

    x_coords = []
    y_coords = []
    scores = []

    with open(output_csv_path, 'w', newline='') as f:

        writer = csv.writer(f, quoting=csv.QUOTE_ALL)

        header = [
            'x', 'y', 'yaw',
            'image_path',
            'Ripe_scores',
            'Raw_scores',
            'tree_score'
        ]

        writer.writerow(header)

        for row, det, img, img_path in zip(rows, detections, original_images, image_paths):

            relative_path = os.path.relpath(img_path, latest_folder)

            ripe_scores = []
            raw_scores = []

            if det is not None:
                for *xyxy, conf, cls in det:

                    cls_name = class_names[int(cls)]
                    conf_val = float(conf.item())

                    if cls_name == 'Ripe_Apple':
                        ripe_scores.append(conf_val)

                    if cls_name == 'Raw_Apple':
                        raw_scores.append(conf_val)

                    # Draw box
                    if img is not None:
                        plot_one_box(
                            xyxy,
                            img,
                            label=f"{cls_name} {conf_val:.2f}",
                            line_thickness=2
                        )

            # Save annotated image
            if img is not None:
                output_path = os.path.join(
                    output_image_dir,
                    os.path.basename(img_path)
                )
                cv2.imwrite(output_path, img)

            # Compute weighted scores
            ripe_value = weight_value(
                len(ripe_scores),
                float(np.mean(ripe_scores)) if ripe_scores else 0.0
            )

            raw_value = weight_value(
                len(raw_scores),
                float(np.mean(raw_scores)) if raw_scores else 0.0
            )

            tree_score = float(ripe_value - raw_value + 0.5)

            # Add x, y, and tree_score for 3D plot
            x_coords.append(float(row[0]))
            y_coords.append(float(row[1]))
            scores.append(tree_score)

            new_row = [
                float(row[0]),  # x
                float(row[1]),  # y
                float(row[2]),  # yaw
                relative_path,
                ripe_scores,
                raw_scores,
                tree_score
            ]

            writer.writerow(new_row)

    # Convert to numpy arrays for plotting
    X_np = np.array(list(zip(x_coords, y_coords)))  # Convert to 2D numpy array for (x, y) coordinates
    plot_3d_scatter(X_np, scores, is_ripe, output_image_dir)

    print("\n? Detection and tree score computation completed successfully.")
    print(f"? Output CSV saved at:\n{output_csv_path}\n")
    print(f"? Annotated images saved in:\n{output_image_dir}")
    return latest_folder


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    run_detection()

    