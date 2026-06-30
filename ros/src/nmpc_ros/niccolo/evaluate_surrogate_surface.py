#!/usr/bin/env python3
"""
evaluate_surrogate.py

Animated evaluation of the trained surrogate confidence model.

Generates a GIF showing the model's predicted confidence surface over
the (x, y) spatial grid as the orientation (azimuth) sweeps from -180
to +180 degrees. A moving robot marker traces a circle around the origin
to show the current azimuth intuitively.

Usage:
  python evaluate_surrogate.py \
    [--model_path /path/to/best_model_weights.pth] \
    [--grid_min -7.0] [--grid_max 7.0] \
    [--resolution 200] [--step_deg 5] \
    [--robot_radius 3.5] \
    [--output nn_surface_anim.gif]
"""

import os
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches


# ===========================================================================
# CLI
# ===========================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",      type=str,   default=None)
parser.add_argument("--data_type",       type=str,   default="fronte")
parser.add_argument("--dataset_version", type=str,   default="orig")
parser.add_argument("--split",           type=str,   default="80_20")
parser.add_argument("--train_num",       type=int,   default=1)
parser.add_argument("--grid_min",        type=float, default=-7.0)
parser.add_argument("--grid_max",        type=float, default=7.0)
parser.add_argument("--resolution",      type=int,   default=200)
parser.add_argument("--step_deg",        type=float, default=5.0)
parser.add_argument("--robot_radius",    type=float, default=0.5,
                    help="Radius of the circle the robot marker traces (m)")
parser.add_argument("--fps",             type=int,   default=8)
parser.add_argument("--output",          type=str,   default="nn_surface_anim.gif")
args = parser.parse_args()


# ===========================================================================
# MODEL — must exactly match train_surrogate.py
# ===========================================================================

"""
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim=4, hidden_size=64, hidden_layers=3):
        super().__init__()
        self.input_layer = spectral_norm(nn.Linear(input_dim, hidden_size))
        self.hidden_layers = nn.ModuleList([
            spectral_norm(nn.Linear(hidden_size, hidden_size))
            for _ in range(hidden_layers)
        ])
        self.out_layer = spectral_norm(nn.Linear(hidden_size, 1))

    def forward(self, x):
        x = torch.tanh(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        return torch.sigmoid(self.out_layer(x))
"""

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=64, hidden_layers=3):
        super().__init__()
        in_features = input_dim if input_dim != 3 else input_dim + 1
        self.input_layer = torch.nn.Linear(in_features, hidden_size)
        self.hidden_layer = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        self.out_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.shape[-1] == 3:
            sin_cos = torch.cat(
                [torch.sin(x[..., -1:]), torch.cos(x[..., -1:])], dim=-1
            )
            x = torch.cat([x[..., :-1], sin_cos], dim=-1)
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = torch.sigmoid(self.out_layer(x))   # bound to (0,1)
        return x


# ===========================================================================
# MODEL LOADING
# ===========================================================================

def find_model_path(args):
    if args.model_path is not None:
        return args.model_path

    script_dir = os.path.dirname(os.path.abspath(__file__)) \
                 if "__file__" in dir() else "."

    for base_prefix in [script_dir, os.path.join(script_dir, "..")]:
        candidate = os.path.join(
            base_prefix,
            f"trainings/{args.data_type}_trainings/"
            f"{args.dataset_version}_dataset/{args.split}/"
            f"training_{args.train_num}/saved_models/best_model_weights.pth",
        )
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find best_model_weights.pth. "
        "Pass --model_path explicitly."
    )


# ===========================================================================
# ANIMATION
# ===========================================================================

def animate_nn_surface(angles_rad, grid_min, grid_max, resolution,
                       robot_radius, fps, filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiLayerPerceptron(input_dim=4, hidden_size=64, hidden_layers=3)
    model_path = find_model_path(args)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[+] Model loaded from: {model_path}")

    # Spatial grid
    x_range = np.linspace(grid_min, grid_max, resolution)
    y_range = np.linspace(grid_min, grid_max, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    x_flat = X.flatten().astype(np.float32)
    y_flat = Y.flatten().astype(np.float32)

    print(f"[*] Pre-computing {len(angles_rad)} frames "
          f"({resolution}x{resolution} grid)...")

    """
    ####### FLAT STATIC PLOT HEATMAP #######

    angle = 40.0  # robot in front of tag10, facing the cube
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cos_flat = np.full_like(x_flat, cos_a)
    sin_flat = np.full_like(x_flat, sin_a)
    grid_points = np.stack([x_flat, y_flat, cos_flat, sin_flat], axis=-1)
    nn_input = torch.tensor(grid_points, dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = model(nn_input).cpu().numpy().flatten().reshape(X.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Z, extent=[grid_min, grid_max, grid_min, grid_max],
                origin='lower', cmap='viridis', aspect='equal',
                vmin=0.0, vmax=1.0)
    ax.plot(0, 0, 'ws', markersize=10, markeredgecolor='white', markeredgewidth=1.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Predicted confidence')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Confidence surface — azimuth {math.degrees(angle):.1f}°')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.savefig('confidence_surface_static.png', dpi=150)  
    """  

    ####### SURFACE STATIC PLOT HEATMAP #######
    
    from mpl_toolkits.mplot3d import Axes3D

    angle = 0.0  # or use the per-point facing-origin convention below
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    # Per-point "always facing the cube" convention (matches paper Fig. 6)
    dx = -x_flat
    dy = -y_flat
    norms = np.sqrt(dx**2 + dy**2) + 1e-8
    cos_flat = (dx / norms)
    sin_flat = (dy / norms)

    theta_flat = np.arctan2(sin_flat,cos_flat).astype(np.float32)
    grid_points = np.stack([x_flat, y_flat, theta_flat], axis=-1)

    #grid_points = np.stack([x_flat, y_flat, cos_flat, sin_flat], axis=-1)
    nn_input = torch.tensor(grid_points, dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = model(nn_input).cpu().numpy().flatten().reshape(X.shape)

    # Mask out the cube's physical footprint — no valid predictions inside this radius
    MIN_RADIUS = 0.0  # metres — matches the minimum data collection radius
    R = np.sqrt(X**2 + Y**2)
    Z_masked = Z.copy()
    Z_masked[R < MIN_RADIUS] = np.nan


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, Y, Z_masked,        # <-- use Z_masked here
        cmap='viridis',
        vmin=0.0, vmax=1.0,
        linewidth=0, antialiased=True, alpha=0.9,
    )

    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1).set_label('Predicted confidence')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Confidence')
    ax.set_zlim(0.0, 1.0)
    ax.set_title('Surrogate model confidence surface\n(robot always facing the cube)')
    fig.savefig('confidence_surface_3d.png', dpi=150, bbox_inches='tight')
    plt.show()


    """Z_frames = []
    for angle in angles_rad:
        cos_a = float(math.cos(angle))
        sin_a = float(math.sin(angle))
        cos_flat = np.full_like(x_flat, cos_a)
        sin_flat = np.full_like(x_flat, sin_a)
        grid_points = np.stack([x_flat, y_flat, cos_flat, sin_flat], axis=-1)
        nn_input = torch.tensor(grid_points, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(nn_input).cpu().numpy().flatten()
        Z_frames.append(outputs.reshape(X.shape))

    # ── Figure layout ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    # Heatmap — fixed colorbar [0,1]
    im = ax.imshow(
        Z_frames[0],
        extent=[grid_min, grid_max, grid_min, grid_max],
        origin="lower", cmap="viridis", aspect="equal",
        vmin=0.0, vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted confidence (yolo_conf)")

    # Cube at origin
    ax.plot(0, 0, "ws", markersize=10, markeredgecolor="white",
            markeredgewidth=1.5, label="Cube (origin)", zorder=5)

    # Orbit path — dashed circle to show the trajectory
    orbit = plt.Circle((0, 0), robot_radius, color="white",
                        fill=False, linestyle="--", linewidth=1.0,
                        alpha=0.5, zorder=3)
    ax.add_patch(orbit)

    # Robot marker — starts at angle[0]
    rx0 = robot_radius * math.cos(angles_rad[0])
    ry0 = robot_radius * math.sin(angles_rad[0])
    robot_dot, = ax.plot(rx0, ry0, "o", color="white", markersize=10,
                         markeredgecolor="black", markeredgewidth=1.5,
                         zorder=6, label="Robot")

    # Arrow from robot toward cube (shows "facing the cube" direction)
    arrow_obj = ax.annotate(
        "", xy=(0, 0), xytext=(rx0, ry0),
        arrowprops=dict(arrowstyle="-|>", color="white",
                        lw=1.5, mutation_scale=12),
        zorder=7,
    )

    ax.set_xlabel("x  (m)")
    ax.set_ylabel("y  (m)")
    ax.set_xlim(grid_min, grid_max)
    ax.set_ylim(grid_min, grid_max)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8,
              facecolor="black", labelcolor="white")

    def update(frame_idx):
        angle = angles_rad[frame_idx]

        # Update heatmap
        im.set_data(Z_frames[frame_idx])

        # Update robot position on orbit
        rx = robot_radius * math.cos(angle)
        ry = robot_radius * math.sin(angle)
        robot_dot.set_data([rx], [ry])

        # Update arrow: from robot marker toward origin (cube)
        arrow_obj.set_position((rx, ry))   # tail
        arrow_obj.xy = (0, 0)              # head (cube)

        ax.set_title(
            f"Azimuth: {math.degrees(angle):.1f}°  "
            f"[cos={math.cos(angle):.2f}, sin={math.sin(angle):.2f}]",
            color="white" if False else "black",
        )

        return [im, robot_dot, arrow_obj]


    print("[*] Rendering GIF...")
    ani = animation.FuncAnimation(
        fig, update, frames=len(angles_rad), blit=False, repeat=True
    )
    ani.save(filename, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"[+] Animation saved to: {filename}")
    """

# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    
    
    step_rad = math.radians(args.step_deg)
    angles   = np.arange(-math.pi, math.pi + 1e-5, step_rad)

    animate_nn_surface(
        angles_rad   = angles,
        grid_min     = args.grid_min,
        grid_max     = args.grid_max,
        resolution   = args.resolution,
        robot_radius = args.robot_radius,
        fps          = args.fps,
        filename     = args.output,
    )
