#!/usr/bin/env python3
import os
import re
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

# =============================================================================
# CONFIGURAZIONE GENERALE E PARAMETRI GRAFICI
# =============================================================================

save_dir = "/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/nmpc_ros/CSV_PAPER/plots"
filename = "surrogate.svg"

# --- Typography & Font Sizes ---
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10
FONT_SIZE_CBAR = 10
FONT_SIZE_CONTOUR_LABELS = 8

# --- Grid & Mesh Settings ---
X_RANGE_DEFAULT = (-6.0, 6.0)
Y_RANGE_DEFAULT = (-6.0, 6.0)
PLOT_XLIM = (-6.0, 6.0)
PLOT_YLIM = (-6.0, 6.0)
DEFAULT_RESOLUTION = 300

# --- Vehicle & Target Coordinates ---
# Dimensione e posizionamento sagoma (es. Citroën C1 + margine)
CAR_LENGTH = 3.46 + 0.4
CAR_WIDTH = 1.62 + 0.4
CAR_FRONT_TIP_X = -0.10

# Punto di mira (Sedili anteriori / Abitacolo)
TARGET_X = -1.30
TARGET_Y = 0.0

# --- Plot Styles & Colors ---
COLORMAP_NAME = 'viridis'
CONTOUR_LEVELS_FILLED = 50
CONTOUR_LEVELS_LINES = 22

COLOR_CAR_EDGE = 'red'
COLOR_CAR_FACE = 'red'
ALPHA_CAR = 0.99

COLOR_REF_MARKER = 'red'
MARKER_REF_STYLE = 'x'
MARKER_REF_SIZE = 8

COLOR_AXIS_LINES = 'gray'
ALPHA_AXIS_LINES = 0.6
GRID_ALPHA = 0.5

# =============================================================================
# MODELLO NEURALE
# =============================================================================

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim=3, hidden_size=64, hidden_layers=3):
        super().__init__()
        in_features = input_dim if input_dim != 3 else input_dim + 1
        self.input_layer = nn.Linear(in_features, hidden_size)
        self.hidden_layer = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.shape[-1] == 3:
            sin_cos = torch.cat([torch.sin(x[..., -1:]), torch.cos(x[..., -1:])], dim=-1)
            x = torch.cat([x[..., :-1], sin_cos], dim=-1)
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x

# =============================================================================
# UTILITIES
# =============================================================================

def get_latest_best_model(cls='car'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models", cls)
    if not os.path.exists(model_dir):
        model_dir = os.path.join(script_dir, "..", "models", cls)

    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if re.match(r"best_model_epoch_(\d+)\.pth", f)]
        if model_files:
            latest_model = max(model_files, key=lambda x: int(re.match(r"best_model_epoch_(\d+)\.pth", x).group(1)))
            return os.path.join(model_dir, latest_model)
    return None

def draw_realistic_car(ax, x_front, car_length, car_width, body_color='#D32F2F'):
    """Disegna la sagoma vettoriale vista dall'alto dell'automobile."""
    x_rear = x_front - car_length
    half_w = car_width / 2.0
    
    nos_x = x_front
    rear_x = x_rear
    hood_x = x_front - 0.20 * car_length
    windshield_top_x = x_front - 0.38 * car_length
    roof_rear_x = x_front - 0.72 * car_length
    rear_glass_end_x = x_front - 0.88 * car_length
    
    w_body = half_w
    w_roof = half_w * 0.72
    w_nose = half_w * 0.82
    
    # Carrozzeria
    body_pts = [
        [nos_x, -w_nose], [hood_x, -w_body], [rear_x + 0.1, -w_body],
        [rear_x, -w_nose], [rear_x, w_nose], [rear_x + 0.1, w_body],
        [hood_x, w_body], [nos_x, w_nose]
    ]
    ax.add_patch(Polygon(body_pts, closed=True, facecolor=body_color, edgecolor='black', linewidth=1.5, zorder=3, label='Car'))

    # Ruote
    wheel_l, wheel_w = car_length * 0.16, car_width * 0.12
    wheel_positions = [
        (hood_x - wheel_l * 0.3, half_w - wheel_w * 0.5),
        (hood_x - wheel_l * 0.3, -half_w - wheel_w * 0.5),
        (roof_rear_x - wheel_l * 0.2, half_w - wheel_w * 0.5),
        (roof_rear_x - wheel_l * 0.2, -half_w - wheel_w * 0.5)
    ]
    for wx, wy in wheel_positions:
        ax.add_patch(Rectangle((wx, wy), wheel_l, wheel_w, facecolor='#111111', edgecolor='black', linewidth=0.8, zorder=2))

    # Vetri (Parabrezza, Lunotto, Finestrini)
    glass_color = '#263238'
    ax.add_patch(Polygon([[hood_x, -w_body * 0.82], [windshield_top_x, -w_roof], [windshield_top_x, w_roof], [hood_x, w_body * 0.82]], closed=True, facecolor=glass_color, edgecolor='#B0BEC5', linewidth=0.8, zorder=4))
    ax.add_patch(Polygon([[roof_rear_x, -w_roof], [rear_glass_end_x, -w_body * 0.78], [rear_glass_end_x, w_body * 0.78], [roof_rear_x, w_roof]], closed=True, facecolor=glass_color, edgecolor='#B0BEC5', linewidth=0.8, zorder=4))
    ax.add_patch(Polygon([[windshield_top_x + 0.02, w_roof], [roof_rear_x - 0.02, w_roof], [rear_glass_end_x, w_body * 0.75], [hood_x, w_body * 0.79]], closed=True, facecolor=glass_color, edgecolor='#B0BEC5', linewidth=0.6, zorder=4))
    ax.add_patch(Polygon([[windshield_top_x + 0.02, -w_roof], [roof_rear_x - 0.02, -w_roof], [rear_glass_end_x, -w_body * 0.75], [hood_x, -w_body * 0.79]], closed=True, facecolor=glass_color, edgecolor='#B0BEC5', linewidth=0.6, zorder=4))

    # Tetto
    ax.add_patch(Polygon([[windshield_top_x, -w_roof], [roof_rear_x, -w_roof], [roof_rear_x, w_roof], [windshield_top_x, w_roof]], closed=True, facecolor=body_color, edgecolor='#424242', linewidth=0.5, zorder=4))

    # Fari Anteriore (Gialli) e Posteriore (Rossi)
    hl = car_length * 0.06
    ax.add_patch(Rectangle((nos_x - hl, w_nose * 0.55), hl, w_nose * 0.35, facecolor='#FFF59D', edgecolor='orange', linewidth=0.8, zorder=5))
    ax.add_patch(Rectangle((nos_x - hl, -w_nose * 0.90), hl, w_nose * 0.35, facecolor='#FFF59D', edgecolor='orange', linewidth=0.8, zorder=5))
    ax.add_patch(Rectangle((rear_x, w_nose * 0.55), hl, w_nose * 0.35, facecolor='#FF1744', edgecolor='#B71C1C', linewidth=0.8, zorder=5))
    ax.add_patch(Rectangle((rear_x, -w_nose * 0.90), hl, w_nose * 0.35, facecolor='#FF1744', edgecolor='#B71C1C', linewidth=0.8, zorder=5))

    # Specchietti
    ml, mw = car_length * 0.05, car_width * 0.08
    ax.add_patch(Rectangle((windshield_top_x, w_body), ml, mw, facecolor='#212121', zorder=5))
    ax.add_patch(Rectangle((windshield_top_x, -w_body - mw), ml, mw, facecolor='#212121', zorder=5))

# =============================================================================
# VISUALIZZAZIONE
# =============================================================================

def plot_model_contour(
    model_path=None, 
    x_range=X_RANGE_DEFAULT, 
    y_range=Y_RANGE_DEFAULT, 
    resolution=DEFAULT_RESOLUTION
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Caricamento del modello
    model = MultiLayerPerceptron(input_dim=3, hidden_size=64, hidden_layers=3)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[+] Modello caricato da: {model_path}")
    else:
        print("[!] Nessun file di pesi trovato o specificato: eseguo con pesi casuali.")
    
    model.to(device)
    model.eval()

    # 2. Creazione griglia 2D (x_rel, y_rel) nella terna della macchina
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # 3. Calcolo orientamento rispetto al punto di mira
    dx_target = TARGET_X - X
    dy_target = TARGET_Y - Y

    # Angolo dell'asse Y del robot orientato verso i sedili anteriori
    theta_y_robot_car = np.arctan2(dy_target, dx_target)
    
    # Azimut normalizzato coerente con la formula
    azimuth_raw = theta_y_robot_car + np.pi
    azimuth_norm = np.arctan2(np.sin(azimuth_raw), np.cos(azimuth_raw))

    # 4. Inserimento nella rete neurale e inferenza
    grid_points = np.stack([X.ravel(), Y.ravel(), azimuth_norm.ravel()], axis=-1)
    input_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(input_tensor).cpu().numpy().reshape(X.shape)

    # 5. Visualizzazione Grafica
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Mappa di colore riempita (Contourf)
    contour_filled = ax.pcolormesh(
    X, Y, preds, 
    cmap=COLORMAP_NAME, 
    shading='gouraud', 
    rasterized=True  # Mantiene lo sfondo rasterizzato nell'SVG per zero artefatti
    )
    
    # Colorbar con gestione dimensione font
    cbar = fig.colorbar(contour_filled, ax=ax)
    cbar.ax.tick_params(labelsize=FONT_SIZE_CBAR)

    # Linee di livello opzionali (scommentare se necessarie)
    # contour_lines = ax.contour(X, Y, preds, levels=CONTOUR_LEVELS_LINES, colors='black', linewidths=0.5, alpha=0.7)
    # ax.clabel(contour_lines, inline=True, fontsize=FONT_SIZE_CONTOUR_LABELS, fmt='%.2f')

    # Posizionamento sagoma auto
    # x_car_origin = CAR_FRONT_TIP_X - CAR_LENGTH

    # Disegno Sagoma dell'Auto
    # car_rect = Rectangle(
    #     (x_car_origin, -CAR_WIDTH / 2.0),
    #     CAR_LENGTH,
    #     CAR_WIDTH,
    #     linewidth=2,
    #     edgecolor=COLOR_CAR_EDGE,
    #     facecolor=COLOR_CAR_FACE,
    #     alpha=ALPHA_CAR,
    #     label='Car'
    # )
    # ax.add_patch(car_rect)

    draw_realistic_car(
        ax, 
        x_front=CAR_FRONT_TIP_X, 
        car_length=CAR_LENGTH, 
        car_width=CAR_WIDTH,
        body_color='#E53935'
    )

    # Evidenzia la Punta Frontale (Riferimento 0,0 della mesh)
    # ax.plot(
    #     0, 0, 
    #     marker=MARKER_REF_STYLE, 
    #     color=COLOR_REF_MARKER, 
    #     markersize=MARKER_REF_SIZE, 
    #     markeredgewidth=2, 
    #     label='Reference'
    # )

    # Assi e Griglie
    ax.axhline(0, color=COLOR_AXIS_LINES, linestyle='--', linewidth=0.8, alpha=ALPHA_AXIS_LINES)
    ax.axvline(0, color=COLOR_AXIS_LINES, linestyle='--', linewidth=0.8, alpha=ALPHA_AXIS_LINES)
    
    # Font degli assi, tick e legenda
    ax.set_xlabel(r"$x$", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"$y$", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    # ax.legend(loc='upper right', fontsize=FONT_SIZE_LEGEND)
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=GRID_ALPHA)

    ax.set_xlim(PLOT_XLIM)
    ax.set_ylim(PLOT_YLIM)

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, filename), format='svg', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    try:
        latest_model = get_latest_best_model('car')
    except Exception:
        latest_model = None

    plot_model_contour(model_path=latest_model)