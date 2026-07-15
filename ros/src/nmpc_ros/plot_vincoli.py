#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. PARAMETRI ESTRAZIONI DAL TUO CODICE
# ==========================================

# Parametri Pali
safe_radius_pole = 0.8
poles_pos = np.array([
    [7.1905, 5.7978],
    [9.7331, 5.1122],
    [16.6093, 4.0214],
    [20.3781, 3.3552],
    [26.0805, 2.1324],
    [29.9115, 1.6864],
    [35.5549, 0.3543],
    [39.2697, -0.2548],
    [45.0900, -1.2590],
    [48.4675, -1.9691],
    [54.8411, -3.1006],
    [58.1188, -3.4193],
], dtype=np.float32)

# Parametri Aiuole (x, y, lunghezza, larghezza, orientamento_rad)
margin_f = 0.5
flowerbeds_pos = np.array([
    [2.1879, -5.5748, 4.6071, 4.5434, -0.175],    # aiuola 1
    [64.6609, -16.5047, 2.8938, 4.5661, -0.175],  # aiuola 2
    [32.1876, -14.3858, 66.7007, 3.5897, -0.175], # aiuola 3
    [3.8231, 4.0245, 1.6749, 4.7252, -0.175],     # aiuola 4
    [5.7205, 7.9094, 4.5549, 4.5383, -0.175],     # aiuola 5
    [61.5571, -4.8938, 2.4004, 9.0567, -0.175],   # aiuola 6
    [9.7648, 16.3899, 2.0120, 4.6903, -0.175],    # aiuola 7
    [58.9274, 7.9500, 2.4176, 4.3564, -0.175],    # aiuola 8
    [34.1108, 15.1653, 50.4020, 1.5882, -0.175],  # aiuola 9
], dtype=np.float32)

# Parametri Auto/Ostacoli
car_length = 3.46 
car_width = 1.62   
margin_x = 1.0    
margin_y = 1.0 
core_safety = 0.8  # Percentuale della superellisse considerata "Hard Constraint"

# Caricamento Posizione Auto (Ostacoli)
trees_pos = None
file_path = "/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/nmpc_ros/niccolo/car_map_final.json"

if os.path.exists(file_path):
    print("Caricamento ostacoli da JSON...")
    with open(file_path, "r", encoding="utf-8") as f:
        cars = json.load(f).get("cars", [])
    extracted = []
    for c in cars:
        extracted.append([c["x"], c["y"], c["orientation_rad"]])
    trees_pos = np.array(extracted, dtype=np.float32)
else:
    print("File JSON non trovato. Utilizzo ostacoli hardcoded (fallback)...")
    trees_pos = np.array([
        [43.04, -2.149, 1.4337],
        [49.701, -7.093, -1.6995],
        [52.743, -7.233, -1.7331],
    ], dtype=np.float32)


# ==========================================
# 2. FUNZIONI GEOMETRICHE
# ==========================================

def generate_superellipse(center_x, center_y, a, b, theta, n=4, num_points=200):
    """
    Genera i punti (x, y) di una superellisse ruotata.
    Equazione base: |x/a|^n + |y/b|^n = 1
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    
    # Parametrizzazione della superellisse
    power = 2.0 / n
    x_rel = a * np.sign(np.cos(t)) * (np.abs(np.cos(t)) ** power)
    y_rel = b * np.sign(np.sin(t)) * (np.abs(np.sin(t)) ** power)
    
    # Rotazione dal sistema locale a quello globale (inverso di quanto fa il MPC)
    dx = x_rel * np.cos(theta) - y_rel * np.sin(theta)
    dy = x_rel * np.sin(theta) + y_rel * np.cos(theta)
    
    return center_x + dx, center_y + dy


# ==========================================
# 3. PLOT DEI VINCOLI HARD
# ==========================================

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_title("Mappa dei Vincoli Hard (Spazio inaccessibile al robot)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

# --- PLOT PALI ---
for p in poles_pos:
    circle = patches.Circle((p[0], p[1]), safe_radius_pole, 
                            color='red', alpha=0.5, label='Pali (Hard)')
    ax.add_patch(circle)
    # Marcatore del centro originario
    ax.plot(p[0], p[1], 'rx', markersize=5)

# --- PLOT AIUOLE ---
for f in flowerbeds_pos:
    fx, fy, flen, fwid, ftheta = f[0], f[1], f[2], f[3], f[4]
    
    # Assi della superellisse con margine
    sigma_x_f = (flen / 2.0) + margin_f
    sigma_y_f = (fwid / 2.0) + margin_f
    
    x_se, y_se = generate_superellipse(fx, fy, sigma_x_f, sigma_y_f, ftheta)
    ax.fill(x_se, y_se, color='green', alpha=0.4, label='Aiuole (Hard)')
    ax.plot(fx, fy, 'g+', markersize=8) # Centro

# --- PLOT AUTO / OSTACOLI ---
for obs in trees_pos:
    car_x, car_y, car_theta = obs[0], obs[1], obs[2]
    
    # Traslazione del centro vettura usata nel tuo MPC
    center_x = car_x - (car_length / 2.0) * np.cos(car_theta)
    center_y = car_y - (car_length / 2.0) * np.sin(car_theta)
    
    sigma_x = (car_length / 2.0) + margin_x
    sigma_y = (car_width / 2.0) + margin_y
    
    # Nel MPC hai dist_norm = (x_rel/sigma_x)^4 + (y_rel/sigma_y)^4
    # E il vincolo hard è: dist_norm >= core_safety**4
    # Questo equivale a una superellisse con semiassi scalati da core_safety
    a_hard = sigma_x * core_safety
    b_hard = sigma_y * core_safety
    
    x_se, y_se = generate_superellipse(center_x, center_y, a_hard, b_hard, car_theta)
    ax.fill(x_se, y_se, color='purple', alpha=0.5, label='Auto/Ostacoli (Hard)')
    
    # Plot del centro geometrico traslato
    ax.plot(center_x, center_y, 'm.', markersize=8)
    # Plot dell'ancoraggio (punta del muso dell'auto da cui misuri la traslazione)
    ax.plot(car_x, car_y, 'k^', markersize=6)


# Fix dei duplicati nella legenda
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left')

plt.tight_layout()
plt.show()