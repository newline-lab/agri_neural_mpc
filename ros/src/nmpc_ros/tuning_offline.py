import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

# ==============================================================================
# 1. TUNING DEI GUADAGNI E PARAMETRI
# ==============================================================================

theta_rob = 0.0      
robot_length = 1.0 
robot_width = 0.67 

margin_pole = 0.8 
Q_pole_hard = 100.0   
alpha_pole = 10.0      

margin_f = 1.0  
Q_flowerbed_hard = 50.0 
alpha_flowerbed = 5.0    

car_length = 3.46 
car_width = 1.62   
A_rep = 0.1     
margin_x = 1    
margin_y = 1    
Q_circ_ori = 200.0 

Q_thresh_pos = 5.0
TARGET_CAR_INDEX = 0

# La lista esatta dei tuoi punti soglia attivi
punti_soglia = [
    [-1.3037, -2.5762, -2.0944],
    [-0.934, -2.3745, -1.5708],
    [3.2101, 0.1273, -0.0],
    [0.1753, 3.5078, 1.0472],
    [-1.1693, 3.066, 1.5708],
    [-0.4298, 2.9508, 2.0944],
]

# ==============================================================================
# 2. DEFINIZIONE DELLA MAPPA E DEGLI OGGETTI
# ==============================================================================

trees_pos = np.array([
    [43.04, -2.149, 1.4337],
    [49.701, -7.093, -1.6995],
    [52.743, -7.233, -1.7331],
], dtype=np.float32)

poles_pos = np.array([
    [7.1905, 5.7978], [9.7331, 5.1122], [16.6093, 4.0214],
    [20.3781, 3.3552], [26.0805, 2.1324], [29.9115, 1.6864],
    [35.5549, 0.3543], [39.2697, -0.2548], [45.0900, -1.2590],
    [48.4675, -1.9691], [54.8411, -3.1006], [58.1188, -3.4193],
], dtype=np.float32)

flowerbeds_pos = np.array([
    [2.1879, -5.5748, 4.6071, 4.5434, -0.175],
    [64.6609, -16.5047, 2.8938, 4.5661, -0.175],
    [32.1876, -14.3858, 66.7007, 3.5897, -0.175],
    [3.8231, 4.0245, 1.6749, 4.7252, -0.175],
    [5.7205, 7.9094, 4.5549, 4.5383, -0.175],
    [61.5571, -4.8938, 2.4004, 9.0567, -0.175],
    [9.7648, 16.3899, 2.0120, 4.6903, -0.175],
    [58.9274, 7.9500, 2.4176, 4.3564, -0.175],
    [34.1108, 15.1653, 50.4020, 1.5882, -0.175],
], dtype=np.float32)

# Zoom sulla zona della prima macchina per vedere meglio i punti
x_min, x_max = 35, 55
y_min, y_max = -10, 10
X, Y = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

# ==============================================================================
# 3. CALCOLO DEL POTENZIALE STATICO (OSTACOLI)
# ==============================================================================
Z_obstacles = np.zeros_like(X)

# Pali
sigma_x_rob_pole = (robot_length / 2.0) + margin_pole
sigma_y_rob_pole = (robot_width / 2.0) + margin_pole
for p in poles_pos:
    px, py = p[0], p[1]
    dx_p = px - X
    dy_p = py - Y
    x_rel_p = dx_p * np.cos(theta_rob) + dy_p * np.sin(theta_rob)
    y_rel_p = -dx_p * np.sin(theta_rob) + dy_p * np.cos(theta_rob)
    dist_norm_p = (x_rel_p / sigma_x_rob_pole)**2 + (y_rel_p / sigma_y_rob_pole)**2
    Z_obstacles += Q_pole_hard * np.exp(alpha_pole * (1.0 - dist_norm_p))

# Aiuole
for f in flowerbeds_pos:
    fx, fy, flen, fwid, ftheta = f[0], f[1], f[2], f[3], f[4]
    dx_f = X - fx
    dy_f = Y - fy
    x_rel_f = dx_f * np.cos(ftheta) + dy_f * np.sin(ftheta)
    y_rel_f = -dx_f * np.sin(ftheta) + dy_f * np.cos(ftheta)
    sigma_x_f = (flen / 2.0) + margin_f
    sigma_y_f = (fwid / 2.0) + margin_f
    dist_norm_f = (x_rel_f / sigma_x_f)**4 + (y_rel_f / sigma_y_f)**4
    Z_obstacles += Q_flowerbed_hard * np.exp(alpha_flowerbed * (1.0 - dist_norm_f))

# Auto
sigma_x_car = (car_length / 2.0) + margin_x
sigma_y_car = (car_width / 2.0) + margin_y
for i, car in enumerate(trees_pos):
    car_x, car_y, car_theta = car[0], car[1], car[2]
    center_x = car_x - (car_length / 2.0) * np.cos(car_theta)
    center_y = car_y - (car_length / 2.0) * np.sin(car_theta)
    dx = X - center_x
    dy = Y - center_y
    x_rel = dx * np.cos(car_theta) + dy * np.sin(car_theta)
    y_rel = -dx * np.sin(car_theta) + dy * np.cos(car_theta)
    dist_norm = (x_rel / sigma_x_car)**4 + (y_rel / sigma_y_car)**4
    
    Z_obstacles += A_rep * dist_norm
    Z_obstacles += Q_circ_ori * np.exp(-dist_norm)

# Coordinate dell'auto target per la proiezione
target_car = trees_pos[TARGET_CAR_INDEX]
tx, ty, theta_target = target_car[0], target_car[1], target_car[2]
dX = X - tx
dY = Y - ty
x_rel_t = dX * np.cos(theta_target) + dY * np.sin(theta_target)
y_rel_t = -dX * np.sin(theta_target) + dY * np.cos(theta_target)

# ==============================================================================
# 4. PLOT MATPLOTLIB A GRIGLIA
# ==============================================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Analisi Campi Potenziali: I 6 Punti Soglia Attivi", fontsize=18)
VMAX_VISUAL = 200.0  

for idx, ax in enumerate(axes.flatten()):
    if idx >= len(punti_soglia):
        ax.axis('off')
        continue
        
    thresh = punti_soglia[idx]
    
    # Calcolo dell'attrazione specifica per questo punto soglia
    dist_to_thresh_sq = (x_rel_t - thresh[0])**2 + (y_rel_t - thresh[1])**2
    Z_attraction = Q_thresh_pos * dist_to_thresh_sq
    
    # Somma potenziale statico + attrazione attiva
    Z_total = Z_obstacles + Z_attraction
    Z_plot = np.clip(Z_total, a_min=None, a_max=VMAX_VISUAL)
    
    # Plot contour
    contour = ax.contourf(X, Y, Z_plot, levels=40, cmap='viridis_r', extend='max')
    
    # Disegno ostacoli
    for p in poles_pos:
        ax.add_patch(plt.Circle((p[0], p[1]), 0.5, color='black'))

    for f in flowerbeds_pos:
        fx, fy, flen, fwid, ftheta = f[0], f[1], f[2], f[3], f[4]
        rect = plt.Rectangle((fx - flen/2, fy - fwid/2), flen, fwid, fill=False, edgecolor='white', lw=1.5)
        t = transforms.Affine2D().rotate_around(fx, fy, ftheta) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    for i, car in enumerate(trees_pos):
        car_x, car_y, car_theta = car[0], car[1], car[2]
        center_x = car_x - (car_length / 2.0) * np.cos(car_theta)
        center_y = car_y - (car_length / 2.0) * np.sin(car_theta)
        rect = plt.Rectangle((center_x - car_length/2, center_y - car_width/2), car_length, car_width, 
                             fill=True, color='red' if i == TARGET_CAR_INDEX else 'orange', alpha=0.6)
        t = transforms.Affine2D().rotate_around(center_x, center_y, car_theta) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        ax.arrow(center_x, center_y, np.cos(car_theta), np.sin(car_theta), head_width=0.4, color='black')

    # Calcolo posizionamento globale del punto soglia corrente
    opt_x_glob = tx + (thresh[0] * np.cos(theta_target)) - (thresh[1] * np.sin(theta_target))
    opt_y_glob = ty + (thresh[0] * np.sin(theta_target)) + (thresh[1] * np.cos(theta_target))
    
    # Plot punto soglia
    ax.plot(opt_x_glob, opt_y_glob, 'x', color='cyan', markersize=14, markeredgewidth=4)
    
    ax.set_title(f"Soglia {idx+1}: [x={thresh[0]:.2f}, y={thresh[1]:.2f}]", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()