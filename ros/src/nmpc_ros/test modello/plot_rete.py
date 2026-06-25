import os
import re
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Riproduzione della struttura della rete ---
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
            sin_cos = torch.cat([torch.sin(x[..., -1:]), torch.cos(x[..., -1:])], dim=-1)
            x = torch.cat([x[..., :-1], sin_cos], dim=-1)
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x

def get_latest_best_model(cls='car'):
    """Funzione helper per trovare l'ultimo modello"""
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    model_dir = os.path.join(script_dir, "models", cls)
    if not os.path.exists(model_dir):
        model_dir = os.path.join(script_dir, "..", "models", cls) 
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Cartella modelli non trovata in {model_dir}. Assicurati di eseguire lo script vicino alla cartella 'models'.")
        
    model_files = [f for f in os.listdir(model_dir) if re.match(r"best_model_epoch_(\d+)\.pth", f)]
    if not model_files:
        raise FileNotFoundError(f"Nessun modello trovato in {model_dir}")
    latest_model = max(model_files, key=lambda x: int(re.match(r"best_model_epoch_(\d+)\.pth", x).group(1)))
    return os.path.join(model_dir, latest_model)

def animate_nn_surface(angoli_rad, grid_min=-5.0, grid_max=5.0, resolution=200, filename="nn_surface_animation.gif"):
    """
    Genera un'animazione GIF con la risposta della rete neurale al variare dell'azimuth.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Caricamento del modello
    model = MultiLayerPerceptron(input_dim=3, hidden_size=64, hidden_layers=3)
    try:
        model_path = get_latest_best_model('car')
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[+] Modello caricato correttamente da: {model_path}")
    except Exception as e:
        print(f"[-] Errore nel caricamento del modello reale: {e}")
        print("[!] Genero pesi casuali solo a scopo dimostrativo per l'animazione.")
    
    model.to(device)
    model.eval()

    # 2. Generazione della griglia spaziale X, Y
    x_range = np.linspace(grid_min, grid_max, resolution)
    y_range = np.linspace(grid_min, grid_max, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    x_flat = X.flatten()
    y_flat = Y.flatten()

    print("[*] Calcolo delle predizioni della rete in corso...")
    
    # 3. Pre-calcolo di tutti i frame per trovare min/max globali (evita sfarfallio dei colori)
    Z_frames = []
    for angle in angoli_rad:
        angle_flat = np.full_like(x_flat, angle)
        grid_points = np.stack([x_flat, y_flat, angle_flat], axis=-1)
        nn_input = torch.tensor(grid_points, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(nn_input).cpu().numpy()
        
        Z_frames.append(outputs.reshape(X.shape))
        
    Z_min, Z_max = np.min(Z_frames), np.max(Z_frames)

    # 4. Configurazione della Figura singola
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Inizializza l'immagine con il primo frame
    im = ax.imshow(Z_frames[0], extent=[grid_min, grid_max, grid_min, grid_max], 
                   origin='lower', cmap='viridis', aspect='equal',
                   vmin=Z_min, vmax=Z_max) # Blocca la scala dei colori sui limiti globali
    
    ax.plot(0, 0, 'ro', markersize=8, label='Centro Macchina')
    ax.set_xlabel("x_rel (m)")
    ax.set_ylabel("y_rel (m)")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Colorbar fissa
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 5. Funzione di Update per l'animazione
    def update(frame_idx):
        # Aggiorna la matrice dell'immagine
        im.set_data(Z_frames[frame_idx])
        
        # Aggiorna il titolo
        angle = angoli_rad[frame_idx]
        angle_deg = math.degrees(angle)
        ax.set_title(f"Azimuth: {angle:.2f} rad ({angle_deg:.1f}°)")
        
        return [im]

    print("[*] Rendering della GIF in corso...")
    
    # Crea l'animazione
    ani = animation.FuncAnimation(
        fig, update, frames=len(angoli_rad), blit=False, repeat=True
    )
    
    # 6. Salvataggio
    # Nota: Assicurati di avere il modulo Pillow installato (`pip install pillow`)
    ani.save(filename, writer='pillow', fps=8)
    
    print(f"[+] Animazione salvata con successo in: {filename}")
    plt.close(fig) # Chiudi la figura per ripulire la memoria

if __name__ == '__main__':
    # Genera angoli da -180° a +180°
    passo_gradi = 5 
    passo_rad = math.radians(passo_gradi)
    angoli_da_testare = np.arange(-math.pi, math.pi + 1e-5, passo_rad)
        
    animate_nn_surface(
        angoli_da_testare, 
        grid_min=-7.0, 
        grid_max=7.0, 
        filename="nn_surface_anim.gif"
    )