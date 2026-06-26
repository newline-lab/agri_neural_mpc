import os
import re
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

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
        # x = torch.sigmoid(self.out_layer(x))   # bound to (0,1)
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

def plot_nn_surfaces(angoli_gradi, grid_min=-7.0, grid_max=7.0, resolution=200):
    """
    Genera un plot statico con la risposta della rete neurale per gli angoli (in gradi) specificati.
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
        print("[!] Utilizzo pesi casuali a scopo dimostrativo.")
    
    model.to(device)
    model.eval()

    # 2. Generazione della griglia spaziale X, Y
    x_range = np.linspace(grid_min, grid_max, resolution)
    y_range = np.linspace(grid_min, grid_max, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    x_flat = X.flatten()
    y_flat = Y.flatten()

    # 3. Setup della griglia di plot (Subplots)
    num_plots = len(angoli_gradi)
    # Imposta un massimo di 3 colonne, calcola le righe necessarie
    cols = min(num_plots, 3)
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    
    # Rendi 'axes' un array 1D per iterarci facilmente sopra
    if num_plots == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    print("[*] Calcolo delle predizioni e generazione dei grafici...")

    # 4. Calcolo e Plot per ogni angolo
    for i, angolo_deg in enumerate(angoli_gradi):
        # Conversione Gradi -> Radianti
        angolo_rad = math.radians(angolo_deg)
        
        # Preparazione input
        angle_flat = np.full_like(x_flat, angolo_rad)
        grid_points = np.stack([x_flat, y_flat, angle_flat], axis=-1)
        nn_input = torch.tensor(grid_points, dtype=torch.float32).to(device)
        
        # Inferenza
        with torch.no_grad():
            outputs = model(nn_input).cpu().numpy()
        Z = outputs.reshape(X.shape)
        
        # Disegno sul subplot corrispondente
        ax = axes[i]
        im = ax.imshow(Z, extent=[grid_min, grid_max, grid_min, grid_max], 
                       origin='lower', cmap='viridis', aspect='equal',
                       vmin=0.0, vmax=1.0) # Fisso la scala tra 0 e 1 vista la sigmoid
        
        ax.plot(0, 0, 'ro', markersize=6, label='Centro Macchina')
        ax.set_title(f"Azimuth: {angolo_deg}° ({angolo_rad:.2f} rad)")
        ax.set_xlabel("x_rel (m)")
        ax.set_ylabel("y_rel (m)")
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Nascondi eventuali subplot vuoti se il numero di angoli non riempie la griglia
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # ---> INSERISCI QUI GLI ANGOLI IN GRADI CHE VUOI TESTARE <---
    angoli_scelti_in_gradi = [0, 45, 90, 135, 180, -90] 
        
    plot_nn_surfaces(
        angoli_scelti_in_gradi, 
        grid_min=-7.0, 
        grid_max=7.0
    )