import os
import re
import math
import numpy as np
import torch

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
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    model_dir = os.path.join(script_dir, "models", cls)
    if not os.path.exists(model_dir):
        model_dir = os.path.join(script_dir, "..", "models", cls) 
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Cartella modelli non trovata in {model_dir}")
        
    model_files = [f for f in os.listdir(model_dir) if re.match(r"best_model_epoch_(\d+)\.pth", f)]
    if not model_files:
        raise FileNotFoundError(f"Nessun modello trovato in {model_dir}")
    latest_model = max(model_files, key=lambda x: int(re.match(r"best_model_epoch_(\d+)\.pth", x).group(1)))
    return os.path.join(model_dir, latest_model)

def extract_points_above_threshold(angoli_rad, grid_min=-5.0, grid_max=5.0, resolution=100, threshold=1.0):
    """
    Trova e restituisce solo il punto [x, y, angolo] a confidenza MASSIMA 
    per ciascun angolo, purché superi la soglia inserita.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Caricamento del modello
    model = MultiLayerPerceptron(input_dim=3, hidden_size=64, hidden_layers=3)
    try:
        model_path = get_latest_best_model('car')
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"# [+] Modello caricato da: {model_path}") 
    except Exception as e:
        print(f"# [-] Modello reale non trovato. Uso pesi casuali. Errore: {e}")
    
    model.to(device)
    model.eval()

    # 2. Generazione della griglia spaziale X, Y
    x_range = np.linspace(grid_min, grid_max, resolution)
    y_range = np.linspace(grid_min, grid_max, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    x_flat = X.flatten()
    y_flat = Y.flatten()

    print(f"# [*] Ricerca dei massimi con risoluzione {resolution}x{resolution}...")
    
    valid_points = []

    # 3. Valutazione e ricerca del punto massimo per ogni angolo
    for angle in angoli_rad:
        angle_flat = np.full_like(x_flat, angle)
        grid_points = np.stack([x_flat, y_flat, angle_flat], axis=-1)
        nn_input = torch.tensor(grid_points, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(nn_input).cpu().numpy().flatten()
        
        # Trova l'indice del valore massimo assoluto per QUESTO angolo
        idx_max = np.argmax(outputs)
        valore_massimo = outputs[idx_max]
        
        # Controlla se il picco massimo supera comunque la tua soglia minima
        if valore_massimo > threshold:
            pt_x = round(float(x_flat[idx_max]), 4)
            pt_y = round(float(y_flat[idx_max]), 4)
            pt_a = round(float(angle), 4)
            valid_points.append([pt_x, pt_y, pt_a])
            
    return valid_points

if __name__ == '__main__':
    # Riduciamo il passo per ottenere meno risultati in console (es. 30 gradi invece di 5)
    passo_gradi = 30 
    passo_rad = math.radians(passo_gradi)
    angoli_da_testare = np.arange(-math.pi, math.pi + 1e-5, passo_rad)
        
    # Esegui l'estrazione
    points = extract_points_above_threshold(
        angoli_da_testare, 
        grid_min=-6.0, 
        grid_max=6.0, 
        resolution=2500, # Modifica questo valore se vuoi una griglia più fitta
        threshold=0.845  # La tua soglia
    )
    
    # 4. Stampa formattata pronta per il copia-incolla
    print("# --- COPIA DA QUI IN GIÙ ---")
    print("punti_soglia = [")
    for p in points:
        print(f"    {p},")
    print("]")

    # --- BLOCCO DA AGGIUNGERE PER IL PRINT DELLE PREDICTIONS ---
    print("\n" + "="*60)
    print("# --- 1. COPIA DA QUI IN GIÙ PER L'MPC ---")
    print("punti_soglia = [")
    for p in points:
        print(f"    {p},")
    print("]")
    
    print("\n" + "="*60)
    print("# --- 2. VALORI DI PREDICTION DELLA RETE (ORDINATI) ---")
    
    # Ricarichiamo al volo il modello per fare la verifica sui punti estratti
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiLayerPerceptron(input_dim=3, hidden_size=64, hidden_layers=3)
    try:
        model.load_state_dict(torch.load(get_latest_best_model('car'), map_location=device))
        model.to(device)
        model.eval()
        
        for i, p in enumerate(points):
            # Prepariamo il singolo punto come tensor per la rete
            nn_input = torch.tensor([p], dtype=torch.float32).to(device)
            with torch.no_grad():
                output_val = model(nn_input).item()
            
            deg = math.degrees(p[2])
            print(f"Punto {i+1:02d} -> Posizione: [{p[0]:8.4f}, {p[1]:8.4f}] | Angolo: {p[2]:7.4f} rad ({deg:6.1f}°) | Prediction: {output_val:.4f}")
            
    except Exception as e:
        print(f"Impossibile calcolare le predizioni di verifica: {e}")
    print("="*60)