#!/usr/bin/env python
"""
nmpc_car_standalone.py
=======================

Versione STANDALONE (senza ROS) dei due nodi originali:

  - nmpc_car_massimi.py   -> controllore NMPC (Neural-MPC) del robot
  - mock_car_score.py     -> nodo "mock" che genera gli score di
                              data-association (rete neurale) usati
                              per aggiornare le credenze bayesiane

Questo script riproduce **esattamente** la stessa logica di:

  * setup iniziale (posizione di partenza del robot, posizione delle
    auto/alberi, pali, aiuole, punti soglia)
  * modello cinematico e ottimizzazione NMPC (mpc_opt)
  * selezione target/ostacoli (get_target_tree_indices,
    get_nearest_tree_indices, get_closest_threshold_state)
  * data association / scoring con la stessa rete neurale MLP e la
    stessa soglia (p_correct > 0.55 -> 0.85, altrimenti 0.0) e lo
    stesso mapping bayesiano [0,1] -> [0.5,1]
  * salvataggio degli stessi identici file CSV
    (metrics / velocities / plot_data)

...ma senza alcuna dipendenza da ROS/rospy/Gazebo: il robot viene
simulato integrando il modello cinematico usato internamente dal
solutore NMPC (motion "ground truth" = modello perfetto, come è
tipico in una simulazione NMPC pura), e viene fornito un plot
matplotlib in tempo reale, attivabile/disattivabile a piacere.

Uso:
    python nmpc_car_standalone.py                 # con plot live
    python nmpc_car_standalone.py --no-plot        # senza plot (piu' veloce)
    python nmpc_car_standalone.py --max-iter 500
    python nmpc_car_standalone.py --device cpu
"""

import os
import re
import csv
import json
import math
import time
import argparse

import numpy as np
import casadi as ca
import torch

import l4casadi as l4c

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle


# ============================================================================
# Rete neurale surrogata (IDENTICA a quella dei due nodi originali)
# ============================================================================
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=64, hidden_layers=3):
        super().__init__()
        # Se input_dim==3 aggiungiamo un input extra per sin/cos dell'angolo.
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


class NeuralMPCStandaloneSim:
    """
    Fonde in un'unica classe, senza ROS:
      - NeuralMPCHusky   (nmpc_car_massimi.py)  -> controllo/ottimizzazione
      - CarScoresMock    (mock_car_score.py)    -> data association / scoring
    """

    def __init__(self, run_dir=None, map_path=None, models_dir=None,
                 device="auto", show_plot=True, N=None, dt=None, offline_mode=False):

        # ------------------------------------------------------------------
        # Parametri MPC (default identici all'originale, ma configurabili)
        # ------------------------------------------------------------------
        self.hidden_size = 64
        self.hidden_layers = 3
        self.nn_input_dim = 3

        # N=20, dt=0.2 nell'originale erano tarati per un controllo
        # REAL-TIME a 5 Hz su hardware: l'orizzonte T=dt*N=4s permette al
        # robot (v_max=0.2 m/s) di "vedere" al massimo ~0.8m in avanti,
        # che e' molto poco rispetto alle distanze reali dai target (spesso
        # decine di metri). Se lo script viene usato come PLANNER OFFLINE
        # (nessun vincolo di 5Hz), ha senso alzare N e/o abbassare dt per
        # avere un orizzonte piu' lungo e/o una discretizzazione piu' fine.
        self.N = N if N is not None else 20
        self.dt = dt if dt is not None else 0.2  # Controllo a 5 Hz (default)
        self.T = self.dt * self.N
        self.offline_mode = offline_mode

        # Specifiche del robot reale (Uniciclo)
        self.nx = 3          # Stato: [x, y, theta]
        self.n_state = 3
        self.n_control = 2   # Ingressi di controllo: [v, omega]

        self.NUM_TARGET_TREES = 1    # subset target vicini da esplorare
        self.NUM_OBSTACLE_TREES = 3  # subset ostacoli vicini da evitare

        self.threshold_entropy = 0.15  # Quando target considerato "visto"

        # ------------------------------------------------------------------
        # Punti soglia (massimi locali) - IDENTICI all'originale
        # ------------------------------------------------------------------
        self.punti_soglia = [
            [-1.3037, -2.5762, -2.0944],  # 1
            [-0.934, -2.3745, -1.5708],   # 2
            [3.2101, 0.1273, -0.0],       # 3
            [-1.1693, 3.066, 1.5708],     # 5
        ]

        # ------------------------------------------------------------------
        # COORDINATE HARDCODED DELLE AUTO/ALBERI (IDENTICO all'originale)
        # ------------------------------------------------------------------
        default_map_path = (
            "/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/"
            "ros/src/nmpc_ros/niccolo/map_results/car_map_final.json"
        )
        file_path = map_path if map_path is not None else default_map_path
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                cars = json.load(f).get("cars", [])
            cars_sorted = sorted(cars, key=lambda c: c["id"])
            extracted = []
            for c in cars_sorted:
                x, y, rad = c["x"], c["y"], c["orientation_rad"]
                extracted.append([x, y, rad])
            self.trees_pos = np.array(extracted, dtype=np.float32)
        else:
            # Fallback hardcoded se il file non esiste (identico all'originale)
            self.trees_pos = np.array(
                [
                    [43.04, -2.149, 1.4337],
                    [49.701, -7.093, -1.6995],
                    [52.743, -7.233, -1.7331],
                ],
                dtype=np.float32,
            )
        print("[Setup] trees_pos:\n", self.trees_pos)

        # ------------------------------------------------------------------
        # COORDINATE PALI (x, y) - IDENTICO all'originale
        # ------------------------------------------------------------------
        self.poles_pos = np.array([
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
        self.NUM_POLES = self.poles_pos.shape[0]
        # Fila di pali ordinata per x: definisce la linea spezzata che
        # separa il parcheggio in due corridoi (i pali vanno da x~7.2 a
        # x~58.1 con y decrescente -> spartitraffico centrale). Usata dalle
        # euristiche di selezione del target per capire se un'auto e'
        # raggiungibile in linea retta o se bisogna aggirare la barriera.
        self._poles_sorted = self.poles_pos[np.argsort(self.poles_pos[:, 0])]

        # ------------------------------------------------------------------
        # COORDINATE AIUOLE (x, y, lunghezza, larghezza, orientamento_rad)
        # IDENTICO all'originale
        # ------------------------------------------------------------------
        self.flowerbeds_pos = np.array([
            [2.1879, -5.5748, 4.6071, 4.5434, -0.175],    # aiuola 1
            [64.6609, -16.5047, 2.8938, 4.5661, -0.175],  # aiuola 2
            [32.1876, -14.3858, 66.7007, 3.5897, -0.175],  # aiuola 3
            [3.8231, 4.0245, 1.6749, 4.7252, -0.175],     # aiuola 4
            [5.7205, 7.9094, 4.5549, 4.5383, -0.175],     # aiuola 5
            [61.5571, -4.8938, 2.4004, 9.0567, -0.175],   # aiuola 6
            [9.7648, 16.3899, 2.0120, 4.6903, -0.175],    # aiuola 7
            [58.9274, 7.9500, 2.4176, 4.3564, -0.175],    # aiuola 8
            [34.1108, 15.1653, 50.4020, 1.5882, -0.175],  # aiuola 9
        ], dtype=np.float32)
        self.NUM_FLOWERBEDS = self.flowerbeds_pos.shape[0]

        # Identificativi reali stabili degli alberi/auto (0: raw, 1: ripe)
        self.trees_gt_id = np.array([0, 1], dtype=np.uint8)

        self.num_total_trees = self.trees_pos.shape[0]

        # ------------------------------------------------------------------
        # Raggiungibilita' statica di ciascuna auto: un'auto e' "esplorabile"
        # solo se almeno uno dei punti soglia (self.punti_soglia) e'
        # geometricamente ammissibile, cioe' non in collisione con altre
        # auto, pali o aiuole. Questo dipende solo dalla geometria statica
        # della scena, quindi va calcolato una volta sola qui (stessi
        # margini di sicurezza usati in get_closest_threshold_state).
        # ------------------------------------------------------------------
        self.trees_admissible_points = [self._compute_tree_admissible_points(t) for t in self.trees_pos]
        self.trees_reachable = np.array([len(pts) > 0 for pts in self.trees_admissible_points], dtype=bool)
        unreachable_idx = np.where(~self.trees_reachable)[0]
        if unreachable_idx.size > 0:
            print(f"[Setup] ATTENZIONE: le auto agli indici {unreachable_idx.tolist()} non hanno "
                  f"alcun punto soglia ammissibile (bloccate da altre auto/pali/aiuole): "
                  f"NON verranno proposte come target.")
        self.entropy_entire_field = self.entropy_f(self.num_total_trees)
        self.beliefs_k = ca.DM.ones(self.num_total_trees, 2) * 0.5

        # ------------------------------------------------------------------
        # Posizione iniziale del robot (IDENTICA a mock_car_score.py)
        # ------------------------------------------------------------------
        self.start_pos = [6.5, 0.5, 0.0]
        self.current_state = list(self.start_pos)
        self._compute_visit_order() 

        # ------------------------------------------------------------------
        # Selezione device (adattamento per portabilita': l'originale
        # nmpc_car_massimi.py forzava 'cuda', mock_car_score.py usava
        # 'cuda' se disponibile altrimenti 'cpu'. Qui riproduciamo il
        # comportamento del nodo mock, cosi' lo script gira anche senza GPU)
        # ------------------------------------------------------------------
        if device == "auto":
            self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.torch_device = torch.device(device)
        self.l4c_device_str = "cuda" if self.torch_device.type == "cuda" else "cpu"
        print(f"[Setup] Device selezionato: {self.torch_device}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir_override = models_dir

        # ------------------------------------------------------------------
        # Reti neurali L4CasADi (usate DENTRO l'ottimizzazione NMPC),
        # stessa identica logica di caricamento di nmpc_car_massimi.py
        # ------------------------------------------------------------------
        self.l4c_nn = []
        for label in ['car', 'car']:
            model = MultiLayerPerceptron(input_dim=self.nn_input_dim,
                                          hidden_size=self.hidden_size,
                                          hidden_layers=self.hidden_layers)
            model_load_path = self.get_latest_best_model(script_dir, label)
            model.load_state_dict(torch.load(model_load_path, map_location=self.torch_device))
            model.eval()
            g_nn = l4c.L4CasADi(model, batched=True, device=self.l4c_device_str, name=label)
            self.l4c_nn.append(g_nn)

        # ------------------------------------------------------------------
        # Rete neurale "plain" torch usata per il DATA ASSOCIATION / SCORING
        # (stessa identica logica di mock_car_score.py: singolo modello
        # 'car' usato per valutare la probabilita' di corretta associazione)
        # ------------------------------------------------------------------
        score_model = MultiLayerPerceptron(input_dim=3, hidden_size=64, hidden_layers=3)
        score_model_path = self.get_latest_best_model(script_dir, 'car')
        score_model.load_state_dict(torch.load(score_model_path, map_location=self.torch_device))
        score_model.to(self.torch_device)
        score_model.eval()
        self.score_model = score_model

        self.num_cars = len(self.trees_pos)  # == self.num_total_trees

        self.entropy_target = self.entropy_f(self.NUM_TARGET_TREES)
        self.current_state = list(self.start_pos)

        self.baselines_dir = (
            os.path.join(script_dir, "../../baselines") if run_dir is None else run_dir
        )

        self.show_plot = show_plot
        self._plot_initialized = False

    # ========================================================================
    # Caricamento modelli (unifica get_latest_best_model dei due nodi
    # originali, con lo stesso fallback usato in mock_car_score.py)
    # ========================================================================
    def get_latest_best_model(self, script_dir, cls=''):
        if self.models_dir_override is not None:
            model_dir = os.path.join(self.models_dir_override, cls)
        else:
            model_dir = os.path.join(script_dir, "models", cls)
            if not os.path.exists(model_dir):
                model_dir = os.path.join(script_dir, "..", "models", cls)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Cartella modelli non trovata: {model_dir}. "
                f"Usa --models-dir per indicare la cartella corretta "
                f"(deve contenere modelli 'best_model_epoch_N.pth' in una "
                f"sottocartella '{cls}')."
            )

        model_files = [f for f in os.listdir(model_dir) if re.match(r"best_model_epoch_(\d+)\.pth", f)]
        if not model_files:
            raise FileNotFoundError(f"Nessun file di modello pesi (.pth) trovato in {model_dir}")
        latest_model = max(model_files, key=lambda x: int(re.match(r"best_model_epoch_(\d+)\.pth", x).group(1)))
        return os.path.join(model_dir, latest_model)

    # ========================================================================
    # Utility statiche (IDENTICHE a nmpc_car_massimi.py)
    # ========================================================================
    @staticmethod
    def get_domain(tree_positions):
        return [np.min(tree_positions[:, 0]), np.min(tree_positions[:, 1])], \
               [np.max(tree_positions[:, 0]), np.max(tree_positions[:, 1])]

    @staticmethod
    def kin_model(dt):
        """ Modello cinematico differenziale dell'uniciclo integrato con RK4 """
        x_sym = ca.SX.sym('x', 3)  # [x, y, theta]
        u_sym = ca.SX.sym('u', 2)  # [v, omega]

        theta = x_sym[2]
        v = u_sym[0]
        omega = u_sym[1]

        x_dot = ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
            omega
        )

        f_continuous = ca.Function('f_cont', [x_sym, u_sym], [x_dot], ['x', 'u'], ['x_dot'])
        k1 = f_continuous(x_sym, u_sym)
        k2 = f_continuous(x_sym + dt / 2 * k1, u_sym)
        k3 = f_continuous(x_sym + dt / 2 * k2, u_sym)
        k4 = f_continuous(x_sym + dt * k3, u_sym)
        x_next = x_sym + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return ca.Function('F', [x_sym, u_sym], [x_next], ['x_k', 'u_k'], ['x_k1'])

    @staticmethod
    def bayes(prior, likelihood):
        unnorm = prior * likelihood
        norm = ca.repmat(ca.sum2(unnorm), 1, 2)
        return unnorm / norm

    @staticmethod
    def entropy_f(num_targets):
        p = ca.MX.sym(f'input_entropy_f{num_targets}_dim', num_targets, 2)
        eps = 1e-6
        p_clipped = ca.fmax(eps, ca.fmin(1 - eps, p))
        entropy_per_target = -ca.sum2(p_clipped * (ca.log(p_clipped) / ca.log(2)))
        return ca.Function(f'entropy_f_{num_targets}_dim', [p], [entropy_per_target])

    # ========================================================================
    # Selezione target/ostacoli
    # ========================================================================
    def _compute_tree_admissible_points(self, tree):
        """
        Calcola, per una singola auto (tree = [tx, ty, theta_target]), quali
        punti soglia (self.punti_soglia) sono geometricamente ammissibili,
        cioe' non in collisione con altre auto, pali o aiuole. Usa
        esattamente gli stessi margini di sicurezza di
        get_closest_threshold_state (SAFETY_MARGIN_CAR/POLE, MARGIN_FLOWERBED).
        Ritorna una lista di dict {'x', 'y', 'azimuth'} per ogni punto
        soglia ammissibile (lista vuota se l'auto non e' raggiungibile).
        """
        tx, ty, theta_target = tree[0], tree[1], tree[2]
        SAFETY_MARGIN_CAR = 1.5
        SAFETY_MARGIN_POLE = 1
        MARGIN_FLOWERBED = 1.5
        admissible = []
        for p in self.punti_soglia:
            p_x_rel, p_y_rel, p_azimuth = p[0], p[1], p[2]
            p_x_glob = tx + (p_x_rel * math.cos(theta_target)) - (p_y_rel * math.sin(theta_target))
            p_y_glob = ty + (p_x_rel * math.sin(theta_target)) + (p_y_rel * math.cos(theta_target))
            ok = True
            for obs in self.trees_pos:
                if math.hypot(obs[0] - tx, obs[1] - ty) < 0.1:
                    continue  # ignora l'auto stessa
                if self._point_collides_with_car(p_x_glob, p_y_glob, obs[0], obs[1], obs[2]):
                    ok = False
                    break
            if ok:
                for pole in self.poles_pos:
                    if math.hypot(p_x_glob - pole[0], p_y_glob - pole[1]) < SAFETY_MARGIN_POLE:
                        ok = False
                        break
            if ok:
                for f in self.flowerbeds_pos:
                    fx, fy, flen, fwid, ftheta = f[0], f[1], f[2], f[3], f[4]
                    dx_f = p_x_glob - fx
                    dy_f = p_y_glob - fy
                    x_rel_f = dx_f * math.cos(ftheta) + dy_f * math.sin(ftheta)
                    y_rel_f = -dx_f * math.sin(ftheta) + dy_f * math.cos(ftheta)
                    sigma_x_f = (flen / 2.0) + MARGIN_FLOWERBED
                    sigma_y_f = (fwid / 2.0) + MARGIN_FLOWERBED
                    dist_norm_f = (x_rel_f / sigma_x_f) ** 4 + (y_rel_f / sigma_y_f) ** 4
                    if dist_norm_f <= 1.0:
                        ok = False
                        break
            if ok:
                admissible.append({'x': p_x_glob, 'y': p_y_glob, 'azimuth': p_azimuth})
        return admissible

    def _pole_line_side(self, xy):
        """
        Ritorna >0 / <0 a seconda del lato della fila di pali su cui si
        trova il punto xy=(x,y) (0 se esattamente sulla linea). La fila di
        pali e' trattata come una linea spezzata (interpolazione lineare
        tra pali consecutivi ordinati per x), che nello scenario reale fa
        da spartitraffico centrale del parcheggio.
        """
        x, y = xy[0], xy[1]
        x_clamped = np.clip(x, self._poles_sorted[0, 0], self._poles_sorted[-1, 0])
        y_line = np.interp(x_clamped, self._poles_sorted[:, 0], self._poles_sorted[:, 1])
        return np.sign(y - y_line)

    def _barrier_aware_distance(self, robot_xy, target_xy):
        """
        Distanza tra robot e target che tiene conto della fila di pali
        centrale: se sono sullo stesso lato la distanza e' quella euclidea
        diretta; se sono su lati opposti il robot non puo' tagliare
        attraverso i pali, quindi la distanza viene approssimata come il
        percorso piu' breve passando da una delle due estremita' della
        fila (unico modo reale per passare da un corridoio all'altro).
        """
        robot_xy = np.asarray(robot_xy, dtype=np.float64)
        target_xy = np.asarray(target_xy, dtype=np.float64)
        direct_dist = np.linalg.norm(robot_xy - target_xy)

        robot_side = self._pole_line_side(robot_xy)
        target_side = self._pole_line_side(target_xy)
        if robot_side == 0 or target_side == 0 or robot_side == target_side:
            return direct_dist

        endpoints = np.array([self._poles_sorted[0, :2], self._poles_sorted[-1, :2]], dtype=np.float64)
        via_dist = min(
            np.linalg.norm(robot_xy - ep) + np.linalg.norm(ep - target_xy)
            for ep in endpoints
        )
        return via_dist

    def get_current_target_index(self):
        """
        Ritorna l'indice dell'auto da raggiungere ora, seguendo l'ordine
        fisso precalcolato in self.visit_order. Salta automaticamente le
        auto la cui entropia e' gia' scesa sotto soglia (es. osservate "di
        passaggio" mentre il robot si dirigeva verso un'altra), avanzando
        self.visit_pointer. Ritorna None se la route e' stata completata.
        """
        H = self.entropy_entire_field(self.beliefs_k).full().flatten()

        while self.visit_pointer < len(self.visit_order):
            idx = self.visit_order[self.visit_pointer]
            if H[idx] <= self.threshold_entropy:
                print(f"[Info] Auto {idx} gia' esplorata, passo al prossimo target della route.")
                self.visit_pointer += 1
                continue
            return idx

        return None

    def get_nearest_tree_indices(self, robot_position, num_obstacle=None):
        distances = np.linalg.norm(self.trees_pos[:, :2] - robot_position, axis=1)
        return np.argsort(distances)[:self.NUM_OBSTACLE_TREES]
    
    def _point_collides_with_car(self, px, py, car_x, car_y, car_theta,
                              car_length=4.0, car_width=2.0,
                              margin_x=1.2, margin_y=1.2):
        """
        Test di collisione punto-auto IDENTICO (stessa ellisse, stesso centro
        corretto, stessi margini) a quello usato in mpc_opt per l'evitamento
        ostacoli. A differenza del vecchio check circolare centrato sul punto
        frontale (x,y) dell'auto, qui il centro geometrico reale dell'auto
        e' arretrato di car_length/2 lungo -theta, cosi' il corpo posteriore
        dell'auto (che si estende fino a ~4m dietro il riferimento frontale)
        viene correttamente considerato ingombro.
        Ritorna True se il punto E' in collisione (troppo vicino).
        """
        center_x = car_x - (car_length / 2.0) * math.cos(car_theta)
        center_y = car_y - (car_length / 2.0) * math.sin(car_theta)
        dx = px - center_x
        dy = py - center_y
        x_rel = dx * math.cos(car_theta) + dy * math.sin(car_theta)
        y_rel = -dx * math.sin(car_theta) + dy * math.cos(car_theta)
        sigma_x = (car_length / 2.0) + margin_x
        sigma_y = (car_width / 2.0) + margin_y
        dist_norm = (x_rel / sigma_x) ** 2 + (y_rel / sigma_y) ** 2
        return dist_norm < 1.0
    
    def _tour_cost(self, order, from_xy):
        """
        Costo totale di un tour (sequenza di indici di auto), partendo da
        from_xy. Usa la distanza barrier-aware, coerente con quella usata
        dal resto della pipeline (get_target_tree_indices, etc.), cosi' il
        costo del percorso riflette davvero cio' che il robot dovra' fare.
        """
        total = 0.0
        cur = np.asarray(from_xy, dtype=np.float64)
        for idx in order:
            nxt = self.trees_pos[idx, :2]
            total += self._barrier_aware_distance(cur, nxt)
            cur = nxt
        return total

    def _two_opt_improve(self, order, from_xy, max_passes=100):
        """
        Raffina un tour iniziale (es. nearest-neighbor) con la classica
        euristica 2-opt: prova a invertire ogni sotto-segmento del tour e
        mantiene l'inversione solo se riduce il costo totale. Necessario
        perche' il nearest-neighbor puro e' greedy e produce spesso percorsi
        con incroci/zig-zag evidenti (un'auto "saltata" e ripresa dopo aver
        gia' superato la sua posizione). Con poche decine di auto per lato
        la complessita' O(n^2) per passata e' trascurabile.
        """
        order = list(order)
        n = len(order)
        if n < 4:
            return order  # nulla da ottimizzare con meno di 4 tappe

        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            best_cost = self._tour_cost(order, from_xy)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    cand_cost = self._tour_cost(candidate, from_xy)
                    if cand_cost < best_cost - 1e-9:
                        order = candidate
                        best_cost = cand_cost
                        improved = True
        return order

    def _or_opt_improve(self, order, from_xy, max_passes=50):
        """
        Passata aggiuntiva Or-opt: prova a spostare (senza invertire) singole
        auto o coppie consecutive in un'altra posizione del tour. Cattura
        miglioramenti che il 2-opt da solo non trova (es. un'auto isolata
        vicino al percorso di ritorno che conviene visitare "al volo" invece
        che in fondo alla coda).
        """
        order = list(order)
        n = len(order)
        if n < 3:
            return order

        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for seg_len in (1, 2):
                i = 0
                while i + seg_len <= n:
                    segment = order[i:i + seg_len]
                    rest = order[:i] + order[i + seg_len:]
                    base_cost = self._tour_cost(order, from_xy)
                    best_cost = base_cost
                    best_candidate = None
                    for k in range(len(rest) + 1):
                        candidate = rest[:k] + segment + rest[k:]
                        if candidate == order:
                            continue
                        c = self._tour_cost(candidate, from_xy)
                        if c < best_cost - 1e-9:
                            best_cost = c
                            best_candidate = candidate
                    if best_candidate is not None:
                        order = best_candidate
                        n = len(order)
                        improved = True
                    i += 1
        return order
    
    def _compute_visit_order(self):
        """
        Precalcola la sequenza fissa (route) di auto da visitare.

        Pipeline per ciascun lato della fila di pali:
        1) Costruzione iniziale con nearest-neighbor greedy.
        2) Raffinamento 2-opt (elimina incroci/zig-zag).
        3) Raffinamento Or-opt (riposiziona singole tappe "fuori posto").

        Il lato di partenza del robot viene sempre visitato per intero prima
        di attraversare la fila verso l'altro lato (un solo attraversamento,
        nel punto in cui e' inevitabile).
        """
        reachable_indices = np.where(self.trees_reachable)[0]
        if reachable_indices.size == 0:
            self.visit_order = []
            self.visit_pointer = 0
            return

        start_xy = np.array(self.start_pos[:2], dtype=np.float64)
        robot_side = self._pole_line_side(start_xy)

        side_same, side_other = [], []
        for i in reachable_indices:
            side = self._pole_line_side(self.trees_pos[i, :2])
            if side == 0 or robot_side == 0 or side == robot_side:
                side_same.append(int(i))
            else:
                side_other.append(int(i))

        def _nearest_neighbor_order(indices, from_xy):
            remaining = list(indices)
            order = []
            cur_xy = np.asarray(from_xy, dtype=np.float64)
            while remaining:
                dists = [self._barrier_aware_distance(cur_xy, self.trees_pos[j, :2]) for j in remaining]
                nearest_pos = int(np.argmin(dists))
                nearest_idx = remaining.pop(nearest_pos)
                order.append(nearest_idx)
                cur_xy = self.trees_pos[nearest_idx, :2]
            return order

        def _build_refined_tour(indices, from_xy):
            if not indices:
                return []
            order = _nearest_neighbor_order(indices, from_xy)
            order = self._two_opt_improve(order, from_xy)
            order = self._or_opt_improve(order, from_xy)
            return order

        order_same = _build_refined_tour(side_same, start_xy)
        last_xy = self.trees_pos[order_same[-1], :2] if order_same else start_xy
        order_other = _build_refined_tour(side_other, last_xy)

        self.visit_order = order_same + order_other
        self.visit_pointer = 0

        cost_same = self._tour_cost(order_same, start_xy)
        cost_other = self._tour_cost(order_other, last_xy)
        print(f"[Setup] Ordine di visita precalcolato ({len(self.visit_order)} auto): {self.visit_order}")
        print(f"[Setup] Costo stimato lato robot: {cost_same:.2f} m | lato opposto: {cost_other:.2f} m")

        order_same = _nearest_neighbor_order(side_same, start_xy)
        last_xy = self.trees_pos[order_same[-1], :2] if order_same else start_xy
        order_other = _nearest_neighbor_order(side_other, last_xy)

        self.visit_order = order_same + order_other
        self.visit_pointer = 0
        print(f"[Setup] Ordine di visita precalcolato ({len(self.visit_order)} auto): {self.visit_order}")

    def select_route_interactively(self):
        """
        Apre una finestra bloccante in cui l'utente clicca sulle auto
        raggiungibili nell'ordine in cui vuole che il robot le visiti.

        - Click sinistro su un'auto  -> la aggiunge in coda alla route
            (mostrata in verde con il numero d'ordine); ri-cliccarla la
            rimuove (toggle).
        - Tasto 'u'                  -> undo (rimuove l'ultima aggiunta).
        - Tasto 'r'                  -> reset completo della selezione.
        - Pulsante "Conferma"        -> chiude la finestra e blocca la route.

        Le auto NON raggiungibili (self.trees_reachable == False) sono
        mostrate con una X nera e non sono selezionabili.

        Le auto raggiungibili che l'utente NON ha selezionato esplicitamente
        vengono aggiunte automaticamente in coda (nearest-neighbor a partire
        dall'ultimo punto scelto), cosi' nessuna auto esplorabile va persa
        per dimenticanza.
        """
        from matplotlib.widgets import Button

        selected = []  # indici, nell'ordine di click dell'utente

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_aspect('equal')
        ax.set_title("Clicca le auto nell'ordine desiderato, poi premi 'Conferma'\n"
                    "(tasto 'u' = annulla ultima, 'r' = reset)")
        ax.grid(True, alpha=0.3)

        if self.NUM_POLES > 0:
            ax.scatter(self.poles_pos[:, 0], self.poles_pos[:, 1], c='gray', marker='|', s=100)
            ax.plot(self._poles_sorted[:, 0], self._poles_sorted[:, 1],
                    c='gray', linestyle=':', alpha=0.7, label='barriera pali')

        for fb in self.flowerbeds_pos:
            fx, fy, flen, fwid, ftheta = fb
            corners = np.array([[-flen/2, -fwid/2], [flen/2, -fwid/2],
                                [flen/2, fwid/2], [-flen/2, fwid/2]])
            c, s = math.cos(ftheta), math.sin(ftheta)
            R = np.array([[c, -s], [s, c]])
            rotated = corners @ R.T + np.array([fx, fy])
            ax.add_patch(MplPolygon(rotated, closed=True, facecolor='green', alpha=0.2, edgecolor='green'))

        ax.plot(self.start_pos[0], self.start_pos[1], marker='s', markersize=10,
                color='blue', label='partenza robot')

        car_dots = {}
        for i, (tx, ty, _) in enumerate(self.trees_pos):
            if self.trees_reachable[i]:
                dot, = ax.plot(tx, ty, 'o', markersize=14, color='lightgray',
                                markeredgecolor='black', picker=8)
            else:
                dot, = ax.plot(tx, ty, 'X', markersize=14, color='black',
                                markeredgecolor='white', picker=False)
            car_dots[i] = dot

        labels = {}
        path_line, = ax.plot([], [], 'b--', linewidth=1.5, alpha=0.7)

        lb, ub = self.get_domain(self.trees_pos)
        margin = 15
        ax.set_xlim(lb[0] - margin, ub[0] + margin)
        ax.set_ylim(lb[1] - margin, ub[1] + margin)

        status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        def _refresh():
            for i, dot in car_dots.items():
                if not self.trees_reachable[i]:
                    continue
                dot.set_color('limegreen' if i in selected else 'lightgray')
            for txt in labels.values():
                txt.remove()
            labels.clear()
            for order_pos, idx in enumerate(selected, start=1):
                tx, ty = self.trees_pos[idx, 0], self.trees_pos[idx, 1]
                labels[idx] = ax.annotate(str(order_pos), xy=(tx, ty), xytext=(tx + 0.6, ty + 0.6),
                                        fontsize=11, fontweight='bold', color='darkgreen')
            if selected:
                pts = np.array([self.start_pos[:2]] + [self.trees_pos[i, :2].tolist() for i in selected])
                path_line.set_data(pts[:, 0], pts[:, 1])
            else:
                path_line.set_data([], [])
            n_reachable = int(np.sum(self.trees_reachable))
            status_text.set_text(f"Selezionate: {len(selected)}/{n_reachable} auto raggiungibili")
            fig.canvas.draw_idle()

        def _on_pick(event):
            for idx, dot in car_dots.items():
                if event.artist is dot and self.trees_reachable[idx]:
                    if idx not in selected:
                        selected.append(idx)
                    else:
                        selected.remove(idx)
                    _refresh()
                    break

        def _on_key(event):
            if event.key == 'u' and selected:
                selected.pop()
                _refresh()
            elif event.key == 'r':
                selected.clear()
                _refresh()

        fig.canvas.mpl_connect('pick_event', _on_pick)
        fig.canvas.mpl_connect('key_press_event', _on_key)

        ax_confirm = fig.add_axes([0.81, 0.02, 0.13, 0.06])
        btn_confirm = Button(ax_confirm, 'Conferma')
        btn_confirm.on_clicked(lambda event: plt.close(fig))

        _refresh()
        plt.show()  # bloccante: il codice riprende solo dopo "Conferma" o chiusura finestra

        # Completa automaticamente le auto raggiungibili non selezionate manualmente
        reachable_indices = np.where(self.trees_reachable)[0].tolist()
        missing = [i for i in reachable_indices if i not in selected]
        if missing:
            last_xy = self.trees_pos[selected[-1], :2] if selected else np.array(self.start_pos[:2])
            remaining = list(missing)
            cur_xy = last_xy
            appended = []
            while remaining:
                dists = [self._barrier_aware_distance(cur_xy, self.trees_pos[j, :2]) for j in remaining]
                nearest_pos = int(np.argmin(dists))
                nearest_idx = remaining.pop(nearest_pos)
                appended.append(nearest_idx)
                cur_xy = self.trees_pos[nearest_idx, :2]
            print(f"[Info] Auto non selezionate manualmente, aggiunte automaticamente in coda: {appended}")
            selected = selected + appended

        self.visit_order = selected
        self.visit_pointer = 0
        print(f"[Setup] Route confermata dall'utente: {self.visit_order}")

    def get_closest_threshold_state(self, robot_state, target_trees, preferred_side=None):
        """
        Trova la configurazione relativa ottima [x_rel, y_rel, azimuth] dai
        punti soglia. SCARTA COMPLETAMENTE i punti in zone non ammissibili
        (auto, pali, aiuole).

        preferred_side: se specificato (output di _pole_line_side sulla
        posizione del robot), tra tutti i punti soglia ammissibili si
        preferiscono SEMPRE quelli che cadono, in coordinate globali, sullo
        stesso lato della fila di pali del robot. Si accetta un punto
        sull'altro lato (cioe' si "taglia" la barriera) solo se non esiste
        NESSUN punto ammissibile sul lato corrente per nessuna delle auto
        candidate.
        """
        rx, ry, rtheta = robot_state[0], robot_state[1], robot_state[2]
        W_theta = 0.5
        SAFETY_MARGIN_POLE = 1
        MARGIN_FLOWERBED = 1.5

        fallback_cost = float('inf')
        fallback_peak = [0.0, 0.0, 0.0]

        best_cost_same_side = float('inf')
        best_peak_same_side = None
        best_cost_any_side = float('inf')
        best_peak_any_side = None

        for tree in target_trees:
            tx, ty, theta_target = tree[0], tree[1], tree[2]
            dX = rx - tx
            dY = ry - ty
            curr_x_rel = dX * math.cos(theta_target) + dY * math.sin(theta_target)
            curr_y_rel = -dX * math.sin(theta_target) + dY * math.cos(theta_target)
            curr_theta_y = rtheta + (math.pi / 2.0)
            curr_azimuth_raw = curr_theta_y - theta_target + math.pi
            curr_azimuth = math.atan2(math.sin(curr_azimuth_raw), math.cos(curr_azimuth_raw))

            for p in self.punti_soglia:
                p_x_rel, p_y_rel, p_azimuth = p[0], p[1], p[2]
                p_x_glob = tx + (p_x_rel * math.cos(theta_target)) - (p_y_rel * math.sin(theta_target))
                p_y_glob = ty + (p_x_rel * math.sin(theta_target)) + (p_y_rel * math.cos(theta_target))

                punto_ammissibile = True
                for obs in self.trees_pos:
                    if math.hypot(obs[0] - tx, obs[1] - ty) < 0.1:
                        continue  # ignora l'auto stessa
                    if self._point_collides_with_car(p_x_glob, p_y_glob, obs[0], obs[1], obs[2]):
                        punto_ammissibile = False
                        break
                if punto_ammissibile and hasattr(self, 'poles_pos'):
                    for pole in self.poles_pos:
                        if math.hypot(p_x_glob - pole[0], p_y_glob - pole[1]) < SAFETY_MARGIN_POLE:
                            punto_ammissibile = False
                            break
                if punto_ammissibile and hasattr(self, 'flowerbeds_pos'):
                    for f in self.flowerbeds_pos:
                        fx, fy, flen, fwid, ftheta = f[0], f[1], f[2], f[3], f[4]
                        dx_f = p_x_glob - fx
                        dy_f = p_y_glob - fy
                        x_rel_f = dx_f * math.cos(ftheta) + dy_f * math.sin(ftheta)
                        y_rel_f = -dx_f * math.sin(ftheta) + dy_f * math.cos(ftheta)
                        sigma_x_f = (flen / 2.0) + MARGIN_FLOWERBED
                        sigma_y_f = (fwid / 2.0) + MARGIN_FLOWERBED
                        dist_norm_f = (x_rel_f / sigma_x_f) ** 4 + (y_rel_f / sigma_y_f) ** 4
                        if dist_norm_f <= 1.0:
                            punto_ammissibile = False
                            break

                dist_geometrica = math.hypot(curr_x_rel - p_x_rel, curr_y_rel - p_y_rel)
                delta_azimuth = p_azimuth - curr_azimuth
                delta_azimuth_norm = math.atan2(math.sin(delta_azimuth), math.cos(delta_azimuth))
                sforzo_rotazione = abs(delta_azimuth_norm)
                costo_totale = dist_geometrica + (W_theta * sforzo_rotazione)

                if costo_totale < fallback_cost:
                    fallback_cost = costo_totale
                    fallback_peak = [p_x_rel, p_y_rel, p_azimuth]

                if not punto_ammissibile:
                    continue

                if costo_totale < best_cost_any_side:
                    best_cost_any_side = costo_totale
                    best_peak_any_side = [p_x_rel, p_y_rel, p_azimuth]

                point_side = self._pole_line_side((p_x_glob, p_y_glob))
                same_side = (preferred_side is None) or (point_side == 0) or (point_side == preferred_side)
                if same_side and costo_totale < best_cost_same_side:
                    best_cost_same_side = costo_totale
                    best_peak_same_side = [p_x_rel, p_y_rel, p_azimuth]

        if best_peak_same_side is not None:
            return best_peak_same_side

        if best_peak_any_side is not None:
            print("[WARN][get_closest_threshold_state] Nessun punto ammissibile sul lato corrente: "
                "si accetta un punto dall'altro lato della fila di pali.")
            return best_peak_any_side

        print("[WARN][get_closest_threshold_state] Tutti i punti ottimi sono occupati! Uso fallback.")
        return fallback_peak

    # ========================================================================
    # Data association / scoring (IDENTICA a mock_car_score.py)
    # ========================================================================
    def compute_tree_scores(self, robot_state):
        """
        Riproduce esattamente la logica dello score-callback del nodo mock:
        per ogni auto/albero, calcola l'input relativo in terna MACCHINA
        (dX, dY, azimuth), lo passa alla rete neurale e applica la stessa
        soglia p_correct > 0.55 -> 0.85 altrimenti 0.0. Ritorna direttamente
        gli score gia' mappati in [0.5, 1] come fa tree_scores_callback in
        nmpc_car_massimi.py.
        """
        X_r, Y_r, theta_r = robot_state[0], robot_state[1], robot_state[2]

        raw_scores = np.zeros((self.num_cars, 2), dtype=np.float32)
        with torch.no_grad():
            for i in range(self.num_cars):
                X_t = self.trees_pos[i, 0]
                Y_t = self.trees_pos[i, 1]
                theta_t = self.trees_pos[i, 2]

                # Vettore globale DALLA macchina AL robot
                dX = X_r - X_t
                dY = Y_r - Y_t
                # Trasformazione in terna MACCHINA
                x_rel = dX * math.cos(theta_t) + dY * math.sin(theta_t)
                y_rel = -dX * math.sin(theta_t) + dY * math.cos(theta_t)
                # Azimut
                theta_y_robot = theta_r + (math.pi / 2.0)
                azimuth_raw = theta_y_robot - theta_t + math.pi
                azimuth_norm = math.atan2(math.sin(azimuth_raw), math.cos(azimuth_raw))

                nn_input = torch.tensor([x_rel, y_rel, azimuth_norm],
                                         dtype=torch.float32).unsqueeze(0).to(self.torch_device)
                logit = self.score_model(nn_input)
                p_correct = logit.item()

                if p_correct > 0.5:
                    raw_scores[i, 0] = 0.85
                else:
                    raw_scores[i, 0] = 0.0
                # raw_scores[i, 1] resta 0.0, come nel nodo mock originale

        # Mapping identico a tree_scores_callback: [0,1] -> [0.5, 1]
        scores = np.zeros_like(raw_scores)
        scores[:, 0] = 0.5 + 0.5 * raw_scores[:, 0]
        scores[:, 1] = 1.0 - scores[:, 0]
        return scores

    # ========================================================================
    # Ottimizzazione NMPC (IDENTICA a nmpc_car_massimi.py)
    # ========================================================================
    def mpc_opt(self, target_trees, target_lambdas, obstacle_trees, closest_thresh, lb, ub, x0, steps=10):
        opti = ca.Opti()
        F_ = self.kin_model(self.dt)

        X = opti.variable(self.n_state, steps + 1)
        U = opti.variable(self.n_control, steps)

        param_size = self.n_state + self.NUM_TARGET_TREES * 5 + self.NUM_OBSTACLE_TREES * 3 + 3 + \
            (self.NUM_POLES * 2) + (self.NUM_FLOWERBEDS * 5)
        P0 = opti.parameter(param_size)

        p_idx = 0
        X0 = P0[p_idx: p_idx + self.n_state]; p_idx += self.n_state
        TARGET_TREES_param = P0[p_idx: p_idx + self.NUM_TARGET_TREES * 3].reshape((self.NUM_TARGET_TREES, 3)).T; p_idx += self.NUM_TARGET_TREES * 3
        L0 = P0[p_idx: p_idx + self.NUM_TARGET_TREES * 2].reshape((self.NUM_TARGET_TREES, 2)); p_idx += self.NUM_TARGET_TREES * 2
        OBSTACLE_TREES_param = P0[p_idx: p_idx + self.NUM_OBSTACLE_TREES * 3].reshape((self.NUM_OBSTACLE_TREES, 3)).T
        p_idx += self.NUM_OBSTACLE_TREES * 3
        OPT_THRESH_param = P0[p_idx: p_idx + 3]
        p_idx += 3
        POLES_param = P0[p_idx: p_idx + self.NUM_POLES * 2].reshape((self.NUM_POLES, 2)).T
        p_idx += self.NUM_POLES * 2
        FLOWERBEDS_param = P0[p_idx: p_idx + self.NUM_FLOWERBEDS * 5].reshape((self.NUM_FLOWERBEDS, 5)).T
        p_idx += self.NUM_FLOWERBEDS * 5

        lambda_evol = [L0]

        Q_dist = 1e-5
        R_v = 1e-5
        R_omega = 1e-5
        attraction = 0
        entropy_w = 40
        obj = 0

        opti.subject_to(X[:, 0] == X0)
        ca_batch = []

        for i in range(steps):
            opti.subject_to(opti.bounded(lb[0] - 50.0, X[0, i], ub[0] + 50.0))
            opti.subject_to(opti.bounded(lb[1] - 50.0, X[1, i], ub[1] + 50.0))
            opti.subject_to(opti.bounded(-2 * np.pi, X[2, i], 2 * np.pi))

            opti.subject_to(opti.bounded(-0.2, U[0, i], 0.2))
            opti.subject_to(opti.bounded(-0.15, U[1, i], 0.15))

            opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))

            # --- Evitamento pali ---
            robot_length = 1.0
            robot_width = 0.67
            margin_pole = 0.8
            Q_pole_hard = 100.0
            alpha_pole = 10.0
            for p in range(self.NUM_POLES):
                px, py = POLES_param[0, p], POLES_param[1, p]
                dx_p = px - X[0, i + 1]
                dy_p = py - X[1, i + 1]
                theta_rob = X[2, i + 1]
                x_rel_p = dx_p * ca.cos(theta_rob) + dy_p * ca.sin(theta_rob)
                y_rel_p = -dx_p * ca.sin(theta_rob) + dy_p * ca.cos(theta_rob)
                sigma_x_rob = (robot_length / 2.0) + margin_pole
                sigma_y_rob = (robot_width / 2.0) + margin_pole
                dist_norm_p = (x_rel_p / sigma_x_rob) ** 2 + (y_rel_p / sigma_y_rob) ** 2
                hard_potential_pole = Q_pole_hard * ca.exp(alpha_pole * (1.0 - dist_norm_p))
                obj = obj + hard_potential_pole

            # --- Evitamento aiuole ---
            margin_f = 1.0
            Q_flowerbed_hard = 50.0
            alpha_flowerbed = 5.0
            for f in range(self.NUM_FLOWERBEDS):
                fx = FLOWERBEDS_param[0, f]
                fy = FLOWERBEDS_param[1, f]
                flen = FLOWERBEDS_param[2, f]
                fwid = FLOWERBEDS_param[3, f]
                ftheta = FLOWERBEDS_param[4, f]
                dx_f = X[0, i + 1] - fx
                dy_f = X[1, i + 1] - fy
                x_rel_f = dx_f * ca.cos(ftheta) + dy_f * ca.sin(ftheta)
                y_rel_f = -dx_f * ca.sin(ftheta) + dy_f * ca.cos(ftheta)
                sigma_x_f = (flen / 2.0) + margin_f
                sigma_y_f = (fwid / 2.0) + margin_f
                dist_norm_f = (x_rel_f / sigma_x_f) ** 4 + (y_rel_f / sigma_y_f) ** 4
                repulsive_flowerbed = Q_flowerbed_hard * ca.exp(alpha_flowerbed * (1.0 - dist_norm_f))
                obj = obj + repulsive_flowerbed
                opti.subject_to(dist_norm_f >= 0.9)

            # --- Evitamento auto (repulsione + circumnavigazione tangenziale) ---
            car_length = 4
            car_width = 2
            A_rep = 50
            margin_x = 1
            margin_y = 1
            Q_circ_ori = 200.0
            direction_of_circ = 1
            for j in range(self.NUM_OBSTACLE_TREES):
                obs_j_pos = OBSTACLE_TREES_param[:, j]
                car_x, car_y, car_theta = obs_j_pos[0], obs_j_pos[1], obs_j_pos[2]
                center_x = car_x - (car_length / 2.0) * ca.cos(car_theta)
                center_y = car_y - (car_length / 2.0) * ca.sin(car_theta)
                dx = X[0, i + 1] - center_x
                dy = X[1, i + 1] - center_y
                x_rel = dx * ca.cos(car_theta) + dy * ca.sin(car_theta)
                y_rel = -dx * ca.sin(car_theta) + dy * ca.cos(car_theta)
                sigma_x = (car_length / 2.0) + margin_x
                sigma_y = (car_width / 2.0) + margin_y
                dist_norm = (x_rel / sigma_x) ** 2 + (y_rel / sigma_y) ** 2
                repulsive_cost = A_rep * ca.exp(alpha_flowerbed * (1.0 - dist_norm))
                v_radial = ca.vcat([x_rel, y_rel])
                u_radial = v_radial / (ca.norm_2(v_radial) + 1e-6)
                if direction_of_circ == 1:
                    u_tangential = ca.vcat([-u_radial[1], u_radial[0]])
                else:
                    u_tangential = ca.vcat([u_radial[1], -u_radial[0]])
                angle_tangential_ref_car_frame = ca.atan2(u_tangential[1], u_tangential[0])
                theta_fut = X[2, i + 1]
                theta_fut_car_frame = theta_fut - car_theta
                angle_diff = angle_tangential_ref_car_frame - theta_fut_car_frame
                angle_diff_norm = ca.atan2(ca.sin(angle_diff), ca.cos(angle_diff))
                cost_circ = Q_circ_ori * ca.exp(-dist_norm) * angle_diff_norm ** 2
                obj = obj + repulsive_cost + cost_circ
                core_safety = 0.8
                opti.subject_to(dist_norm >= core_safety ** 2)

            theta_fut = X[2, i + 1]
            distances_sq = []
            nn_batch = []
            side_observation_cost = 0
            for j in range(self.NUM_TARGET_TREES):
                obj_j_pos = TARGET_TREES_param[:, j]
                theta_target = obj_j_pos[2]
                dX = obj_j_pos[0] - X[0, i + 1]
                dY = obj_j_pos[1] - X[1, i + 1]
                dX, dY = -dX, -dY
                dist_sq = dX ** 2 + dY ** 2 + 1e-6
                distances_sq.append(dist_sq)
                x_rel = dX * ca.cos(theta_target) + dY * ca.sin(theta_target)
                y_rel = -dX * ca.sin(theta_target) + dY * ca.cos(theta_target)
                theta_y_robot = theta_fut + (ca.pi / 2.0)
                azimuth_raw = theta_y_robot - theta_target + ca.pi
                azimuth_norm = ca.atan2(ca.sin(azimuth_raw), ca.cos(azimuth_raw))
                nn_input = ca.horzcat(x_rel, y_rel, azimuth_norm)
                nn_batch.append(nn_input)

                dist_to_thresh_sq = (x_rel - OPT_THRESH_param[0]) ** 2 + (y_rel - OPT_THRESH_param[1]) ** 2
                angle_diff_raw = OPT_THRESH_param[2] - azimuth_norm
                angle_diff_norm = ca.atan2(ca.sin(angle_diff_raw), ca.cos(angle_diff_raw))
                angle_error_sq = angle_diff_norm ** 2
                Q_thresh_pos = 15.0
                Q_thresh_ori = 10.0
                obj = obj + Q_thresh_pos * dist_to_thresh_sq + Q_thresh_ori * angle_error_sq

            ca_batch.append(ca.vcat([*nn_batch]))

            min_dist_sq = distances_sq[0]
            for j in range(1, self.NUM_TARGET_TREES):
                min_dist_sq = ca.fmin(min_dist_sq, distances_sq[j])
            attraction = attraction + min_dist_sq * Q_dist

            obj = obj + R_v * (U[0, i] ** 2) + R_omega * (U[1, i] ** 2)

        nn_full_batch_input = ca.vcat(ca_batch)
        surrogate_output_ripe = self.l4c_nn[0](nn_full_batch_input)
        surrogate_output_raw = self.l4c_nn[1](nn_full_batch_input)

        prob_ripe = 1.0 / (1.0 + ca.exp(-surrogate_output_ripe))
        prob_raw = 1.0 / (1.0 + ca.exp(-surrogate_output_raw))

        L0_ext = ca.vcat([L0 for _ in range(steps)])
        sel = L0_ext[:, 0] >= L0_ext[:, 1]
        p_selected = ca.if_else(sel, prob_ripe, prob_raw)

        p_mapped = 0.5 + 0.5 * p_selected
        z_k_bin = ca.horzcat(p_mapped, 1.0 - p_mapped)

        for i in range(steps):
            lambda_next = self.bayes(lambda_evol[-1], z_k_bin[i * self.NUM_TARGET_TREES:(i + 1) * self.NUM_TARGET_TREES, :])
            lambda_evol.append(lambda_next)

        entropy_obj = 0
        for i in range(1, steps + 1):
            entropy_future = self.entropy_target(lambda_evol[i])
            entropy_obj += ca.exp(-2 * i) * ca.logsumexp(-entropy_w * entropy_future)

        sq_dist_to_targets = ca.sum1((X0[:2] - TARGET_TREES_param[:2, :]) ** 2)
        min_sq_dist = ca.mmin(sq_dist_to_targets)

        threshold_sq_dist = 15.0
        sigmoid_steepness = 0.5
        sigmoid_factor = 1.0 / (1.0 + ca.exp(-sigmoid_steepness * (min_sq_dist - threshold_sq_dist)))
        modulated_attraction_term = attraction * sigmoid_factor
        modulated_side_cost = side_observation_cost * sigmoid_factor

        last_azimuth = nn_batch[-1][2]
        terminal_angle_diff = ca.atan2(ca.sin(OPT_THRESH_param[2] - last_azimuth), ca.cos(OPT_THRESH_param[2] - last_azimuth))
        obj = obj + 100.0 * (terminal_angle_diff ** 2)

        opti.minimize(obj - 0.1 * entropy_obj)

        if self.offline_mode:
            # Profilo "planner offline": nessun vincolo di 5Hz, quindi
            # lasciamo IPOPT convergere davvero invece di fermarsi presto.
            # - max_iter molto piu' alto (niente deadline di controllo)
            # - tol/acceptable_tol piu' stretti -> soluzione piu' precisa
            # - acceptable_iter piu' alto -> non accontentarsi subito di un
            #   plateau numerico
            options = {
                "ipopt": {
                    "tol": 1e-6,
                    "acceptable_tol": 1e-5,
                    "acceptable_iter": 15,
                    "max_iter": 2000,
                    "warm_start_init_point": "yes",
                    "print_level": 0,
                    "sb": "no",
                    "hessian_approximation": 'limited-memory'
                }
            }
        else:
            # Profilo "controllo real-time" (IDENTICO all'originale
            # nmpc_car_massimi.py, pensato per stare sotto dt=0.2s)
            options = {
                "ipopt": {
                    "tol": 1e-4,
                    "acceptable_tol": 1e-3,
                    "acceptable_iter": 5,
                    "max_iter": 100,
                    "warm_start_init_point": "yes",
                    "print_level": 0,
                    "sb": "no",
                    "hessian_approximation": 'limited-memory'
                }
            }
        opti.solver("ipopt", options)
        inputs = [P0, opti.x, opti.lam_g]
        outputs = [U[:, 0], X, opti.x, opti.lam_g]

        p0_val = ca.vertcat(
            x0,
            ca.reshape(target_trees, 3 * self.NUM_TARGET_TREES, 1),
            ca.reshape(target_lambdas, 2 * self.NUM_TARGET_TREES, 1),
            ca.reshape(obstacle_trees, 3 * self.NUM_OBSTACLE_TREES, 1),
            ca.DM(closest_thresh),
            ca.reshape(self.poles_pos, 2 * self.NUM_POLES, 1),
            ca.reshape(self.flowerbeds_pos, 5 * self.NUM_FLOWERBEDS, 1)
        )
        opti.set_value(P0, p0_val)

        opti.set_initial(X, ca.repmat(x0, 1, steps + 1))
        opti.set_initial(U, ca.DM.zeros(self.n_control, steps))

        sol = opti.solve()
        mpc_step_func = opti.to_function("mpc_step", inputs, outputs, ["p", "x_init", "x_lam"], ["u_opt", "x_pred", "x_opt", "lam_opt"])

        return (mpc_step_func, ca.DM(sol.value(U[:, 0])), ca.DM(sol.value(X)),
                ca.DM(sol.value(opti.x)), ca.DM(sol.value(opti.lam_g)))

    # ========================================================================
    # Plot in tempo reale
    # ========================================================================
    def _init_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.grid(True, alpha=0.3)

        # Pali + linea spartitraffico (usata da _barrier_aware_distance
        # per capire se un target e' raggiungibile in linea retta)
        if self.NUM_POLES > 0:
            self.ax.scatter(self.poles_pos[:, 0], self.poles_pos[:, 1],
                             c='gray', marker='|', s=100, label='pali')
            self.ax.plot(self._poles_sorted[:, 0], self._poles_sorted[:, 1],
                         c='gray', linestyle=':', linewidth=1.5, alpha=0.8,
                         label='barriera pali')

        # Aiuole (rettangoli ruotati, approssimazione visiva)
        for fb in self.flowerbeds_pos:
            fx, fy, flen, fwid, ftheta = fb
            corners = np.array([
                [-flen / 2, -fwid / 2], [flen / 2, -fwid / 2],
                [flen / 2, fwid / 2], [-flen / 2, fwid / 2]
            ])
            c, s = math.cos(ftheta), math.sin(ftheta)
            R = np.array([[c, -s], [s, c]])
            rotated = corners @ R.T + np.array([fx, fy])
            poly = MplPolygon(rotated, closed=True, facecolor='green', alpha=0.25, edgecolor='green')
            self.ax.add_patch(poly)

        # Auto/alberi target.
        # IMPORTANTE: in trees_pos[i] = (x, y, theta) il punto (x, y) e' il
        # muso/riferimento frontale dell'auto (stessa convenzione usata sia
        # in mpc_opt per l'evitamento ostacoli, sia in spawn_cars_in_gazebo
        # del nodo mock), NON il centro. Il corpo dell'auto si estende
        # all'INDIETRO rispetto a (x, y) lungo -theta.
        self.car_patches = []
        self.car_safety_patches = []
        # Dimensioni reali del veicolo (identiche a quelle spawnate in
        # Gazebo da mock_car_score.py: l_car, w_car)
        real_car_length, real_car_width = 3.46, 1.62
        # Dimensioni dell'ellisse di sicurezza usata dai vincoli MPC
        # (car_length, car_width in mpc_opt)
        safety_car_length, safety_car_width = 4.0, 2.0

        def _car_corners(cx_ref, cy_ref, theta, length, width):
            # Centro geometrico dell'auto: arretrato di length/2 dal
            # riferimento frontale, esattamente come in mpc_opt
            center = np.array([
                cx_ref - (length / 2.0) * math.cos(theta),
                cy_ref - (length / 2.0) * math.sin(theta),
            ])
            corners = np.array([
                [-length / 2, -width / 2], [length / 2, -width / 2],
                [length / 2, width / 2], [-length / 2, width / 2]
            ])
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s], [s, c]])
            return corners @ R.T + center

        for i, tr in enumerate(self.trees_pos):
            tx, ty, ttheta = tr

            # Contorno tratteggiato = margine di sicurezza usato dai
            # vincoli dell'MPC (utile per capire perche' il robot mantiene
            # una certa distanza)
            safety_pts = _car_corners(tx, ty, ttheta, safety_car_length, safety_car_width)
            safety_poly = MplPolygon(safety_pts, closed=True, facecolor='none',
                                      edgecolor='orange', linestyle='--', linewidth=1.0, alpha=0.7)
            self.ax.add_patch(safety_poly)
            self.car_safety_patches.append(safety_poly)

            # Corpo reale dell'auto (dimensioni Gazebo)
            body_pts = _car_corners(tx, ty, ttheta, real_car_length, real_car_width)
            poly = MplPolygon(body_pts, closed=True, facecolor='red', alpha=0.6, edgecolor='black')
            self.ax.add_patch(poly)
            self.car_patches.append(poly)

            # Freccia dal riferimento frontale nella direzione theta, per
            # rendere immediatamente visibile l'orientamento dell'auto
            self.ax.annotate('', xy=(tx + 0.8 * math.cos(ttheta), ty + 0.8 * math.sin(ttheta)),
                              xytext=(tx, ty),
                              arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

            # Auto NON raggiungibile (tutti i punti soglia bloccati da
            # altre auto/pali/aiuola): marcatura permanente con una X nera,
            # sempre disegnata sopra qualsiasi colore di credenza.
            if not self.trees_reachable[i]:
                self.ax.plot(tx, ty, marker='X', markersize=14, color='black',
                              markeredgecolor='white', markeredgewidth=1.2, zorder=6,
                              label='non esplorabile' if 'non esplorabile' not in
                              [l.get_label() for l in self.ax.get_lines()] else None)

        # Marker del "massimo" (punto soglia) che il robot sta attualmente
        # cercando di raggiungere: stella gialla + freccia con l'orientamento
        # desiderato, aggiornati ad ogni step in _update_plot
        self.target_star, = self.ax.plot([], [], marker='*', markersize=20,
                                          color='gold', markeredgecolor='black',
                                          markeredgewidth=1.0, zorder=7, label='massimo target')
        self.target_arrow = None

        # Robot: triangolo + traiettoria + orizzonte predetto
        self.robot_patch = MplPolygon(np.zeros((3, 2)), closed=True, facecolor='blue', edgecolor='black', zorder=5)
        self.ax.add_patch(self.robot_patch)
        self.traj_line, = self.ax.plot([], [], 'b-', linewidth=1.0, alpha=0.6, label='traiettoria')
        self.pred_line, = self.ax.plot([], [], 'c--', linewidth=1.5, label='orizzonte MPC')

        lb, ub = self.get_domain(self.trees_pos)
        margin = 15
        self.ax.set_xlim(lb[0] - margin, ub[0] + margin)
        self.ax.set_ylim(lb[1] - margin, ub[1] + margin)
        self.ax.legend(loc='upper right', fontsize=8)
        self.title = self.ax.set_title("Simulazione NMPC standalone")
        self._plot_initialized = True
        self._traj_xy = []
        plt.show(block=False)
        plt.pause(0.001)

    def _update_plot(self, state_now, x_traj, entropy_val, mpciter, beliefs, target_point=None):
        if not self._plot_initialized:
            self._init_plot()

        # Colora le auto in base alla credenza corrente (rosso=raw, verde=ripe)
        beliefs_np = beliefs.full()
        for patch, bel in zip(self.car_patches, beliefs_np):
            color = 'green' if bel[1] > bel[0] else 'red'
            patch.set_facecolor(color)
            patch.set_alpha(0.3 + 0.5 * max(bel[0], bel[1]))

        # Robot: triangolo orientato
        x, y, theta = state_now
        L = 1.2
        tri = np.array([
            [L, 0], [-L * 0.6, L * 0.5], [-L * 0.6, -L * 0.5]
        ])
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        tri_rot = tri @ R.T + np.array([x, y])
        self.robot_patch.set_xy(tri_rot)

        self._traj_xy.append((x, y))
        traj_arr = np.array(self._traj_xy)
        self.traj_line.set_data(traj_arr[:, 0], traj_arr[:, 1])

        pred_xy = np.array(x_traj[:2, :]).T
        self.pred_line.set_data(pred_xy[:, 0], pred_xy[:, 1])

        # "Massimo" target corrente (punto soglia scelto da
        # get_closest_threshold_state, convertito in coordinate globali)
        if self.target_arrow is not None:
            self.target_arrow.remove()
            self.target_arrow = None
        if target_point is not None:
            tgt_x, tgt_y, tgt_theta = target_point
            self.target_star.set_data([tgt_x], [tgt_y])
            self.target_arrow = self.ax.annotate(
                '', xy=(tgt_x + 1.2 * math.cos(tgt_theta), tgt_y + 1.2 * math.sin(tgt_theta)),
                xytext=(tgt_x, tgt_y),
                arrowprops=dict(arrowstyle='->', color='goldenrod', lw=2.0), zorder=7)
        else:
            self.target_star.set_data([], [])

        self.title.set_text(f"Step {mpciter} | Entropia globale: {entropy_val:.4f}")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    # ========================================================================
    # Loop di simulazione (sostituisce run_simulation, senza ROS)
    # ========================================================================
    def run_simulation(self, max_iter=100000, real_time=False):
        lb, ub = self.get_domain(self.trees_pos)

        x_k = ca.DM(self.current_state)

        all_trajectories = []
        lambda_history, entropy_history, durations = [], [], []
        velocity_command_log, pose_history, time_history = [], [], []

        sim_start_time = time.time()
        total_distance = 0.0
        total_commands = 0
        prev_x, prev_y = float(x_k[0]), float(x_k[1])

        mpciter = 0
        warm_start = True
        x_dec_prev, lam_g_prev, mpc_step = None, None, None

        try:
            while mpciter < max_iter:
                loop_iter_start = time.time()
                print(f"[Control Loop] Step: {mpciter}")

                state_now = self.current_state
                current_sim_time = time.time() - sim_start_time
                pose_history.append(state_now)
                time_history.append(current_sim_time)
                x_k = ca.DM(state_now)

                # --- Data association / scoring (sostituisce il topic /parking/scores) ---
                scores = self.compute_tree_scores(state_now)
                if mpciter % 2 == 0:
                    self.beliefs_k = self.bayes(self.beliefs_k, ca.DM(scores))

                robot_position_xy = np.array(state_now[:2])

                current_target_idx = self.get_current_target_index()
                if current_target_idx is None:
                    print("[Info] Route completata: tutte le auto raggiungibili sono state esplorate.")
                    current_target_idx = self.visit_order[-1] if self.visit_order else 0

                target_indices = np.array([current_target_idx])
                obstacle_indices = self.get_nearest_tree_indices(robot_position_xy, num_obstacle=self.NUM_OBSTACLE_TREES)

                target_trees_subset = self.trees_pos[target_indices]
                obstacle_trees_subset = self.trees_pos[obstacle_indices]
                target_lambdas = self.beliefs_k[target_indices, :]

                robot_side_now = self._pole_line_side(robot_position_xy)
                closest_thresh_state = self.get_closest_threshold_state(state_now, target_trees_subset, preferred_side=robot_side_now)               

                # Conversione del "massimo" (punto soglia scelto, in
                # coordinate relative all'auto target) in coordinate
                # globali, solo per la visualizzazione. Valida per
                # NUM_TARGET_TREES == 1 (default), dove target_trees_subset
                # contiene un'unica auto di riferimento.
                target_point_global = None
                if self.show_plot and self.NUM_TARGET_TREES == 1:
                    ttx, tty, ttheta = target_trees_subset[0]
                    p_x_rel, p_y_rel, p_azimuth = closest_thresh_state
                    tgt_x = ttx + (p_x_rel * math.cos(ttheta)) - (p_y_rel * math.sin(ttheta))
                    tgt_y = tty + (p_x_rel * math.sin(ttheta)) + (p_y_rel * math.cos(ttheta))
                    tgt_theta_raw = p_azimuth + ttheta + (math.pi / 2.0)
                    tgt_theta = math.atan2(math.sin(tgt_theta_raw), math.cos(tgt_theta_raw))
                    target_point_global = (tgt_x, tgt_y, tgt_theta)

                step_start_time = time.time()
                try:
                    if warm_start or mpc_step is None:
                        mpc_step, u, x_traj, x_dec_prev, lam_g_prev = self.mpc_opt(
                            target_trees_subset, target_lambdas, obstacle_trees_subset,
                            closest_thresh_state, lb, ub, x_k, steps=self.N
                        )
                        warm_start = False
                    else:
                        P0_val = ca.vertcat(
                            x_k,
                            ca.reshape(target_trees_subset, 3 * self.NUM_TARGET_TREES, 1),
                            ca.reshape(target_lambdas, 2 * self.NUM_TARGET_TREES, 1),
                            ca.reshape(obstacle_trees_subset, 3 * self.NUM_OBSTACLE_TREES, 1),
                            ca.DM(closest_thresh_state),
                            ca.reshape(self.poles_pos, 2 * self.NUM_POLES, 1),
                            ca.reshape(self.flowerbeds_pos, 5 * self.NUM_FLOWERBEDS, 1)
                        )
                        u, x_traj, x_dec_prev, lam_g_prev = mpc_step(P0_val, x_dec_prev, lam_g_prev)
                    step_duration = time.time() - step_start_time
                    durations.append(step_duration)

                except Exception as e:
                    print(f"[ERROR] Eccezione fatale durante la risoluzione MPC al ciclo {mpciter}: {e}")
                    break

                v_cmd = float(u[0])
                omega_cmd = float(u[1])

                velocity_command_log.append([current_sim_time, "MPC", v_cmd, omega_cmd])
                total_commands += 1

                curr_x, curr_y = float(x_traj[0, 1]), float(x_traj[1, 1])
                total_distance += math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                prev_x, prev_y = curr_x, curr_y

                entropy_k = self.entropy_entire_field(self.beliefs_k)
                lambda_history.append(self.beliefs_k.full().flatten().tolist())
                entropy_history.append(ca.sum1(entropy_k).full().flatten()[0])
                all_trajectories.append(x_traj[:self.nx, :].full())

                # --- "Simulazione" del robot: applichiamo lo stato predetto dal
                # modello cinematico interno all'NMPC (motion ground-truth =
                # modello perfetto, standard in simulazione NMPC pura) ---
                self.current_state = [curr_x, curr_y, float(x_traj[2, 1])]

                if self.show_plot and (mpciter % 10 == 0):
                    self._update_plot(state_now, np.array(x_traj[:self.nx, 1:]),
                                       entropy_history[-1], mpciter, self.beliefs_k,
                                       target_point=target_point_global)

                mpciter += 1
                print(f"[Info] Entropia globale del sistema: {entropy_history[-1]}")

                if all(v <= self.threshold_entropy for v in entropy_k.full().flatten()):
                    print("[Info] Target informativo di riduzione entropia completato.")
                    # (non interrompiamo il loop, come nell'originale)

                loop_elapsed = time.time() - loop_iter_start
                if real_time:
                    sleep_time = self.dt - loop_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        print(f"[WARN] Frequenza di ciclo violata! L'ottimizzazione ha impiegato "
                              f"{loop_elapsed:.4f}s rispetto al limite di {self.dt}s")

        except KeyboardInterrupt:
            print("[Info] Rilevato Ctrl+C. Interruzione del loop e avvio del salvataggio dati.")

        finally:
            print("[Info] Arresto del robot e salvataggio in formato CSV...")
            self.save_performance_data(sim_start_time, total_distance, total_commands,
                                        entropy_history, time_history, pose_history,
                                        lambda_history, velocity_command_log)
            if self.show_plot and self._plot_initialized:
                plt.ioff()
                print("[Info] Chiudi la finestra del plot per terminare.")
                plt.show()

        return all_trajectories, entropy_history, lambda_history, durations, self.l4c_nn, self.trees_pos, lb, ub

    # ========================================================================
    # Salvataggio CSV (IDENTICO a nmpc_car_massimi.py)
    # ========================================================================
    def save_performance_data(self, start_time, dist, cmds, entropy_hist, time_hist,
                               pose_hist, lambda_hist, vel_log):
        total_execution_time = time.time() - start_time
        avg_wp_time = total_execution_time / cmds if cmds > 0 else 0.0

        os.makedirs(self.baselines_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        perf_csv = os.path.join(self.baselines_dir, f"husky_mpc_{timestamp}_metrics.csv")
        with open(perf_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([
                ["Total Execution Time (s)", total_execution_time],
                ["Total Distance (m)", dist],
                ["Average Step Time (s)", avg_wp_time],
                ["Final Entropy", entropy_hist[-1] if entropy_hist else "N/A"],
                ["Total Commands Issued", cmds]
            ])

        vel_csv = os.path.join(self.baselines_dir, f"husky_mpc_{timestamp}_velocities.csv")
        with open(vel_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time (s)", "Tag", "linear_velocity_v", "angular_velocity_omega"])
            writer.writerows(vel_log)

        plot_csv = os.path.join(self.baselines_dir, f"husky_mpc_{timestamp}_plot_data.csv")
        with open(plot_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["tree_positions"] + self.trees_pos.flatten().tolist())
            writer.writerow(["trees_gt_id"] + self.trees_gt_id.tolist())

            header = ["time", "x", "y", "theta", "entropy"]
            if lambda_hist:
                header += [f"lambda_{i}" for i in range(len(lambda_hist[0]))]
            writer.writerow(header)

            for i in range(len(time_hist)):
                x, y, theta = pose_hist[i]
                row = [time_hist[i], x, y, theta, entropy_hist[i]] + lambda_hist[i]
                writer.writerow(row)

        print(f"[Info] Report delle prestazioni salvato correttamente in: {self.baselines_dir}")


# ============================================================================
# Entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Simulazione NMPC standalone (senza ROS)")
    parser.add_argument("--no-plot", dest="show_plot", action="store_false",
                         help="Disabilita il plot in tempo reale")
    parser.add_argument("--max-iter", type=int, default=100000,
                         help="Numero massimo di iterazioni di controllo (default: 100000)")
    parser.add_argument("--real-time", action="store_true",
                         help="Ritma il loop sul dt reale (0.2s), utile per una visualizzazione fluida")
    parser.add_argument("--run-dir", type=str, default=None,
                         help="Cartella di output per i CSV (default: ../../baselines relativo allo script)")
    parser.add_argument("--map", type=str, default=None,
                         help="Percorso al file car_map_final.json (default: percorso hardcoded originale)")
    parser.add_argument("--models-dir", type=str, default=None,
                         help="Cartella contenente le sottocartelle dei modelli 'car' (best_model_epoch_N.pth)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                         help="Device torch/L4CasADi da usare (default: auto)")
    parser.add_argument("--N", type=int, default=None,
                         help="Numero di step dell'orizzonte MPC (default originale: 20). "
                              "Aumentarlo allunga l'orizzonte T=dt*N e/o ne aumenta la "
                              "risoluzione temporale.")
    parser.add_argument("--dt", type=float, default=None,
                         help="Passo di discretizzazione del controllo in secondi "
                              "(default originale: 0.2). Ridurlo aumenta la risoluzione "
                              "temporale a parita' di N.")
    parser.add_argument("--offline", action="store_true",
                         help="Usa il profilo IPOPT da planner offline (max_iter e "
                              "tolleranze molto piu' spinti, niente compromesso da 5Hz). "
                              "Consigliato insieme a --N piu' alto e/o --dt piu' basso.")
    parser.set_defaults(show_plot=True)
    parser.add_argument("--interactive-route", action="store_true", default=True,
                     help="Apre una finestra per scegliere manualmente l'ordine di "
                          "visita delle auto prima di avviare la simulazione")
    args = parser.parse_args()

    sim = NeuralMPCStandaloneSim(
        run_dir=args.run_dir,
        map_path=args.map,
        models_dir=args.models_dir,
        device=args.device,
        show_plot=args.show_plot,
        N=args.N,
        dt=args.dt,
        offline_mode=args.offline,
    )
    if args.interactive_route:
        sim.select_route_interactively()
    else:
        sim._compute_visit_order()  # fallback automatico (NN + 2-opt + Or-opt)

    sim.run_simulation(max_iter=args.max_iter, real_time=args.real_time)

if __name__ == "__main__":
    main()