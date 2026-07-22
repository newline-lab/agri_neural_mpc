#!/usr/bin/env python
import os
import re
import time
import csv
import math
import threading
import json

import rospy
import casadi as ca
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm

import l4casadi as l4c
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray
import tf

from geometry_msgs.msg import Pose, Point, Quaternion, Pose2D


torch.jit.set_fusion_strategy([('STATIC', 0)])

# Nota: Assicurati che queste funzioni custom nel tuo workspace ROS accettino la posa [x, y, theta]
from nmpc_ros_package.ros_com_lib.sensors import create_path_from_mpc_prediction, create_tree_markers


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=64, hidden_layers=3):
        super().__init__()
        # If input_dim==3 we add an extra input for sin/cos of the angle.
        in_features = input_dim if input_dim != 3 else input_dim + 1
        self.input_layer = torch.nn.Linear(in_features, hidden_size)
        self.hidden_layer = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        self.out_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        # If the last dimension is 3, assume the third element is an angle
        # and replace it with its sin and cos.
        if x.shape[-1] == 3:
            sin_cos = torch.cat([torch.sin(x[..., -1:]), torch.cos(x[..., -1:])], dim=-1)
            x = torch.cat([x[..., :-1], sin_cos], dim=-1)
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x


class NeuralMPCHusky:
    def __init__(self, run_dir=None, trajectory_csv_path="/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/nmpc_ros/trajectory2follow/test_traj.csv"):
        self.hidden_size = 64
        self.hidden_layers = 3
        self.nn_input_dim = 3

        self.N = 5
        self.dt = 0.2  # Controllo a 5 Hz
        self.T = self.dt * self.N
        
        # Specifiche del robot reale (Uniciclo)
        self.nx = 3          # Stato: [x, y, theta]
        self.n_state = 3
        self.n_control = 2   # Ingressi di controllo: [v, omega]
        
        self.NUM_TARGET_TREES = 1   # subset alberi vicini da esplorare
        self.NUM_OBSTACLE_TREES = 3 # subset alberi vicini da evitare

        self.threshold_entropy = 0.15 # Quando albero considerato visto

        # ----------------------------------------------------------------------
        # Massimi locali
        # ----------------------------------------------------------------------
        self.punti_soglia = [
            [-1.3037, -2.5762, -2.0944], # 1
            [-0.934, -2.3745, -1.5708],  # 2
            [3.2101, 0.1273, -0.0],      # 3
            # [2.9268, 0.569, 0.5236],
            # [0.1753, 3.5078, 1.0472],    # 4
            [-1.1693, 3.066, 1.5708],    # 5
            # [-0.4298, 2.9508, 2.0944],   # 6
            # [-1.5582, 3.2533, 2.618],
        ]

        # ----------------------------------------------------------------------
        # COORDINATE HARDCODED DEGLI ALBERI (Origine coincidente con lo zero dell'Odom)
        # ----------------------------------------------------------------------
        # self.trees_pos = np.array([
        #     [-4.0, -1.0, 0.0],
        #     # [-4.0, 5.0, 0.0],
        #     # [-3.5, 10.0, 0.0],
        #     [4.0, 15.0, -np.pi]
        # ], dtype=np.float32)
        file_path = "/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/nmpc_ros/niccolo/map_results/car_map_final.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                cars = json.load(f).get("cars", [])
            # Ordiniamo la lista dei dizionari in base al valore della chiave "id"
            cars_sorted = sorted(cars, key=lambda c: c["id"])
            extracted = []
            for c in cars_sorted:
                x, y, rad = c["x"], c["y"], c["orientation_rad"]
                # if c["class"] == "car_back":
                #     x += 4.0 * np.cos(rad)
                #     y += 4.0 * np.sin(rad)
                extracted.append([x, y, rad])
            self.trees_pos = np.array(extracted, dtype=np.float32)
        else:
            # Fallback hardcoded se il file non esiste
            self.trees_pos = np.array(
                [
                    [43.04, -2.149, 1.4337],
                    [49.701, -7.093, -1.6995],
                    [52.743, -7.233, -1.7331],
                ],
                dtype=np.float32,
            )
        print(self.trees_pos)

        # All'interno di def __init__(self, run_dir=None):

        # ----------------------------------------------------------------------
        # COORDINATE PALI (x, y)
        # ----------------------------------------------------------------------
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
        # ----------------------------------------------------------------------
        # COORDINATE AIUOLE (x, y, lunghezza, larghezza, orientamento_rad)
        # L'orientamento ti permette di ruotare la superellisse.
        # [12.0, 10.0, 10.0, 2.0, 0.0], # Esempio: aiuola lunga 10m e larga 2m
        # ----------------------------------------------------------------------
        self.flowerbeds_pos = np.array([
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
        self.NUM_FLOWERBEDS = self.flowerbeds_pos.shape[0]

        # Identificativi reali stabili degli alberi (0: raw, 1: ripe)
        self.trees_gt_id = np.array([0, 1], dtype=np.uint8)
        
        self.num_total_trees = self.trees_pos.shape[0]
        self.entropy_entire_field = self.entropy_f(self.num_total_trees)
        self.beliefs_k = ca.DM.ones(self.num_total_trees, 2) * 0.5

        # ----------------------------------------------------------------------
        # Inizializzazione e caricamento delle reti neurali L4CasADi
        # ----------------------------------------------------------------------
        self.l4c_nn = []
        for label in ['car', 'car']:
            model = MultiLayerPerceptron(input_dim=self.nn_input_dim,
                                        hidden_size=self.hidden_size,
                                        hidden_layers=self.hidden_layers)
            model_load_path = self.get_latest_best_model(label)
            model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cuda')))
            model.eval()
            g_nn = l4c.L4CasADi(model, batched=True, device='cuda', name=label)
            self.l4c_nn.append(g_nn)

        self.entropy_target = self.entropy_f(self.NUM_TARGET_TREES)
        self.latest_trees_scores = None
        self.current_state = None  

        # ----------------------------------------------------------------------
        # Configurazione nodi e Topic ROS reali
        # ----------------------------------------------------------------------
        rospy.init_node("nmpc_husky_node", anonymous=True, log_level=rospy.DEBUG)
        
        # Sottoscrizioni ai sensori fisici e moduli di percezione
        # rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)
        rospy.Subscriber('/gps_data', Pose2D, self.gps_callback)
        rospy.Subscriber("/parking/scores", Float32MultiArray, self.tree_scores_callback)
        
        # Pubblicazioni per l'hardware e monitoraggio (Rviz)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pred_path_pub = rospy.Publisher("predicted_path", Path, queue_size=10)
        self.tree_markers_pub = rospy.Publisher("tree_markers", MarkerArray, queue_size=10)

        self.baselines_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../baselines") if run_dir is None else run_dir

        # ----------------------------------------------------------------------
        # Traiettoria di riferimento da inseguire (stessa struttura del
        # "plot_data.csv" salvato da save_performance_data: colonne
        # time,x,y,theta,entropy,lambda_0,...). Viene caricata una volta e
        # l'MPC insegue i punti IN SUCCESSIONE (nessuna sincronizzazione col
        # tempo di registrazione originale).
        # ----------------------------------------------------------------------
        self.wp_reach_radius = 0.8   # [m] distanza sotto la quale un waypoint si considera raggiunto
        self.ref_wp_idx = 0          # indice del prossimo waypoint da inseguire
        self.ref_trajectory = self.load_reference_trajectory(trajectory_csv_path)

    # def odom_callback(self, msg):
    #     """ Estrae la posa [x, y, theta] direttamente dal topic dell'odometria filtrata """
    #     px = msg.pose.pose.position.x
    #     py = msg.pose.pose.position.y
        
    #     # Conversione del quaternione in Yaw (Angolo Theta attorno all'asse Z)
    #     ori = msg.pose.pose.orientation
    #     quat = [ori.x, ori.y, ori.z, ori.w]
    #     (_, _, yaw) = tf.transformations.euler_from_quaternion(quat)
        
    #     self.current_state = [px, py, yaw]

    def gps_callback(self, msg):
        # Il messaggio Pose2D ha già x, y e theta (yaw)
        self.current_state = [msg.x, msg.y, msg.theta]

    def tree_scores_callback(self, msg):
        data = msg.data
        if len(data) % 2 != 0:
            rospy.logwarn("[tree_scores_callback] Ricevuto array di score non conforme (dispari).")
            return

        if msg.layout.dim and len(msg.layout.dim) >= 2:
            shape = (msg.layout.dim[0].size, msg.layout.dim[1].size)
        else:
            shape = (len(data) // 2, 2)

        try:
            raw_scores = np.array(data).reshape(shape)
            # Se la shape è invertita (2, N) invece di (N, 2), trasponi la matrice
            if raw_scores.shape[1] != 2 and raw_scores.shape[0] == 2:
                raw_scores = raw_scores.T
            
            # Mapping da [0, 1] a [0.5, 1]
            scores = np.zeros_like(raw_scores)
            scores[:, 0] = 0.5 + 0.5 * raw_scores[:, 0] 
            scores[:, 1] = 1.0 - scores[:, 0]
        except ValueError as e:
            rospy.logerr(f"[tree_scores_callback] Errore di rimodulazione shape degli score: {e}")
            return

        self.latest_trees_scores = scores.copy()

    def get_latest_best_model(self, cls=''):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, "models", cls)
        model_files = [f for f in os.listdir(model_dir) if re.match(r"best_model_epoch_(\d+)\.pth", f)]
        if not model_files:
            raise FileNotFoundError(f"Nessun file di modello pesi (.pth) trovato in {model_dir}")
        latest_model = max(model_files, key=lambda x: int(re.match(r"best_model_epoch_(\d+)\.pth", x).group(1)))
        return os.path.join(model_dir, latest_model)

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

        # Equazioni differenziali non-olonome
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
        entropy_per_target = -ca.sum2(p_clipped * (ca.log(p_clipped)/ca.log(2)))
        return ca.Function(f'entropy_f_{num_targets}_dim', [p], [entropy_per_target])

    def get_target_tree_indices(self, robot_position, num_target=None):

        entropy_threshold=self.threshold_entropy

        if num_target is None:
            num_target = self.NUM_TARGET_TREES
        distances = np.linalg.norm(self.trees_pos[:, :2] - robot_position, axis=1)
        H = self.entropy_entire_field(self.beliefs_k).full()
        candidate_indices = np.where(H > entropy_threshold)[0]
        
        if candidate_indices.size == 0:
            return np.argsort(distances)[:num_target]

        sorted_candidates = candidate_indices[np.argsort(distances[candidate_indices])]
        if sorted_candidates.size < num_target:
            repeats = int(np.ceil(num_target / sorted_candidates.size))
            sorted_candidates = np.tile(sorted_candidates, repeats)[:num_target]
        return sorted_candidates[:num_target]

    def get_nearest_tree_indices(self, robot_position, num_obstacle=None):
        distances = np.linalg.norm(self.trees_pos[:, :2] - robot_position, axis=1)
        return np.argsort(distances)[:self.NUM_OBSTACLE_TREES]

    def _find_latest_plot_data_csv(self):
        """Cerca l'ultimo file '*_plot_data.csv' nella cartella dei baseline."""
        if not os.path.isdir(self.baselines_dir):
            return None
        candidates = [f for f in os.listdir(self.baselines_dir) if f.endswith("_plot_data.csv")]
        if not candidates:
            return None
        candidates.sort()
        return os.path.join(self.baselines_dir, candidates[-1])

    def load_reference_trajectory(self, csv_path=None):
        """
        Carica la traiettoria [x, y, theta] da un file CSV con la stessa
        struttura del "plot_data.csv" prodotto da save_performance_data:
            riga 0: "tree_positions", ...
            riga 1: "trees_gt_id", ...
            riga 2: header (time, x, y, theta, entropy, lambda_0, ...)
            righe successive: dati
        Se non viene passato un percorso, cerca automaticamente l'ultimo
        file "*_plot_data.csv" salvato in self.baselines_dir.
        """
        if csv_path is None:
            csv_path = self._find_latest_plot_data_csv()

        if csv_path is None or not os.path.exists(csv_path):
            rospy.logwarn(f"[load_reference_trajectory] Nessun file di traiettoria trovato (path={csv_path}). "
                           "L'MPC non avrà una traiettoria da inseguire finché non ne viene fornita una.")
            return np.zeros((0, 3), dtype=np.float64)

        with open(csv_path, newline='') as f:
            rows = list(csv.reader(f))

        if len(rows) < 4:
            rospy.logwarn(f"[load_reference_trajectory] File '{csv_path}' non contiene dati sufficienti.")
            return np.zeros((0, 3), dtype=np.float64)

        header = rows[2]
        try:
            idx_x = header.index("x")
            idx_y = header.index("y")
            idx_theta = header.index("theta")
        except ValueError:
            rospy.logerr(f"[load_reference_trajectory] Header non conforme in '{csv_path}': {header}")
            return np.zeros((0, 3), dtype=np.float64)

        data = []
        step_row = 3
        for row in rows[3::step_row]:
            if not row:
                continue
            data.append([float(row[idx_x]), float(row[idx_y]), float(row[idx_theta])])

        traj = np.array(data, dtype=np.float64)
        rospy.loginfo(f"[load_reference_trajectory] Caricati {traj.shape[0]} waypoint da '{csv_path}'.")
        return traj

    def get_reference_window(self, robot_xy, steps):
        if self.ref_trajectory is None or self.ref_trajectory.shape[0] == 0:
            # Fallback di sicurezza: resta fermo sul posto
            return np.tile([robot_xy[0], robot_xy[1], 0.0], (steps + 1, 1))
        last_idx = self.ref_trajectory.shape[0] - 1
        # 1. LOGICA STRETTAMENTE SEQUENZIALE
        # Avanza di indice SOLO se il waypoint corrente è stato fisicamente raggiunto.
        while self.ref_wp_idx < last_idx:
            dist_curr = math.hypot(self.ref_trajectory[self.ref_wp_idx, 0] - robot_xy[0],
                                   self.ref_trajectory[self.ref_wp_idx, 1] - robot_xy[1])
            if dist_curr < self.wp_reach_radius:
                self.ref_wp_idx += 1
            else:
                break
        # 2. IL TRUCCO PER FERMARE L'OSCILLAZIONE:
        # Riempiamo l'intera finestra predittiva dell'MPC con lo STESSO IDENTICO PUNTO.
        # Niente più conflitti temporali: il robot ha un solo ed unico obiettivo e 
        # non viene "tirato in avanti" da punti futuri che non può ancora raggiungere.
        window = np.empty((steps + 1, 3), dtype=np.float64)
        for k in range(steps + 1):
            window[k, :] = self.ref_trajectory[self.ref_wp_idx, :]
        return window

    # def get_reference_window(self, robot_xy, steps):
    #     if self.ref_trajectory is None or self.ref_trajectory.shape[0] == 0:
    #         # Fallback di sicurezza
    #         return np.tile([robot_xy[0], robot_xy[1], 0.0], (steps + 1, 1))
    #     last_idx = self.ref_trajectory.shape[0] - 1
    #     # 1. Aggiorna l'indice del waypoint corrente basandosi sulla distanza
    #     while self.ref_wp_idx < last_idx:
    #         dist_curr = math.hypot(self.ref_trajectory[self.ref_wp_idx, 0] - robot_xy[0],
    #                                self.ref_trajectory[self.ref_wp_idx, 1] - robot_xy[1])
    #         if dist_curr < self.wp_reach_radius:
    #             self.ref_wp_idx += 1
    #         else:
    #             break
    #     # 2. Crea la finestra dinamica scivolando in avanti lungo la traiettoria
    #     window = np.empty((steps + 1, 3), dtype=np.float64)
    #     # Lookahead spaziale: se i punti nel CSV sono troppo vicini, l'MPC non 
    #     # "guarderà" abbastanza lontano (dt * steps = tempo, ma i punti sono nello spazio).
    #     # Prova ad alzarlo (es. 2 o 3) se noti che il robot è troppo "miope".
    #     skip_step = 1 
    #     for k in range(steps + 1):
    #         # Calcoliamo l'indice futuro. Il min() serve a non sforare l'array quando 
    #         # arriviamo verso la fine della traiettoria globale.
    #         future_idx = min(self.ref_wp_idx + (k * skip_step), last_idx)
    #         window[k, :] = self.ref_trajectory[future_idx, :]
    #     return window
    
    def mpc_opt(self, obstacle_trees, ref_window, lb, ub, x0, steps=10):
        """
        NMPC di inseguimento traiettoria.
        `ref_window` è un array (steps+1, 3) di punti [x, y, theta] da
        inseguire in successione (vedi get_reference_window): NON è
        sincronizzato col tempo, è semplicemente la sequenza di waypoint
        futuri a partire da quello attualmente inseguito.
        """
        opti = ca.Opti()
        F_ = self.kin_model(self.dt)

        X = opti.variable(self.n_state, steps + 1)
        U = opti.variable(self.n_control, steps)

        # OBSTACLE occupa 3 spazi (x, y, theta) per ostacolo, REF_TRAJ occupa 3 spazi per ogni punto dell'orizzonte
        param_size = self.n_state + self.NUM_OBSTACLE_TREES * 3 + 3 * (steps + 1) + (self.NUM_POLES * 2) + (self.NUM_FLOWERBEDS * 5)
        P0 = opti.parameter(param_size)

        p_idx = 0
        X0 = P0[p_idx : p_idx + self.n_state]; p_idx += self.n_state
        # Ostacoli (le altre auto)
        OBSTACLE_TREES_param = P0[p_idx : p_idx + self.NUM_OBSTACLE_TREES*3].reshape((self.NUM_OBSTACLE_TREES, 3)).T
        p_idx += self.NUM_OBSTACLE_TREES * 3
        # Traiettoria di riferimento da inseguire: (steps+1) punti [x, y, theta]
        REF_TRAJ_param = P0[p_idx : p_idx + 3*(steps + 1)].reshape((steps + 1, 3)).T
        p_idx += 3 * (steps + 1)
        POLES_param = P0[p_idx : p_idx + self.NUM_POLES*2].reshape((self.NUM_POLES, 2)).T 
        p_idx += self.NUM_POLES * 2
        FLOWERBEDS_param = P0[p_idx : p_idx + self.NUM_FLOWERBEDS*5].reshape((self.NUM_FLOWERBEDS, 5)).T 
        p_idx += self.NUM_FLOWERBEDS * 5

        # Configurazione pesi della funzione di costo dell'MPC
        R_v = 1e-3
        R_omega = 1e-3
        Q_track_pos = 8.0     # Peso di inseguimento sulla posizione (X, Y)
        Q_track_ori = 0.5     # Peso di inseguimento sull'orientamento (Theta)
        Q_track_pos_terminal = 40.0   # Peso extra sull'ultimo punto della finestra (convergenza al waypoint)
        Q_track_ori_terminal = 15.0
        obj = 0

        opti.subject_to(X[:, 0] == X0)

        for i in range(steps):
            opti.subject_to(opti.bounded(lb[0] - 50.0, X[0, i], ub[0] + 50.0))
            opti.subject_to(opti.bounded(lb[1] - 50.0, X[1, i], ub[1] + 50.0))
            opti.subject_to(opti.bounded(-2*np.pi, X[2, i], 2*np.pi))

            # Limiti di attuazione motori fisici del Clearpath Husky
            opti.subject_to(opti.bounded(-0.2, U[0, i], 0.2))       # Velocità lineare massima v (m/s)
            opti.subject_to(opti.bounded(-0.15, U[1, i], 0.15))       # Velocità angolare massima omega (rad/s)
            
            opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))

            # ---------------------------------------------------------
            # EVITAMENTO PALI (Potenziale "Hard" Orientato)
            # ---------------------------------------------------------
            # Dimensioni approssimative del Clearpath Husky (in metri)
            robot_length = 1.0 
            robot_width = 0.67 
            margin_pole = 0.8 
            # Parametri del potenziale rigido (Muro Esponenziale)
            Q_pole_hard = 100.0   # Costo base altissimo al confine dell'ostacolo
            alpha_pole = 3.0      # Ripidezza estrema. Più è alto, più il "muro" è verticale
            for p in range(self.NUM_POLES):
                px, py = POLES_param[0, p], POLES_param[1, p]
                # 1. Distanza globale
                dx_p = px - X[0, i+1]
                dy_p = py - X[1, i+1]
                theta_rob = X[2, i+1]
                # 2. Proiezione del PALO nel sistema locale del ROBOT
                x_rel_p = dx_p * ca.cos(theta_rob) + dy_p * ca.sin(theta_rob)
                y_rel_p = -dx_p * ca.sin(theta_rob) + dy_p * ca.cos(theta_rob)
                # 3. Semiassi dell'area di ingombro
                sigma_x_rob = (robot_length / 2.0) + margin_pole
                sigma_y_rob = (robot_width / 2.0) + margin_pole
                # 4. Metrica di distanza ellittica (1.0 = bordo esatto del margine)
                dist_norm_p = (x_rel_p / sigma_x_rob)**2 + (y_rel_p / sigma_y_rob)**2
                # 5. Potenziale Hard: ca.exp( alpha * (1 - distanza) )
                # - Se dist_norm_p = 1.0 (sul bordo), il costo è Q_pole_hard
                # - Se dist_norm_p < 1.0 (dentro), il costo esplode istantaneamente (es. exp(5) * 5000)
                # - Se dist_norm_p > 1.0 (fuori), il costo decade a zero quasi subito (nessuna interferenza)
                hard_potential_pole = Q_pole_hard * ca.exp(alpha_pole * (1.0 - dist_norm_p))
                # Aggiungiamo il costo all'obiettivo
                obj = obj + hard_potential_pole
            # ---------------------------------------------------------
            # EVITAMENTO AIUOLE (Barriera Esponenziale + Hard Constraint)
            # ---------------------------------------------------------
            margin_f = 1.0  
            Q_flowerbed_hard = 50.0 # Costo base sul perimetro
            alpha_flowerbed = 2.0    # Ripidezza della barriera. Più è alto, più spinge fuori.
            for f in range(self.NUM_FLOWERBEDS):
                fx = FLOWERBEDS_param[0, f]
                fy = FLOWERBEDS_param[1, f]
                flen = FLOWERBEDS_param[2, f]
                fwid = FLOWERBEDS_param[3, f]
                ftheta = FLOWERBEDS_param[4, f]
                dx_f = X[0, i+1] - fx
                dy_f = X[1, i+1] - fy
                # Proiezione nel sistema di riferimento dell'aiuola
                x_rel_f = dx_f * ca.cos(ftheta) + dy_f * ca.sin(ftheta)
                y_rel_f = -dx_f * ca.sin(ftheta) + dy_f * ca.cos(ftheta)
                sigma_x_f = (flen / 2.0) + margin_f
                sigma_y_f = (fwid / 2.0) + margin_f
                # Metrica superellittica (esponente 4 per forma più rettangolare)
                dist_norm_f = (x_rel_f / sigma_x_f)**4 + (y_rel_f / sigma_y_f)**4
                # 1. SOFT CONSTRAINT FORTE (Barriera Esponenziale)
                # Se dist_norm_f < 1 (dentro l'aiuola), l'esponente diventa positivo e il costo esplode.
                # Se dist_norm_f > 1 (fuori dall'aiuola), l'esponente diventa negativo e il costo decade.
                repulsive_flowerbed = Q_flowerbed_hard * ca.exp(alpha_flowerbed * (1.0 - dist_norm_f))
                obj = obj + repulsive_flowerbed
                # 2. HARD CONSTRAINT
                # Forza il solutore a non considerare mai stati all'interno dell'aiuola.
                # Se IPOPT fatica a trovare soluzioni, puoi abbassare leggermente a 0.8 per renderlo meno rigido.
                opti.subject_to(dist_norm_f >= 0.9)
            

            # Prevenzione delle collisioni
            # for j in range(self.NUM_OBSTACLE_TREES):
            #     obs_j_pos = OBSTACLE_TREES_param[:, j]
            #     dist_sq_obs = ca.sumsqr(X[:2, i+1] - obs_j_pos[:2])
            #     opti.subject_to(dist_sq_obs >= safe_distance**2)
            # ------------------------------------------------------------------
            # EVITAMENTO OSTACOLI AVANZATO (Repulsione + Circumnavigazione Tangenziale)
            # ------------------------------------------------------------------
            # Dimensioni dell'auto
            car_length = 4# 3.46 
            car_width = 2# 1.62   
            # Intensità della forza repulsiva
            A_rep = 50     
            # Margini di sicurezza per l'APF (non sono hard constraints)
            margin_x = 1.2    
            margin_y = 1.2                
            # Parametri della circumnavigazione tangenziale (forza scivolante)
            # Questo peso controlla quanto aggressivamente il robot sterza per circumnavigare
            Q_circ_ori = 200.0 # Regola questo parametro per ottenere la manovra fluida
            # Direzione della circumnavigazione rispetto al diagramma di `untitled.png`
            # 1 per circumnavigare a destra dell'auto (per chi la guarda dall'alto)
            # -1 per circumnavigare a sinistra dell'auto
            direction_of_circ = 1 
            for j in range(self.NUM_OBSTACLE_TREES):
                obs_j_pos = OBSTACLE_TREES_param[:, j]
                car_x, car_y, car_theta = obs_j_pos[0], obs_j_pos[1], obs_j_pos[2]
                # Centro dell'auto (traslato indietro di L/2 rispetto alla punta del muso)
                center_x = car_x - (car_length / 2.0) * ca.cos(car_theta)
                center_y = car_y - (car_length / 2.0) * ca.sin(car_theta)
                dx = X[0, i+1] - center_x
                dy = X[1, i+1] - center_y
                # Proiezione nel sistema locale dell'auto
                x_rel = dx * ca.cos(car_theta) + dy * ca.sin(car_theta)
                y_rel = -dx * ca.sin(car_theta) + dy * ca.cos(car_theta)
                sigma_x = (car_length / 2.0) + margin_x
                sigma_y = (car_width / 2.0) + margin_y
                # 1. Base per la Gaussiana (Ovale perfetto, esponente 2)
                dist_norm = (x_rel / sigma_x)**2 + (y_rel / sigma_y)**2
                # Soft Constraint 1: Campo Potenziale Repulsivo (Gaussiana 2D)
                # alpha_flowerbed agisce qui come fattore di forma della campana (la "varianza")
                repulsive_cost = A_rep * ca.exp(alpha_flowerbed * (1.0 - dist_norm))
                # --- INIZIO CIRCUMNAVIGAZIONE ---
                # --- 1. Calcolo dell'orientamento tangenziale desiderato ---
                v_radial = ca.vcat([x_rel, y_rel])
                u_radial = v_radial / (ca.norm_2(v_radial) + 1e-6)
                if direction_of_circ == 1:
                    u_tangential = ca.vcat([-u_radial[1], u_radial[0]])
                else:
                    u_tangential = ca.vcat([u_radial[1], -u_radial[0]])
                angle_tangential_ref_car_frame = ca.atan2(u_tangential[1], u_tangential[0])
                # --- 2. Allineamento dell'orientamento del robot ---
                theta_fut = X[2, i+1]
                theta_fut_car_frame = theta_fut - car_theta
                angle_diff = angle_tangential_ref_car_frame - theta_fut_car_frame
                angle_diff_norm = ca.atan2(ca.sin(angle_diff), ca.cos(angle_diff))
                # Soft Constraint 2: Costo di circumnavigazione tangenziale
                # Anche qui usiamo la forma gaussiana per far decadere la forza dolcemente
                cost_circ = Q_circ_ori * ca.exp(-dist_norm) * angle_diff_norm**2
                # --- FINE CODICE CIRCUMNAVIGAZIONE ---
                # Aggiunta dei costi al costo globale
                obj = obj + repulsive_cost #+ cost_circ
                # Hard Constraint di sicurezza assoluta (80% dell'ovale)
                core_safety = 0.8  
                # ATTENZIONE: Usa l'esponente 2 (o nessuno se elevi il core_safety a 2)
                # perché dist_norm ora è calcolata al quadrato, non alla quarta!
                opti.subject_to( dist_norm >= core_safety**2 )

            # ---------------------------------------------------------
            # INSEGUIMENTO TRAIETTORIA DI RIFERIMENTO
            # ---------------------------------------------------------
            # Punto di riferimento associato al passo i+1 dell'orizzonte
            x_ref = REF_TRAJ_param[0, i+1]
            y_ref = REF_TRAJ_param[1, i+1]
            theta_ref = REF_TRAJ_param[2, i+1]

            theta_fut = X[2, i+1]
            theta_err_raw = theta_ref - theta_fut
            theta_err = ca.atan2(ca.sin(theta_err_raw), ca.cos(theta_err_raw))

            tracking_cost = Q_track_pos * ((X[0, i+1] - x_ref)**2 + (X[1, i+1] - y_ref)**2) \
                            + Q_track_ori * theta_err**2
            obj = obj + tracking_cost

            obj = obj + R_v * (U[0, i]**2) + R_omega * (U[1, i]**2)

        # ---------------------------------------------------------
        # Costo terminale: convergenza pesante sull'ultimo waypoint della finestra
        # ---------------------------------------------------------
        x_ref_N = REF_TRAJ_param[0, steps]
        y_ref_N = REF_TRAJ_param[1, steps]
        theta_ref_N = REF_TRAJ_param[2, steps]
        theta_err_N_raw = theta_ref_N - X[2, steps]
        theta_err_N = ca.atan2(ca.sin(theta_err_N_raw), ca.cos(theta_err_N_raw))

        obj = obj + Q_track_pos_terminal * ((X[0, steps] - x_ref_N)**2 + (X[1, steps] - y_ref_N)**2) \
                  + Q_track_ori_terminal * theta_err_N**2

        opti.minimize(obj)
        
        # options = {
        #     "ipopt": {
        #         "tol": 1e-5,
        #         "warm_start_init_point": "yes",
        #         "print_level": 0,
        #         "sb": "no",
        #         "hessian_approximation": 'limited-memory',
        #         "max_iter": 1000,
        #     }
        # }
        options = {
            "ipopt": {
                "tol": 1e-4,                      # Rilassa la tolleranza generale
                "acceptable_tol": 1e-3,           # Accetta soluzioni meno precise...
                "acceptable_iter": 5,             # ...se si mantengono stabili per 5 iterazioni
                "max_iter": 100,                   # Tassativo: ferma il solutore per evitare di sforare il dt
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
            ca.reshape(obstacle_trees, 3 * self.NUM_OBSTACLE_TREES, 1),
            ca.reshape(ca.DM(ref_window), 3 * (steps + 1), 1),
            ca.reshape(self.poles_pos, 2 * self.NUM_POLES, 1),
            ca.reshape(self.flowerbeds_pos, 5 * self.NUM_FLOWERBEDS, 1)
        )
        opti.set_value(P0, p0_val)

        opti.set_initial(X, ca.repmat(x0, 1, steps + 1))
        opti.set_initial(U, ca.DM.zeros(self.n_control, steps))

        sol = opti.solve()
        mpc_step_func = opti.to_function("mpc_step", inputs, outputs, ["p", "x_init", "x_lam"], ["u_opt", "x_pred", "x_opt", "lam_opt"])

        return (mpc_step_func, ca.DM(sol.value(U[:, 0])), ca.DM(sol.value(X)), ca.DM(sol.value(opti.x)), ca.DM(sol.value(opti.lam_g)))

    def run_simulation(self):
        lb, ub = self.get_domain(self.trees_pos)
        
        rospy.loginfo("In attesa di dati validi dal topic di odometria...")
        while self.current_state is None and not rospy.is_shutdown():
            rospy.sleep(0.05)
        rospy.loginfo("Modulo di localizzazione agganciato.")

        x_k = ca.DM(self.current_state)

        all_trajectories = []
        lambda_history, entropy_history, durations = [], [], []
        velocity_command_log, pose_history, time_history = [], [], []

        sim_start_time = time.time()
        total_distance = 0.0
        total_commands = 0
        prev_x, prev_y = float(x_k[0]), float(x_k[1])

        sim_time = 12000000
        mpciter = 0
        rate = rospy.Rate(int(1/self.dt))
        warm_start = True
        x_dec_prev, lam_g_prev, mpc_step = None, None, None

        try:
            # while mpciter < sim_time and not rospy.is_shutdown():
            while not rospy.is_shutdown():
                loop_iter_start = time.time()
                rospy.loginfo('Control Loop Step: %d', mpciter)
                
                state_now = self.current_state
                current_sim_time = time.time() - sim_start_time
                pose_history.append(state_now)
                time_history.append(current_sim_time)
                x_k = ca.DM(state_now)

                # Sincronizzazione bloccante per attendere i dati dei classificatori visivi degli alberi
                while self.latest_trees_scores is None and not rospy.is_shutdown():
                    rospy.sleep(0.01)
                
                scores = self.latest_trees_scores.copy()
                if mpciter % 2 == 0: 
                    self.beliefs_k = self.bayes(self.beliefs_k, ca.DM(scores))

                # Invio dei marker geometrici per la visualizzazione grafica (Rviz)
                tree_markers_msg = create_tree_markers(self.trees_pos, self.beliefs_k.full())
                self.tree_markers_pub.publish(tree_markers_msg)
                
                robot_position_xy = np.array(state_now[:2])
                obstacle_indices = self.get_nearest_tree_indices(robot_position_xy, num_obstacle=self.NUM_OBSTACLE_TREES)
                obstacle_trees_subset = self.trees_pos[obstacle_indices]

                # Finestra di traiettoria di riferimento da inseguire per l'orizzonte corrente
                # (avanzamento sui waypoint per prossimità, non sincronizzato col tempo originale)
                ref_window = self.get_reference_window(robot_position_xy, self.N)

                step_start_time = time.time()
                try:
                    if warm_start or mpc_step is None:
                        mpc_step, u, x_traj, x_dec_prev, lam_g_prev = self.mpc_opt(
                            obstacle_trees_subset, ref_window, lb, ub, x_k, steps=self.N
                        )
                        warm_start = False
                    else:
                        P0_val = ca.vertcat(
                            x_k,
                            ca.reshape(obstacle_trees_subset, 3 * self.NUM_OBSTACLE_TREES, 1),
                            ca.reshape(ca.DM(ref_window), 3 * (self.N + 1), 1),
                            ca.reshape(self.poles_pos, 2 * self.NUM_POLES, 1),
                            ca.reshape(self.flowerbeds_pos, 5 * self.NUM_FLOWERBEDS, 1)
                        )
                        u, x_traj, x_dec_prev, lam_g_prev = mpc_step(P0_val, x_dec_prev, lam_g_prev)
                    step_duration = time.time() - step_start_time
                    durations.append(step_duration)

                except Exception as e:
                    rospy.logerr(f"Eccezione fatale durante la risoluzione MPC al ciclo {mpciter}: {e}")
                    # Mandiamo un comando di stop immediato per sicurezza se fallisce il solutore
                    self.cmd_vel_pub.publish(Twist())
                    return

                v_cmd = float(u[0])
                omega_cmd = float(u[1])

                # INVIO COMANDI CINEMATICI DIRETTAMENTE ALL'HUSKY
                twist_msg = Twist()
                twist_msg.linear.x = v_cmd
                twist_msg.angular.z = omega_cmd
                self.cmd_vel_pub.publish(twist_msg)

                # Pubblicazione del cammino pianificato predittivo
                predicted_path_msg = create_path_from_mpc_prediction(x_traj[:self.nx, 1:])
                self.pred_path_pub.publish(predicted_path_msg)

                velocity_command_log.append([current_sim_time, "MPC", v_cmd, omega_cmd])
                total_commands += 1

                curr_x, curr_y = float(x_traj[0, 1]), float(x_traj[1, 1])
                total_distance += math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                prev_x, prev_y = curr_x, curr_y

                entropy_k = self.entropy_entire_field(self.beliefs_k)
                lambda_history.append(self.beliefs_k.full().flatten().tolist())
                entropy_history.append(ca.sum1(entropy_k).full().flatten()[0])
                all_trajectories.append(x_traj[:self.nx, :].full())

                mpciter += 1
                rospy.loginfo("Entropia globale del sistema: %s", entropy_history[-1])
                
                if all(v <= self.threshold_entropy for v in entropy_k.full().flatten()):
                    rospy.loginfo("Target informativo di riduzione entropia completato.")
                    # break

                loop_elapsed = time.time() - loop_iter_start
                sleep_time = self.dt - loop_elapsed
                if sleep_time > 0:
                    rate.sleep()
                else:
                    rospy.logwarn(f"Frequenza di ciclo violata! L'ottimizzazione ha impiegato {loop_elapsed:.4f}s rispetto al limite di {self.dt}s")

            # Arresto di sicurezza assoluto a fine loop
            self.cmd_vel_pub.publish(Twist())

        except rospy.ROSInterruptException:
            rospy.loginfo("Rilevato Ctrl+C. Interruzione del loop e avvio del salvataggio dati.")
        
        finally:
            rospy.loginfo("Arresto del robot e salvataggio in formato CSV...")
            # Arresto di sicurezza assoluto a fine loop
            self.cmd_vel_pub.publish(Twist())

            # Salvataggio asincrono dei dati telemetrici in formato CSV per l'analisi post-esperimento
            self.save_performance_data(sim_start_time, total_distance, total_commands, entropy_history, time_history, pose_history, lambda_history, velocity_command_log)

        return all_trajectories, entropy_history, lambda_history, durations, self.l4c_nn, self.trees_pos, lb, ub

    def save_performance_data(self, start_time, dist, cmds, entropy_hist, time_hist, pose_hist, lambda_hist, vel_log):
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
                
        rospy.loginfo(f"Report delle prestazioni salvato correttamente in: {self.baselines_dir}")


if __name__ == "__main__":
    try:
        mpc = NeuralMPCHusky()
        mpc.run_simulation()
    except rospy.ROSInterruptException:
        pass