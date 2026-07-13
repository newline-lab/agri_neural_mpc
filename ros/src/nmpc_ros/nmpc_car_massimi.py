#!/usr/bin/env python
import os
import re
import time
import csv
import math
import threading

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
    def __init__(self, run_dir=None):
        self.hidden_size = 64
        self.hidden_layers = 3
        self.nn_input_dim = 3

        self.N = 10
        self.dt = 0.2  # Controllo a 5 Hz
        self.T = self.dt * self.N
        
        # Specifiche del robot reale (Uniciclo)
        self.nx = 3          # Stato: [x, y, theta]
        self.n_state = 3
        self.n_control = 2   # Ingressi di controllo: [v, omega]
        
        self.NUM_TARGET_TREES = 1   # subset alberi vicini da esplorare
        self.NUM_OBSTACLE_TREES = 2 # subset alberi vicini da evitare

        self.threshold_entropy = 0.15 # Quando albero considerato visto

        # ----------------------------------------------------------------------
        # Massimi locali
        # ----------------------------------------------------------------------
        self.punti_soglia = [
            [-1.3037, -2.5762, -2.0944],
            [-0.934, -2.3745, -1.5708],
            [3.2101, 0.1273, -0.0],
            [2.9268, 0.569, 0.5236],
            [0.1753, 3.5078, 1.0472],
            [-1.1693, 3.066, 1.5708],
            [-0.4298, 2.9508, 2.0944],
            [-1.5582, 3.2533, 2.618],
        ]

        # ----------------------------------------------------------------------
        # COORDINATE HARDCODED DEGLI ALBERI (Origine coincidente con lo zero dell'Odom)
        # ----------------------------------------------------------------------
        self.trees_pos = np.array([
            [-4.0, -1.0, 0.0],
            # [-4.0, 5.0, 0.0],
            # [-3.5, 10.0, 0.0],
            [4.0, 15.0, -np.pi]
        ], dtype=np.float32)
        
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
        rospy.Subscriber("tree_scores", Float32MultiArray, self.tree_scores_callback)
        
        # Pubblicazioni per l'hardware e monitoraggio (Rviz)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pred_path_pub = rospy.Publisher("predicted_path", Path, queue_size=10)
        self.tree_markers_pub = rospy.Publisher("tree_markers", MarkerArray, queue_size=10)

        self.baselines_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../baselines") if run_dir is None else run_dir

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
    

    def get_closest_threshold_state(self, robot_state, target_trees):
        """
        Trova la configurazione relativa ottima [x_rel, y_rel, azimuth] dai punti soglia.
        Invece di lavorare nel frame globale, calcola lo stato relativo attuale del robot
        e sceglie il picco più vicino nello spazio relativo (che coincide con l'input della rete).
        """
        rx, ry, rtheta = robot_state[0], robot_state[1], robot_state[2]
        best_cost = float('inf')
        best_peak = [0.0, 0.0, 0.0]

        # W_theta pesa l'importanza della rotazione rispetto alla distanza.
        W_theta = 1.2 

        for tree in target_trees:
            tx, ty, theta_target = tree[0], tree[1], tree[2]
            
            # --- 1. STATO RELATIVO ATTUALE DEL ROBOT ---
            dX = rx - tx
            dY = ry - ty
            
            # Proiezione della posizione nel frame della macchina
            curr_x_rel = dX * math.cos(theta_target) + dY * math.sin(theta_target)
            curr_y_rel = -dX * math.sin(theta_target) + dY * math.cos(theta_target)
            
            # Calcolo Azimut attuale (0 se asse y robot e asse x macchina si guardano)
            curr_theta_y = rtheta + (math.pi / 2.0)
            curr_azimuth_raw = curr_theta_y - theta_target + math.pi
            curr_azimuth = math.atan2(math.sin(curr_azimuth_raw), math.cos(curr_azimuth_raw))
            
            # --- 2. RICERCA DEL PICCO OTTIMO NELLO SPAZIO RELATIVO ---
            for p in self.punti_soglia:
                # p è [x_rel_opt, y_rel_opt, azimuth_opt] estratti dall'analisi della rete
                p_x_rel, p_y_rel, p_azimuth = p[0], p[1], p[2]
                
                # Distanza puramente nel piano relativo
                dist_geometrica = math.hypot(curr_x_rel - p_x_rel, curr_y_rel - p_y_rel)
                
                # Sforzo di rotazione sull'azimut
                delta_azimuth = p_azimuth - curr_azimuth
                delta_azimuth_norm = math.atan2(math.sin(delta_azimuth), math.cos(delta_azimuth))
                sforzo_rotazione = abs(delta_azimuth_norm)
                
                costo_totale = dist_geometrica + (W_theta * sforzo_rotazione)
                
                if costo_totale < best_cost:
                    best_cost = costo_totale
                    # Passiamo direttamente il target in coordinate RELATIVE
                    best_peak = [p_x_rel, p_y_rel, p_azimuth]

        return best_peak

    def mpc_opt(self, target_trees, target_lambdas, obstacle_trees, closest_thresh, lb, ub, x0, steps=10):
        opti = ca.Opti()
        F_ = self.kin_model(self.dt)

        X = opti.variable(self.n_state, steps + 1)
        U = opti.variable(self.n_control, steps)

        # TARGET_TREES occupa 3 spazi (x, y, theta), L0 occupa 2 spazi, OBSTACLE occupa 3 spazi + 3 per attrazione
        param_size = self.n_state + self.NUM_TARGET_TREES * 5 + self.NUM_OBSTACLE_TREES * 3 + 3
        P0 = opti.parameter(param_size)

        p_idx = 0
        X0 = P0[p_idx : p_idx + self.n_state]; p_idx += self.n_state
        # Cambia *2 in *3 per i tree param
        TARGET_TREES_param = P0[p_idx : p_idx + self.NUM_TARGET_TREES*3].reshape((self.NUM_TARGET_TREES, 3)).T; p_idx += self.NUM_TARGET_TREES*3
        # I lambda rimangono a dimensione 2
        L0 = P0[p_idx : p_idx + self.NUM_TARGET_TREES*2].reshape((self.NUM_TARGET_TREES, 2)); p_idx += self.NUM_TARGET_TREES * 2
        # Cambia *2 in *3 per gli ostacoli
        OBSTACLE_TREES_param = P0[p_idx : p_idx + self.NUM_OBSTACLE_TREES*3].reshape((self.NUM_OBSTACLE_TREES, 3)).T        
        p_idx += self.NUM_OBSTACLE_TREES * 3
        # AGGIUNGI QUESTE DUE RIGHE per estrarre il punto
        OPT_THRESH_param = P0[p_idx : p_idx + 3]
        lambda_evol = [L0]

        # Configurazione pesi della funzione di costo dell'MPC
        Q_dist = 1e-5
        R_v = 1e-5
        R_omega = 1e-5
        attraction = 0
        safe_distance = 0  # Distanza di sicurezza dagli alberi-ostacolo (metri)
        entropy_w = 40
        obj = 0

        opti.subject_to(X[:, 0] == X0)
        ca_batch = []

        for i in range(steps):
            opti.subject_to(opti.bounded(lb[0] - 20.0, X[0, i], ub[0] + 20.0))
            opti.subject_to(opti.bounded(lb[1] - 20.0, X[1, i], ub[1] + 20.0))
            opti.subject_to(opti.bounded(-2*np.pi, X[2, i], 2*np.pi))

            # Limiti di attuazione motori fisici del Clearpath Husky
            opti.subject_to(opti.bounded(-0.5, U[0, i], 0.5))       # Velocità lineare massima v (m/s)
            opti.subject_to(opti.bounded(-1.0, U[1, i], 1.0))       # Velocità angolare massima omega (rad/s)
            
            opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))

            # Prevenzione delle collisioni
            for j in range(self.NUM_OBSTACLE_TREES):
                obs_j_pos = OBSTACLE_TREES_param[:, j]
                dist_sq_obs = ca.sumsqr(X[:2, i+1] - obs_j_pos[:2])
                opti.subject_to(dist_sq_obs >= safe_distance**2)


            theta_fut = X[2, i+1]  # Orientamento futuro del ROBOT (asse X)
            distances_sq = []
            nn_batch = []
            side_observation_cost = 0
            for j in range(self.NUM_TARGET_TREES):
                obj_j_pos = TARGET_TREES_param[:, j]
                theta_target = obj_j_pos[2]
                
                # Vettore differenza globale (Macchina - Centro Robot)
                dX = obj_j_pos[0] - X[0, i+1]
                dY = obj_j_pos[1] - X[1, i+1]
                # DALLA macchina AL robot (posizione del robot relativa alla macchina)
                dX, dY = -dX, -dY

                dist_sq = dX**2 + dY**2 + 1e-6
                distances_sq.append(dist_sq)
                # Proiezione nel sistema di riferimento LOCALE del ROBOT
                # # Asse X del robot = avanti, Asse Y = sinistra
                # x_rel = dX * ca.cos(theta_fut) + dY * ca.sin(theta_fut)
                # y_rel = -dX * ca.sin(theta_fut) + dY * ca.cos(theta_fut)
                # DALLA macchina AL robot (posizione del robot relativa alla macchina)
                x_rel = dX * ca.cos(theta_target) + dY * ca.sin(theta_target)
                y_rel = -dX * ca.sin(theta_target) + dY * ca.cos(theta_target)


                # Calcolo Azimuth
                theta_y_robot = theta_fut + (ca.pi / 2.0)
                azimuth_raw = theta_y_robot - theta_target + ca.pi
                azimuth_norm = ca.atan2(ca.sin(azimuth_raw), ca.cos(azimuth_raw))
                # rete [dx, dy, azimuth]
                nn_input = ca.horzcat(x_rel, y_rel, azimuth_norm)
                nn_batch.append(nn_input)

                ### MASSIMI
                # 1. Distanza quadratica dalla posizione del punto soglia ottimale (già presente)
                dist_to_thresh_sq = (x_rel - OPT_THRESH_param[0])**2 + (y_rel - OPT_THRESH_param[1])**2
                # 2. Calcolo dell'errore di orientamento rispetto al punto soglia
                # theta_fut è l'orientamento del robot al passo i+1, OPT_THRESH_param[2] è il theta desiderato
                angle_diff_raw = OPT_THRESH_param[2] - azimuth_norm
                # Normalizzazione dell'errore angolare tra -pi e +pi usando CasADi
                angle_diff_norm = ca.atan2(ca.sin(angle_diff_raw), ca.cos(angle_diff_raw))
                angle_error_sq = angle_diff_norm**2
                # 3. Pesi della funzione obiettivo (DA TARARE)
                Q_thresh_pos = 2.0   # Peso di attrazione sulla posizione (X, Y)
                Q_thresh_ori = 1.0   # Peso per l'orientamento (Theta). 
                # 4. Aggiornamento della funzione obiettivo complessiva
                obj = obj + Q_thresh_pos * dist_to_thresh_sq + Q_thresh_ori * angle_error_sq

            ca_batch.append(ca.vcat([*nn_batch]))
                        
            min_dist_sq = distances_sq[0]
            for j in range(1, self.NUM_TARGET_TREES):
                min_dist_sq = ca.fmin(min_dist_sq, distances_sq[j])
            attraction = attraction + min_dist_sq * Q_dist
            
            obj = obj + R_v * (U[0, i]**2) + R_omega * (U[1, i]**2)
            

        # Inferenza batched con L4CasADi
        nn_full_batch_input = ca.vcat(ca_batch)
        surrogate_output_ripe = self.l4c_nn[0](nn_full_batch_input)
        surrogate_output_raw = self.l4c_nn[1](nn_full_batch_input)

        # Adattamento output 1D: applicazione Sigmoide per mappare logit -> probabilità [p, 1-p]
        prob_ripe = 1.0 / (1.0 + ca.exp(-surrogate_output_ripe))
        prob_raw = 1.0 / (1.0 + ca.exp(-surrogate_output_raw))

        L0_ext = ca.vcat([L0 for _ in range(steps)])
        sel = L0_ext[:, 0] >= L0_ext[:, 1]
        p_selected = ca.if_else(sel, prob_ripe, prob_raw)
        
        # Rigenerazione del vettore di likelihood bidimensionale per il calcolo Bayesiano
        # z_k_bin = ca.horzcat(p_selected, 1.0 - p_selected)
        # Mapping da [0, 1] a [0.5, 1] nel grafo CasADi ---
        p_mapped = 0.5 + 0.5 * p_selected
        # Rigenerazione del vettore di likelihood bidimensionale per il calcolo Bayesiano
        z_k_bin = ca.horzcat(p_mapped, 1.0 - p_mapped)
        
        for i in range(steps):
            lambda_next = self.bayes(lambda_evol[-1], z_k_bin[i*self.NUM_TARGET_TREES:(i+1)*self.NUM_TARGET_TREES,:])
            lambda_evol.append(lambda_next)
            
        entropy_obj = 0
        for i in range(1, steps+1):
            entropy_future = self.entropy_target(lambda_evol[i])
            entropy_obj += ca.exp(-2*i)*ca.logsumexp(-entropy_w*entropy_future)

        sq_dist_to_targets = ca.sum1((X0[:2] - TARGET_TREES_param[:2, :])**2)
        min_sq_dist = ca.mmin(sq_dist_to_targets)

        threshold_sq_dist = 15.0
        sigmoid_steepness = 0.5
        sigmoid_factor = 1.0 / (1.0 + ca.exp(-sigmoid_steepness * (min_sq_dist - threshold_sq_dist)))
        modulated_attraction_term = attraction * sigmoid_factor

        modulated_side_cost = side_observation_cost * sigmoid_factor

        opti.minimize(obj - 1*entropy_obj + 0*modulated_attraction_term + 0*modulated_side_cost)                                 
        
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
            ca.reshape(target_trees, 3 * self.NUM_TARGET_TREES, 1),
            ca.reshape(target_lambdas, 2 * self.NUM_TARGET_TREES, 1),
            ca.reshape(obstacle_trees, 3 * self.NUM_OBSTACLE_TREES, 1),
            ca.DM(closest_thresh)
        )
        opti.set_value(P0, p0_val)

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

        sim_time = 1200
        mpciter = 0
        rate = rospy.Rate(int(1/self.dt))
        warm_start = True
        x_dec_prev, lam_g_prev, mpc_step = None, None, None

        while mpciter < sim_time and not rospy.is_shutdown():
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
            target_indices = self.get_target_tree_indices(robot_position_xy, num_target=self.NUM_TARGET_TREES)
            obstacle_indices = self.get_nearest_tree_indices(robot_position_xy, num_obstacle=self.NUM_OBSTACLE_TREES)

            target_trees_subset = self.trees_pos[target_indices]
            obstacle_trees_subset = self.trees_pos[obstacle_indices]
            target_lambdas = self.beliefs_k[target_indices, :]

            closest_thresh_state = self.get_closest_threshold_state(state_now, target_trees_subset)

            step_start_time = time.time()
            try:
                if warm_start or mpc_step is None:
                    # AGGIUNGI closest_thresh_state AI PARAMETRI
                    mpc_step, u, x_traj, x_dec_prev, lam_g_prev = self.mpc_opt(
                        target_trees_subset, target_lambdas, obstacle_trees_subset, closest_thresh_state, lb, ub, x_k, steps=self.N
                    )
                    warm_start = False
                else:
                    P0_val = ca.vertcat(
                        x_k,
                        ca.reshape(target_trees_subset, 3 * self.NUM_TARGET_TREES, 1),
                        ca.reshape(target_lambdas, 2 * self.NUM_TARGET_TREES, 1),
                        ca.reshape(obstacle_trees_subset, 3 * self.NUM_OBSTACLE_TREES, 1),
                        ca.DM(closest_thresh_state) # <--- AGGIUNGI IN CODA
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
                break

            loop_elapsed = time.time() - loop_iter_start
            sleep_time = self.dt - loop_elapsed
            if sleep_time > 0:
                rate.sleep()
            else:
                 rospy.logwarn(f"Frequenza di ciclo violata! L'ottimizzazione ha impiegato {loop_elapsed:.4f}s rispetto al limite di {self.dt}s")

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