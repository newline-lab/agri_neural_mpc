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
import tf2_ros
from scipy.stats import norm
from scipy.sparse.csgraph import minimum_spanning_tree

import l4casadi as l4c
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ROS message imports
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Float32MultiArray, Int32MultiArray, Float64MultiArray, Int32
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Path
import tf

# Service import for tree poses (as in sensors.py)
from nmpc_ros.srv import GetTreesPoses

# Import helper functions from sensors.py
from nmpc_ros_package.ros_com_lib.sensors import create_path_from_mpc_prediction, create_tree_markers

# Custom message for predictions
from nmpc_ros.msg import Trajectory, MultiTraj
from std_msgs.msg import Header

# ---------------------------
# Simple Neural Network
# ---------------------------
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

# ---------------------------
# Neural MPC Class (Modified Version)
# ---------------------------
class NeuralMPC:
    def __init__(self):
        # Global Constants and Parameters
        self.hidden_size = 64
        self.hidden_layers = 3
        self.nn_input_dim = 3

        self.N = 5
        self.dt = 0.5
        self.T = self.dt * self.N
        self.nx = 3  # Represents [x, y, theta]

        # For storing the latest tree scores from the sensor callback.
        self.latest_trees_scores = None
        # For storing the latest robot state from the gps callback.
        self.current_state = None

        rospy.init_node("nmpc_node", anonymous=True, log_level=rospy.DEBUG)
        rospy.on_shutdown(self.save_all)

        # agent number
        self.n_agent = rospy.get_param('~n_agent', 1) # default 1
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Variable to hold the current robot state [x, y, yaw]
        self.current_state = None

        # Start the robot state update thread at 30 Hz.
        self.robot_state_thread = threading.Thread(target=self.robot_state_update_thread)
        self.robot_state_thread.daemon = True
        self.robot_state_thread.start()
        # Subscribers
        # tree scores
        rospy.Subscriber("tree_scores", Float32MultiArray, self.tree_scores_callback)
        # assigned trees
        rospy.Subscriber("cluster", Int32MultiArray, self.assignment_callback)
        # Publishers
        self.cmd_pose_pub = rospy.Publisher("cmd/pose", Pose, queue_size=1)
        self.pred_path_pub = rospy.Publisher("predicted_path", Path, queue_size=1)
        self.tree_markers_pub = rospy.Publisher("tree_markers", MarkerArray, queue_size=1)
        # send lambda values
        self.lambda_pub = rospy.Publisher('lambda', Float64MultiArray, queue_size=1)

        # Get tree positions from service using the sensors.py serializer logic.
        self.trees_pos = self.get_trees_poses()
        self.lambda_k = ca.DM.ones(self.trees_pos.shape[0], 1) * 0.5

        # robot positions
        rospy.Subscriber("/robot_states", Float64MultiArray, self.positions_callback)
        self.robot_positions = None

        # Connectivity
        self.epsilon = 0.1
        self.R = rospy.get_param('~R', 3.0)
        self.alpha_elem = rospy.get_param('~alpha_elem', 1.0)
        self.lambda2 = ca.DM.ones(1,1)
        self.beta = ca.DM.zeros(2,1)
        self.adjacency = None
        self.span_tree = None

        # Consensus protocol
        # Network connection (fully connected)
        n_robots = 3
        neighbors = list(range(1, n_robots+1))
        neighbors.remove(int(self.n_agent))
        self.neighbors_id = neighbors
        # subscibers
        subscribers_net = []
        for neighbor in neighbors:
            topic = f"/agent_{neighbor}/lambda"
            sub = rospy.Subscriber(topic, Float64MultiArray, self.consensus_lambda)
            subscribers_net.append(sub)
        # neighbors' positions
        self.neighbors_pos = ca.DM.zeros(2*(n_robots+1), 1) # id vicino e id+1 = posizione x e y in lista
        # Lambda consensus variables
        self.lambda_cons = ca.DM.ones(self.trees_pos.shape[0], 1) * 0.5 # consensus value
        self.latest_detection = np.zeros(self.trees_pos.shape[0]) # time last detected lambda
        self.aggregated_data = False # True: a consensus is performed

        # Start the neighbors state update thread at 30 Hz.
        self.neighbors_state_thread = threading.Thread(target=self.neighbors_state_update_thread)
        self.neighbors_state_thread.daemon = True
        self.neighbors_state_thread.start()

        # MPC horizon (number of steps)
        self.mpc_horizon = self.N
 
        # List of assigned trees (ID)
        self.assigned = None

        # Sync
        self.ok_mpc = rospy.Publisher('/mpc_ok', Trajectory, queue_size=1)
        rospy.Subscriber("/predictions", MultiTraj, self.get_predictions)
        rospy.Subscriber("/current_step", Int32, self.next_step)
        self.current_setp = 0 # sync step
        self.mpc_step = 0     # agent i step

        # Predictions
        self.traj_x = None
        self.traj_y = None

    # ---------------------------
    # Callback Functions
    # ---------------------------
    def robot_state_update_thread(self):
        """Continuously update the robot's state using TF at 30 Hz."""
        rate = rospy.Rate(30)  # 30 Hz update rate
        while not rospy.is_shutdown():
            try:
                # Look up the transform from 'map' to 'base_link_n'
                link_str = 'base_link_' + str(self.n_agent)
                trans = self.tf_buffer.lookup_transform('map', link_str, rospy.Time(0))
                # Extract the yaw angle from the quaternion
                (_, _, yaw) = tf.transformations.euler_from_quaternion([
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w
                ])
                # Update current_state with [x, y, yaw]
                self.current_state = [trans.transform.translation.x,
                                      trans.transform.translation.y,
                                      yaw]
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn("Failed to get transform: %s", e)
            rate.sleep()

    def neighbors_state_update_thread(self):
        """Continuously update the neighbors' state using TF at 30 Hz."""
        rate = rospy.Rate(30)  # 30 Hz update rate
        poss = [0.0 for _ in range(self.neighbors_pos.shape[0])]
        while not rospy.is_shutdown():
            for n in self.neighbors_id:
                try:
                    # Look up the transform from 'map' to 'base_link_n'
                    link_str = 'base_link_' + str(n)
                    trans = self.tf_buffer.lookup_transform('map', link_str, rospy.Time(0))
                    poss[2*n] = trans.transform.translation.x
                    poss[2*n+1] = trans.transform.translation.y
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn("Failed to get transform: %s", e)
            self.neighbors_pos = ca.DM(poss)
            rate.sleep()

    def get_predictions(self, msg):
        """
        Get robot preditictions
        """
        self.traj_x = [[]]  # index 0 is empty
        self.traj_y = [[]]  # index 0 is empty

        idx = 0
        for length in msg.lengths:
            self.traj_x.append(msg.traj_x[idx:idx+length])
            self.traj_y.append(msg.traj_y[idx:idx+length])
            idx += length
        # Output: self.traj_x[i] = traj_x of robot i = 1,...,N (id=0 no robots)

    def next_step(self, msg):
        """
        Perform next control
        """
        self.current_setp += 1

    def tree_scores_callback(self, msg):
        """
        Callback for tree scores.
        """
        self.latest_trees_scores = np.array(msg.data).reshape(-1, 1)

    def positions_callback(self, msg):
        """
        Callback for robot positions
        """
        self.robot_positions = np.array(msg.data)


    def assignment_callback(self, msg):
        """
        Callback for tree assignemt.
        """
        self.assigned = np.array(msg.data)

    def consensus_lambda(self, msg):
        """
        Callback for consensus lambda.
        """
        # Take data
        # neighbors = msg.data
        # lambda_curr = self.lambda_cons.full().flatten()
        # # Compute maximum/minimum
        # result_max = np.maximum(neighbors, lambda_curr)
        # result_min = np.minimum(neighbors, lambda_curr)
        # for i in range(len(result_max)):
        #     val_max = result_max[i] - 0.5
        #     val_min = 0.5 - result_min[i]
        #     if val_max > val_min:
        #         result_min[i] = result_max[i]
        # self.lambda_cons = ca.DM(result_min)

        # Get latest detections
        self.lambda_cons = self.lambda_k
        new_lambdas = msg.data[:self.trees_pos.shape[0]]
        new_times = msg.data[self.trees_pos.shape[0]:]
        for i in range(self.trees_pos.shape[0]):
            if self.latest_detection[i] < new_times[i]:
                self.lambda_cons[i] = new_lambdas[i]
                self.latest_detection[i] = new_times[i]
        self.aggregated_data = True

    # ---------------------------
    # Service Call to Get Trees Poses
    # ---------------------------
    def get_trees_poses(self):
        """
        Calls the GetTreesPoses service and returns tree positions as an (N,2) numpy array.
        The serializer logic is taken from sensors.py.
        """
        rospy.wait_for_service("/obj_pose_srv")
        try:
            trees_srv = rospy.ServiceProxy("/obj_pose_srv", GetTreesPoses)
            response = trees_srv()  # Adjust parameters if needed
            # Using the serializer from sensors.py:
            trees_pos = np.array([[pose.position.x, pose.position.y] for pose in response.trees_poses.poses])
            return trees_pos
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
            return np.array([])

    # ---------------------------
    # Utility Functions
    # ---------------------------
    def get_latest_best_model(self):
        # Get the directory where THIS script is stored.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Models are stored in a subdirectory "models/" relative to this script.
        model_dir = os.path.join(script_dir, "models")
        # Find the latest best model file matching the pattern.
        model_files = [
            f for f in os.listdir(model_dir)
            if re.match(r"best_model_epoch_(\d+)\.pth", f)
        ]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        latest_model = max(
            model_files,
            key=lambda x: int(re.match(r"best_model_epoch_(\d+)\.pth", x).group(1))
        )
        return os.path.join(model_dir, latest_model)

    @staticmethod
    def get_domain(tree_positions):
        """Return the domain (bounding box) of the tree positions."""
        x_min = np.min(tree_positions[:, 0])
        x_max = np.max(tree_positions[:, 0])
        y_min = np.min(tree_positions[:, 1])
        y_max = np.max(tree_positions[:, 1])
        return [x_min, y_min], [x_max, y_max]

    # @staticmethod
    # def kin_model(nx, dt): # double integrator
    #     """
    #     Kinematic model: state X = [x, y, theta, vx, vy, omega] and control U = [ax, ay, angular_acc].
    #     Uses simple Euler integration.
    #     """
    #     X = ca.MX.sym('X', nx * 2)  # 6 states
    #     U = ca.MX.sym('U', nx)      # 3 controls
    #     rhs = ca.vertcat(X[nx:], U)
    #     f = ca.Function('f', [X, U], [rhs])
    #     intg_opts = {"number_of_finite_elements": 1, "simplify": 1}
    #     intg = ca.integrator('intg', 'rk', {'x': X, 'p': U, 'ode': f(X, U)}, 0, dt, intg_opts)
    #     xf = intg(x0=X, p=U)['xf']
    #     return ca.Function('F', [X, U], [xf])

    @staticmethod
    def kin_model(nx, dt): # single integrator
        """
        Kinematic model (single integrator): 
        state X = [x, y, theta], control U = [vx, vy, omega].
        Uses Euler integration.
        """
        X = ca.MX.sym('X', nx)  # [x, y, theta]
        U = ca.MX.sym('U', nx)  # [vx, vy, omega]
        
        # Dynamics: derivative of state is equal to control input
        rhs = U  # dx/dt = vx, dy/dt = vy, dtheta/dt = omega
        
        # Create CasADi function
        f = ca.Function('f', [X, U], [rhs])
        
        # Euler integration: x_{k+1} = x_k + dt * f(x_k, u_k)
        xf = X + dt * f(X, U)
        
        return ca.Function('F', [X, U], [xf])


    @staticmethod
    def bayes(lambda_prev, z):
        """
        Bayesian update for belief:
           lambda_next = (lambda_prev * z) / (lambda_prev * z + (1 - lambda_prev) * (1 - z))
        """
        prod = lambda_prev * z
        denom = prod + (1 - lambda_prev) * (1 - z) + 1e-9
        return prod / denom

    @staticmethod
    def entropy(p):
        """
        Compute the binary entropy of a probability p.
        Values are clipped to avoid log(0).
        """
        p = ca.fmax(ca.fmin(p, 1 - 1e-6), 1e-9)
        return (-p * ca.log10(p) - (1 - p) * ca.log10(1 - p)) / ca.log10(2)
    
    @staticmethod
    def penalty_2d(x, y, x_c, y_c, p=10, s=1, a=1):
        """Compute penalty term"""
        # return np.exp(-(x**p + y**p)/((2.0*s)**p))
        return a*ca.exp(-((x-x_c)**p + (y-y_c)**p)/((2.0*s)**p))
    
    def aggregation_2d(self, x, y, evol, idx, a=1, d=256):
        """Compute aggregation term"""
        x_c = self.trees_pos[idx][0]
        y_c = self.trees_pos[idx][1]
        lambda_c = evol[idx]
        # Norma quadra
        # return (a**2) * (1-lambda_c) * ((x - x_c)**2 + (y - y_c)**2) / (d**2)
        # Quadrati
        # p=4
        # s=1
        # return -a*(1-lambda_c) *ca.exp(-((x-x_c)**p + (y-y_c)**p)/((2.0*s)**p))
        # Distanza 
        # return a * (1-lambda_c) * ca.sqrt((x - x_c)**2 + (y - y_c)**2 + 1e-6) / d
        knowledge_term = (lambda_c>=0.5)*(1-lambda_c) + (lambda_c<0.5)*lambda_c
        knowledge_term = (knowledge_term>=0.05)*knowledge_term 
        return a * knowledge_term * ca.sqrt((x - x_c)**2 + (y - y_c)**2 + 1e-6) / d
    
    def d_lambda2_dx(self, positions):
        """Calcola adjacency, lambda2 e beta a partire da positions (1D: [x1,y1,...,xn,yn])."""
        n = len(positions) // 3
        q_flat = positions             # vettore 1D [x1, y1, ..., xn, yn]
        q_all = q_flat.reshape((n, 3)) # matrice n x 3
        q = q_all[:,0:2]               # elimina yaw

        R = self.R
        sigma = np.sqrt((R ** 4) / np.log(2))

        # Calcolo matrice di adiacenza A
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                t0 = q[i] - q[j]
                d2 = np.dot(t0, t0)  # distanza al quadrato
                if d2 < R**2:
                    t1 = R**2 - d2
                    A_ij = self.alpha_elem * (np.exp((t1**2) / (sigma**2)) - 1)
                    A[i, j] = A[j, i] = A_ij
                else:
                    A[i, j] = A[j, i] = 0.0

        self.adjacency = A

        # Costruzione Laplaciano L = D - A
        D = np.diag(np.sum(A, axis=1))
        L = D - A

        # Calcolo lambda2 e autovettori
        eigvals, eigvecs = np.linalg.eigh(L)
        eigvals_sorted = np.sort(eigvals)
        idx_sorted = np.argsort(eigvals)
        v2 = eigvecs[:, idx_sorted[1]] if n > 1 else np.zeros(n)
        self.lambda2 = eigvals_sorted[1] if n > 1 else 0.0

        # Calcolo gradiente beta_i
        beta = np.zeros(2)
        i = self.n_agent-1
        for j in range(n):
            if i == j or A[i, j] == 0:
                continue

            dv = v2[i] - v2[j]
            t0 = q[i] - q[j]  # vettore (2,)
            t1 = R**2 - np.dot(t0, t0)
            t2 = sigma**2
            exp_term = np.exp((t1**2) / t2)
            dadxi = -4 * t1 * exp_term / t2 * t0
            beta += dadxi * (dv ** 2)

        self.beta = beta

    def d_lambda2_dx_predicition(self, positions):
        """
        Compute lambda, nabla lambda2 given current positions and adjacency matrix
        (used for prection)
        """
        n = len(positions) // 2
        q_flat = positions             # vettore 1D [x1, y1, ..., xn, yn]
        q = q_flat.reshape((n, 2))     # matrice n x 2

        R = self.R
        sigma = np.sqrt((R ** 4) / np.log(2))

        # Calcolo matrice di adiacenza A
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                t0 = q[i] - q[j]
                d2 = np.dot(t0, t0)  # distanza al quadrato
                if d2 < R**2 and self.adjacency[i, j] > 0:
                    t1 = R**2 - d2
                    A_ij = self.alpha_elem * (np.exp((t1**2) / (sigma**2)) - 1)
                    A[i, j] = A[j, i] = A_ij
                else:
                    A[i, j] = A[j, i] = 0.0

        D = np.diag(np.sum(A, axis=1))
        L = D - A
        # Calcolo lambda2 e autovettori
        eigvals, eigvecs = np.linalg.eigh(L)
        eigvals_sorted = np.sort(eigvals)
        idx_sorted = np.argsort(eigvals)
        v2 = eigvecs[:, idx_sorted[1]] if n > 1 else np.zeros(n)
        lambda2 = eigvals_sorted[1] if n > 1 else 0.0

        # Calcolo gradiente beta_i
        beta = np.zeros(2)
        i = self.n_agent-1
        for j in range(n):
            if i == j or A[i, j] == 0:
                continue

            dv = v2[i] - v2[j]
            t0 = q[i] - q[j]  # vettore (2,)
            t1 = R**2 - np.dot(t0, t0)
            t2 = sigma**2
            exp_term = np.exp((t1**2) / t2)
            dadxi = -4 * t1 * exp_term / t2 * t0
            beta += dadxi * (dv ** 2)

        return lambda2, beta
    
    def compute_minimum_spanning_tree(self):
        # Invert A because it is reverse-weighted (for min span tree)
        inverted_adjacency = np.copy(self.adjacency)
        non_zero_elements = inverted_adjacency != 0
        inverted_adjacency[non_zero_elements] = 1.0 / inverted_adjacency[non_zero_elements]
        mst_sparse = minimum_spanning_tree(inverted_adjacency)
        self.span_tree = mst_sparse.toarray()
        # Make it symmetric matrix 
        # (The higher the value the higher, the lower the distance)
        self.span_tree = np.maximum(self.span_tree, self.span_tree.T)
    
    def convert_traj(self):
        """
        Restituisce le posizioni di tutti gli agenti per ogni istante di tempo.
        Output:
            positions[t] = [x1, y1, x2, y2, ..., xN, yN] al tempo t
                        (se un agente non ha posizione al tempo t, inserisce NaN)
        """
        positions = []
        for t in range(self.N):
            frame = []
            for i in range(1, len(self.traj_x)):  # agenti da 1 a N
                if t < len(self.traj_x[i]) and t < len(self.traj_y[i]):
                    frame.append(self.traj_x[i][t])
                    frame.append(self.traj_y[i][t])
                else:
                    frame.append(float('nan'))
                    frame.append(float('nan'))
            positions.append(frame)
        return positions

    def lambda_betas_predictions(self):
        positions = self.convert_traj()
        lambda2s = ca.DM(self.lambda2)
        betas = ca.DM(self.beta)
        for i in range(1, self.N):
            lambda_k, beta_k = self.d_lambda2_dx_predicition(np.array(positions[i]))
            lambda2s = ca.vertcat(lambda2s, lambda_k)
            betas = ca.vertcat(betas, beta_k)
        return lambda2s, betas


    # ---------------------------
    # MPC Optimization Function 
    # ---------------------------
    # sostituisci lambda2 e beta con x_traj, y_traj e neigh
    # x_traj e y_traj sono le posizioni x e y di tutti i robot
    # neigh è un vettore di 0 e 1, che si usa nel vincolo di connessione per renderlo effettivo o no
    def mpc_opt(self, g_nn, trees, lb, ub, x0, lambda_vals, neighbors_positions, assigned_tree, x_traj, y_traj, span, steps=10):
        nx_local = 3                           # For clarity in this function
        # n_state = nx_local * 2               # 6-dimensional state: [x, y, theta, vx, vy, omega]
        n_state = nx_local                     # 3-dimensional state: [x, y, theta]
        n_control = nx_local                   # 3-dimensional control: [ax, ay, angular_acc]
        opti = ca.Opti()
        F_ = self.kin_model(self.nx, self.dt)  # kinematic model function

        # Decision variables:
        X = opti.variable(n_state, steps + 1)
        U = opti.variable(n_control, steps)

        # x_init_guess = ca.repmat(x0, 1, steps + 1) # Ripete x0 per tutti gli stati della traiettoria
        # u_init_guess = ca.DM.zeros(n_control, steps) # Controlli iniziali a zero
        # opti.set_initial(X, x_init_guess)
        # opti.set_initial(U, u_init_guess)

        # Parameter vector: initial state and tree beliefs.
        num_trees = trees.shape[0]
        num_neighbors = neighbors_positions.shape[0]
        num_x_traj = x_traj.shape[0]
        num_y_traj = y_traj.shape[0]
        num_span = span.shape[0]
        P0 = opti.parameter(n_state + num_trees + num_neighbors + num_x_traj + num_y_traj + num_span)
        X0 = P0[: n_state]
        L0 = P0[n_state:n_state+num_trees]
        N0 = P0[n_state+num_trees:n_state+num_trees+num_neighbors]
        TX0 = P0[n_state+num_trees+num_neighbors:n_state+num_trees+num_neighbors+num_x_traj]
        TY0 = P0[n_state+num_trees+num_neighbors+num_x_traj:n_state+num_trees+num_neighbors+num_x_traj+num_y_traj]
        S0 = P0[n_state+num_trees+num_neighbors+num_x_traj+num_y_traj:]
        # Initialize belief evolution.
        lambda_evol = [L0]

        # Convert tree positions to a CasADi DM.
        trees_dm = ca.DM(trees)  # Expected shape: (num_trees, 2)

        # Weights and safety parameters.
        w_control = 1e-3         # Control effort weight
        w_ang = 1e-4             # Angular control weight
        w_entropy = 1e1          # Weight for final entropy
        w_attract = 1e-2         # Weight for low-entropy attraction
        safe_distance = 1        # Safety margin (meters)

        # Initialize the objective.
        obj = 0

        # Initial condition constraint.
        opti.subject_to(X[:, 0] == X0)

        # Penalty term for cells
        penalty_cells = 0 # unassigned
        aggregation = 0   # assigned
        nearest_tree_attraction = 0 # closest tree

        # Not assigned trees (ID)
        not_assigned_tree = [num for num in list(range(num_trees)) if num not in assigned_tree]

        # Max vel
        max_vel = 2

        # Loop over the prediction horizon.
        for i in range(steps+1):
            # State and input bounds.
            opti.subject_to(opti.bounded(lb[0] - 4., X[0, i], ub[0] + 4.))
            opti.subject_to(opti.bounded(lb[1] - 2., X[1, i], ub[1] + 2.))
            opti.subject_to(opti.bounded(-6*np.pi, X[2, i], +6*np.pi))

            # double integrator
            # opti.subject_to(opti.bounded(0.0, ca.sumsqr(X[3:5, i]),4.00))
            # opti.subject_to(opti.bounded(-3.14/4, X[5, i], 3.14 / 4))

            # Robot-Robot Collision avoidance
            pi = X0[0:2] 
            for n in self.neighbors_id:
                pj = N0[2*n:2*n+2]
                pij = pj - pi
                dirs = ca.dot(X[0:2, i] - (pi + pj) / 2, pij)
                dist = 0.3 * ca.norm_2(pij)
                opti.subject_to(opti.bounded(-ca.inf, dirs+dist , 0))

            if i < steps:
                # opti.subject_to(opti.bounded(0.0, ca.sumsqr(U[0:2, i]),max_vel**2))
                t = opti.variable() 
                opti.subject_to(t >= 0)
                opti.subject_to(t <= max_vel - 1e-1)
                opti.subject_to(U[0, i]**2 + U[1, i]**2 <= t**2)
                opti.subject_to(opti.bounded(-3.14/2, U[-1, i], 3.14/2))
                obj += w_control * ca.sumsqr(U[0:2, i]) + w_ang * ca.sumsqr(U[2, i])

            # Collision avoidance: ensure safety margin from trees.
            delta = X[:2, i] - trees_dm.T
            sq_dists = ca.diag(ca.mtimes(delta.T, delta))
            # opti.subject_to(ca.mmin(sq_dists) >= safe_distance**2)
            # Smooth min usando log-sum-exp
            alpha = 10  # parametro di smoothing
            smooth_min = -ca.log(ca.sum1(ca.exp(-alpha * sq_dists))) / alpha
            opti.subject_to(smooth_min >= safe_distance**2)

            if i < steps:
                # opti.subject_to(X[:, i + 1] == X[:, i] + self.dt * U[:, i]) #F_(X[:, i], U[:, i]))                
                opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))                

        nn_batch = []
        for i in range(steps):
            for j in range(trees_dm.shape[0]):
                obj_j_pos = trees_dm[j, :].T
                diff = X[:2, i + 1] - obj_j_pos
                heading_target = X[2, i+1]
                nn_batch.append(ca.horzcat(diff.T, heading_target))

        g_out = g_nn(ca.vcat([*nn_batch]))
        z_k = ca.fmax(g_out, 0.5)
        L0_ext = ca.vcat([L0 for _ in range(steps)])
        z_k_binary = (L0_ext>=0.5)*z_k + (L0_ext<0.5)*(1-z_k)

        for i in range(steps):
            lambda_next = self.bayes(lambda_evol[-1], z_k_binary[i*trees_dm.shape[0]:(i+1)*trees_dm.shape[0]])
            lambda_evol.append(lambda_next)

        # Limited area (Cells) and attraction
        for i in range(steps+1):
            # for n_a in not_assigned_tree:
            #     # penalty for unassigned cells
            #     penalty_cells += self.penalty_2d(X[0, i], X[1, i], self.trees_pos[n_a][0], self.trees_pos[n_a][1], p=10, s=0.9, a=5)
            for a_a in assigned_tree:
                # aggregation term for assigned cells 
                aggregation += self.aggregation_2d(X[0, i], X[1, i], lambda_evol[i], idx=a_a, a=0.1) # / len(assigned_tree) #a=13

        # # Calcola distanze quadrate dalla posizione iniziale a tutti gli alberi
        # w_nearest_attract = 1e1      # Weight for nearest tree attraction
        # sq_dist_to_targets = ca.sum1((X0[:2] - trees_dm.T)**2)
        # min_sq_dist = ca.mmin(sq_dist_to_targets)
        # # 2. Define sigmoid parameters
        # threshold_sq_dist = 81.0
        # sigmoid_steepness = 10.0
        # # 3. Calculate the sigmoid factor
        # # This factor smoothly goes from ~0 (when min_sq_dist << threshold) to ~1 (when min_sq_dist >> threshold)
        # sigmoid_factor = 1.0 / (1.0 + ca.exp(-sigmoid_steepness * (min_sq_dist - threshold_sq_dist)))
        # # 4. Apply the modulation to the attraction term
        # modulated_attraction_term = w_nearest_attract * sigmoid_factor
            
        # Compute entropy terms for the objective.
        entropy_future = self.entropy(ca.vcat([*lambda_evol[1:]]))
        # entropy_term = ca.sum1( ca.vcat([ca.exp(-2*i)*ca.DM.ones(num_trees) for i in range(steps)]) * entropy_future) * w_entropy
        #--------------------------- Annulla la funzione degli alberi che non mi interessano
        mask = ca.DM.zeros(num_trees * steps, 1)
        for step in range(steps):
            for i in assigned_tree:
                idx = i + step * num_trees  # indices step successivi
                mask[idx] = 1
        exp_weights = ca.vcat([ca.exp(-2*i) * ca.DM.ones(num_trees, 1) for i in range(steps)])
        # entropy_term = ca.sum1((mask * exp_weights) * entropy_future) * w_entropy
        entropy_term = ca.logsumexp((mask * exp_weights) * entropy_future) * w_entropy
        #---------------------------        
        # Add terms to the objective.
        obj += entropy_term
        # obj += penalty_cells
        obj += aggregation
        # obj += modulated_attraction_term
        opti.minimize(obj)

        # Solver options
        # options = {
        #     "ipopt": {
        #         "tol": 1e-2,
        #         "warm_start_init_point": "yes",
        #         "hessian_approximation": "limited-memory",
        #         "print_level": 0,
        #         "sb": "no",
        #         "mu_strategy": "monotone",
        #         "max_iter": 3000 #3000
        #     },
        #     "print_time": False               # Disattiva stime di tempo
        # }
        options = {
            "ipopt": {
                # ===== VINCOLI HARD - TOLLERANZE STRETTE =====
                "constr_viol_tol": 1e-3,        # VINCOLI MOLTO STRETTI (hard)
                "compl_inf_tol": 1e-6,          # Complementarità stretta per vincoli
                # ===== OTTIMALITÀ RILASSATA =====
                "tol": 1e-3,                    # Ottimalità molto rilassata (sub-ottimo OK)
                "dual_inf_tol": 1e-3,           # Infeasibility duale rilassata
                # ===== FALLBACK SUB-OTTIMO =====
                "acceptable_tol": 1e-1,            # Tolleranza emergency molto alta
                "acceptable_constr_viol_tol": 1e-4,  # Ma vincoli sempre rispettati!
                "acceptable_dual_inf_tol": 10,  # Duale emergency rilasciato              
                # ===== STRATEGIA VELOCE =====
                "mu_strategy": "adaptive",      # Monotona più veloce
                "mu_init": 1e-1,               # Valore iniziale medio
                "mu_min": 1e-6,                
                "barrier_tol_factor": 10,       # Fattore barriera minimo
                # ===== HESSIANA MINIMA =====
                "hessian_approximation": "limited-memory",
                "limited_memory_max_history": 2,  # Storia brevissima
                # ===== WARM START ESSENZIALE =====
                "warm_start_init_point": "yes",
                "warm_start_bound_push": 1e-6,
                "warm_start_bound_frac": 1e-6,
                "warm_start_slack_bound_push": 1e-6,
                "warm_start_slack_bound_frac": 1e-6,
                # ===== STEP AGGRESSIVI =====
                "alpha_min_frac": 0.1,         # Step minimo basso per rispettare vincoli
                "accept_after_max_steps": 1,   # Accetta dopo 1 step
                # ===== GESTIONE BOUNDS PRECISA =====
                "bound_push": 1e-6,            # Push bounds piccolo (vincoli hard)
                "bound_frac": 1e-6,            # Frazione bounds piccola
                "slack_bound_push": 1e-6,      # Slack bounds precisi
                "slack_bound_frac": 1e-6,      
                # ===== SCALING DISABILITATO =====
                "nlp_scaling_method": "none",  # No scaling per velocità
                # ===== ALGORITMI DISABILITATI =====
                "mehrotra_algorithm": "no",    # Disabilita per velocità
                # ===== SOLVER LINEARE VELOCE =====
                "linear_solver": "mumps",      # Robusto e veloce
                # ===== PERTURBAZIONI MINIME =====
                "min_hessian_perturbation": 1e-12,
                "first_hessian_perturbation": 1e-6,
                # ===== EVITA RESTORATION =====
                "expect_infeasible_problem": "no",
                "start_with_resto": "no",
                "required_infeasibility_reduction": 0.1,
                # ===== OUTPUT MINIMO =====
                "print_level": 0,
                "sb": "no",
                "print_timing_statistics": "no",
                # ===== CALCOLI VELOCI =====
                "fast_step_computation": "yes",
                "derivative_test": "none",
                "check_derivatives_for_naninf": "no",
            },
            "print_time": False
        }
        # options = {
        #     "ipopt": {
        #         "hessian_approximation": "limited-memory",
        #         "print_level": 0,
        #         "mu_strategy": "monotone",
        #     },
        #     "print_time": False               # Disattiva stime di tempo
        # }
        opti.solver("ipopt", options)
        # Set the parameter values.
        opti.set_value(P0, ca.vertcat(x0, lambda_vals, neighbors_positions, x_traj, y_traj, span))
        sol = opti.solve()

        # check solution        
        stats = sol.stats()
        status = stats['return_status']
        success_states = ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
        if status not in success_states:
            rospy.logerr(f"Agent {self.n_agent} - Solver failed: {status}")

        # Create the MPC step function for warm starting.
        inputs = [P0, opti.x, opti.lam_g]
        outputs = [U[:, 0], X, opti.x, opti.lam_g]
        mpc_step = opti.to_function("mpc_step", inputs, outputs)

        return (mpc_step,
                ca.DM(sol.value(U[:, 0])),
                ca.DM(sol.value(X)),
                ca.DM(sol.value(opti.x)),
                ca.DM(sol.value(opti.lam_g)))

    # ---------------------------
    # Simulation Function
    # ---------------------------
    def run_simulation(self):
        F_ = self.kin_model(self.nx, self.dt)
        lb, ub = self.get_domain(self.trees_pos)
        self.mpc_horizon = self.N

        # Wait until a GPS message has been received.
        rospy.loginfo("Waiting for GPS data...")
        while self.current_state is None and not rospy.is_shutdown():
            rospy.sleep(0.05)
        rospy.loginfo("GPS data received.")

        # ---------------------------
        # Load the Learned Neural Network Models
        # ---------------------------
        model = MultiLayerPerceptron(input_dim=self.nn_input_dim,
                                     hidden_size=self.hidden_size,
                                     hidden_layers=self.hidden_layers)
        model.load_state_dict(torch.load(self.get_latest_best_model(), weights_only=True))
        model.eval()
        g_nn = l4c.L4CasADi(model, batched=True, device='cuda')

        # Initialize robot state from the latest GPS callback.
        initial_state = self.current_state  # [x, y, theta]
        vx_k = ca.DM.zeros(self.nx)  # velocity component: [vx, vy, omega]
        # x_k = ca.vertcat(ca.DM(initial_state), vx_k) # double integrator
        x_k = ca.DM(initial_state)


        # Containers for simulation output.
        all_trajectories = []
        self.lambda_history = []
        self.entropy_history = []
        durations = []

        self.velocity_command_log = []
        self.pose_history = []      # [x, y, theta] for each step
        self.time_history = []      # simulation time at each step

        # event driven communication values
        self.last_sent_lambda = None           # Ultimo lambda inviato
        self.cumulative_info_change = 0.0      # Cambiamento cumulativo di informazione
        self.info_threshold = 0.1              # Soglia per il cambiamento cumulativo
        self.tree_completion_threshold = 0.95  # Soglia per considerare un albero maturo
        self.tree_rejection_threshold = 0.05   # Soglia per considerare un albero acerbo
        self.last_tree_states = None           # Stati precedenti degli alberi per rilevare completamenti

        self.sim_start_time = time.time()
        self.total_distance = 0.0
        self.total_commands = 0
        sum_vx = 0.0
        sum_vy = 0.0
        sum_yaw = 0.0
        sum_trans_speed = 0.0
        prev_x = float(x_k[0])
        prev_y = float(x_k[1])

        sim_time = 2800  # Total simulation time in seconds
        mpciter = 0
        rate = rospy.Rate(int(1/self.dt))
        warm_start = True

        prev_lambda = self.lambda_k.full().flatten().copy()

        # Main MPC loop.
        while mpciter < sim_time and not rospy.is_shutdown():
            # rospy.loginfo('Step: %d', mpciter)
            # Update state from the latest GPS callback.
            while (self.current_state is None or self.robot_positions is None or self.traj_x is None or self.traj_y is None) and not rospy.is_shutdown():
                if mpciter == 0 and self.robot_positions is not None and self.current_state is not None:
                    break
                rospy.sleep(0.05)
            # current_state = self.current_state
            current_state = self.robot_positions[3*(self.n_agent-1):3*(self.n_agent-1)+3]
            current_sim_time = time.time() - self.sim_start_time
            ############## Update lambda (current + consensus)
            if self.aggregated_data:
                self.lambda_k =  self.lambda_cons # Commenta per togliere consenso
                self.aggregated_data = False
            ##############
            self.pose_history.append(current_state)
            self.time_history.append(current_sim_time)
            
            x_k = ca.DM(current_state)  
            # print(self.n_agent, ", x_k: ",x_k)        
            # print(self.n_agent, ", state: ",self.current_state)        
            # Wait until tree scores have been received.
            while self.latest_trees_scores is None and not rospy.is_shutdown():
                rospy.sleep(0.05)
            latest_trees_scores = self.latest_trees_scores.copy()
            ######### Get detected trees (time)
            # print("=====================\n", latest_trees_scores.T)
            detected_trees = (latest_trees_scores != 0.5).astype(float)
            times_detect = np.array(time.time() * detected_trees).flatten()
            self.latest_detection = np.where(times_detect != 0, times_detect, self.latest_detection) # update new detections
            # print(self.latest_detection, "\n=====================")
            ########
            self.lambda_k = self.bayes(self.lambda_k, latest_trees_scores)
            self.lambda_k = np.ceil(self.lambda_k*1000)/1000
            # rospy.loginfo("Current state x_k: %s", x_k)
            # rospy.loginfo("Lambda: %s", self.lambda_k)
            # rospy.loginfo("Current tree scores: %s", latest_trees_scores.flatten())
            # rospy.loginfo("VICINI: %s", self.neighbors_pos)
            current_lambda = self.lambda_k.full().flatten()

            # LOGICA PER DECIDERE SE INVIARE I DATI LAMBDA
            should_send_lambda = False
            send_reason = ""
            # Regola 1: Completamento/rifiuto di un albero
            for i, (prev_val, curr_val) in enumerate(zip(prev_lambda, current_lambda)):
                if i in self.assigned: # Solo albero assegnato
                    # Albero completato (da <0.95 a >=0.95)
                    if prev_val < self.tree_completion_threshold and curr_val >= self.tree_completion_threshold:
                        should_send_lambda = True
                        send_reason = f"Tree {i} completed (λ={curr_val:.3f})"
                        rospy.loginfo(f"\033[92mTree {i} completed! λ={curr_val:.3f}\033[0m")
                        break
                    # Albero rifiutato (da >0.05 a <=0.05)
                    elif prev_val > self.tree_rejection_threshold and curr_val <= self.tree_rejection_threshold:
                        should_send_lambda = True
                        send_reason = f"Tree {i} completed (λ={curr_val:.3f})"
                        rospy.loginfo(f"\033[91mTree {i} rejected! λ={curr_val:.3f}\033[0m")
                        break
            # Regola 2: Cambiamento cumulativo dell'informazione
            # if not should_send_lambda and prev_lambda is not None:
            #     # Calcola il cambiamento di informazione (puoi usare diverse metriche)
            #     # Opzione A: Distanza euclidea
            #     info_change = np.linalg.norm(current_lambda - prev_lambda)
            #     # Opzione B: Differenza di entropia (commentata, usa una delle due)
            #     # prev_entropy = -np.sum(prev_lambda * np.log(prev_lambda + 1e-8) + 
            #     #                       (1-prev_lambda) * np.log(1-prev_lambda + 1e-8))
            #     # curr_entropy = -np.sum(current_lambda * np.log(current_lambda + 1e-8) + 
            #     #                       (1-current_lambda) * np.log(1-current_lambda + 1e-8))
            #     # info_change = abs(curr_entropy - prev_entropy)
            #     self.cumulative_info_change += info_change
            #     if self.cumulative_info_change >= self.info_threshold:
            #         should_send_lambda = True
            #         send_reason = f"Cumulative info change threshold reached ({self.cumulative_info_change:.3f})"
            #         self.cumulative_info_change = 0.0  # Reset del contatore
            # Primo step: invia sempre
            if self.last_sent_lambda is None:
                should_send_lambda = True
                send_reason = "Initial step"



            # Publish tree markers using the helper function from sensors.py.
            tree_markers_msg = create_tree_markers(self.trees_pos, self.lambda_k.full().flatten())
            self.tree_markers_pub.publish(tree_markers_msg)

            if self.assigned is not None and self.robot_positions is not None:
                self.d_lambda2_dx(self.robot_positions)
                # print(self.n_agent, ":", "\033[97m" + str(self.lambda2) + "\033[0m", "|", self.beta)
                self.compute_minimum_spanning_tree() 
                # if self.n_agent==1:
                #     print("\033[97m" + str(self.span_tree) + "\033[0m")
                # Adjacency for
                connections = np.where(self.span_tree > 0, 1, 0)
                adj_dm = ca.DM(connections[self.n_agent-1])
                step_start_time = time.time()
                if warm_start or not np.array_equal(self.assigned, prev_assigned): # MPC initialization or reinitialization
                    # Repeat start
                    x_traj_flat = np.repeat(self.robot_positions[::3], self.N+1)
                    y_traj_flat = np.repeat(self.robot_positions[1::3], self.N+1)
                    # Convert to CasADi DM
                    x_traj_dm = ca.DM(x_traj_flat)
                    y_traj_dm = ca.DM(y_traj_flat)
                    # MPC
                    mpc_step, u, x_traj, x_dec, lam = self.mpc_opt(g_nn, self.trees_pos, lb, ub, x_k, self.lambda_k, self.neighbors_pos, self.assigned, x_traj_dm, y_traj_dm, adj_dm, self.mpc_horizon)
                    warm_start = False
                    prev_assigned = self.assigned.copy()
                else: # MPC step
                    # Move to accomplish the task (if not completed)
                    if np.any(self.lambda_k.full().flatten()[self.assigned] < 0.95):
                        # Flattening agent trajectories (excluding dummy index 0)
                        x_traj_flat = [elem for traj in self.traj_x[1:] for elem in traj] #traj[:-1]]
                        y_traj_flat = [elem for traj in self.traj_y[1:] for elem in traj] #traj[:-1]]
                        # print(self.n_agent, "=================")
                        # print(self.n_agent, "Trajs: ", self.traj_x)
                        # # print(self.n_agent, "X: ", x_traj_flat)
                        # np.set_printoptions(precision=17, suppress=True) # 17 cifre per float64, suppress per notazione non scientifica
                        # print(self.n_agent, "x_k:", x_k.full())
                        # print(self.n_agent, "=================")
                        # Convert to CasADi DM
                        x_traj_dm = ca.DM(x_traj_flat)
                        y_traj_dm = ca.DM(y_traj_flat)
                        # MPC
                        # print(self.n_agent, ": ", adj_dm)
                        u, x_traj, x_dec, lam = mpc_step(ca.vertcat(x_k, self.lambda_k, ca.DM(self.neighbors_pos), x_traj_dm, y_traj_dm, adj_dm), x_dec, lam)
                        # mpc_step, u, x_traj, x_dec, lam = self.mpc_opt(g_nn, self.trees_pos, lb, ub, x_k, self.lambda_k, self.neighbors_pos, self.assigned, x_traj_dm, y_traj_dm, adj_dm, self.mpc_horizon)

                # msg = Int32()
                # msg.data = self.n_agent
                # self.ok_mpc.publish(msg)

                self.robot_positions = None
                self.traj_x = None
                self.traj_y = None

                # Sync msg
                msg = Trajectory()
                msg.header = Header()
                msg.header.stamp = rospy.Time.now()
                msg.id = self.n_agent
                msg.positions_x = x_traj[0, :].full().flatten().tolist()
                msg.positions_y = x_traj[1, :].full().flatten().tolist()
                msg.theta = x_traj[2, :].full().flatten().tolist()
                self.ok_mpc.publish(msg)
                # rospy.loginfo("\033[92mOk " + str(self.n_agent) + " \033[0m")

                durations.append(time.time() - step_start_time)
                # Log the MPC velocity command.
                u_np = np.array(u.full()).flatten()

                # Compute the command pose.
                if np.any(self.lambda_k.full().flatten()[self.assigned] < 0.95): # Stay still if task completed
                    cmd_pose = x_traj[:,1] # x_k + self.dt * u[:, 0]  # x_traj[:,1] # F_(x_k, u[:, 0])
                    # if self.n_agent == 2: # debug
                    #     print("====================")
                    #     print("predicitions, ", x_traj_flat)
                    #     print("x_k:", x_k)
                    #     print("x: ", msg.positions_x)
                    #     print("y: ", msg.positions_y)
                    #     print("u:", u[:, 0])
                    #     print("CMD pose:", cmd_pose)
                    #     if np.abs(x_k[0]-msg.positions_x[0]) > 0.1:
                    #         rospy.loginfo("\033[92m" + " ******** VALORI DIVERSI" + "\033[0m")
                else:
                    rospy.loginfo("\033[92mAgent " + str(self.n_agent) + ": done\033[0m")

                # Publish predicted path.
                predicted_path_msg = create_path_from_mpc_prediction(x_traj[:self.nx, 1:])
                self.pred_path_pub.publish(predicted_path_msg)

                # Build and publish the cmd_pose message.
                quaternion = tf.transformations.quaternion_from_euler(0, 0, float(cmd_pose[2]))
                cmd_pose_msg = Pose()
                cmd_pose_msg.position = Point(x=float(cmd_pose[0]), y=float(cmd_pose[1]), z=0.0)
                cmd_pose_msg.orientation = Quaternion(x=quaternion[0],
                                                    y=quaternion[1],
                                                    z=quaternion[2],
                                                    w=quaternion[3])
                self.cmd_pose_pub.publish(cmd_pose_msg)

                # vx_k = cmd_pose[self.nx:] # double integrator
                vx_k = u[:, 0]
                self.velocity_command_log.append([current_sim_time, "MPC", vx_k[0], vx_k[1], vx_k[2]])

                # Update metrics.
                vx_val = float(vx_k[0])
                vy_val = float(vx_k[1])
                yaw_val = float(vx_k[2])
                sum_vx += vx_val
                sum_vy += vy_val
                sum_yaw += yaw_val
                trans_speed = math.sqrt(vx_val**2 + vy_val**2)
                sum_trans_speed += trans_speed
                self.total_commands += 1

                curr_x = float(x_traj[0, 1])
                curr_y = float(x_traj[1, 1])
                distance_step = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                self.total_distance += distance_step
                prev_x, prev_y = curr_x, curr_y

                entropy_k = ca.sum1(self.entropy(self.lambda_k)).full().flatten()[0]
                self.lambda_history.append(self.lambda_k.full().flatten().tolist())
                self.entropy_history.append(entropy_k)
                all_trajectories.append(x_traj[:self.nx, :].full())

                mpciter += 1
                rospy.loginfo("Entropy: %s", entropy_k)
                if all( v >=0.95 for v in  self.lambda_k.full().flatten()):
                    break

            # # send lambda values
            # lambda_msg = Float64MultiArray()
            # lambda_msg.data = np.concatenate([self.lambda_k.full().flatten(), self.latest_detection]) # self.lambda_k.full().flatten()
            # self.lambda_pub.publish(lambda_msg)

                if should_send_lambda:
                    lambda_msg = Float64MultiArray()
                    lambda_msg.data = np.concatenate([current_lambda, self.latest_detection])
                    self.lambda_pub.publish(lambda_msg)
                    # Aggiorna stato per prossime decisioni
                    self.last_sent_lambda = current_lambda.copy()
                    rospy.loginfo(f"\033[94mLambda sent: {send_reason}\033[0m")

                prev_lambda = self.lambda_k.full().flatten().copy()
            
            rate.sleep()

        return all_trajectories, self.entropy_history, self.lambda_history, durations, g_nn, self.trees_pos, lb, ub

    def save_all(self):
        """
        Final Metrics Calculation and CSV Output
        """
        self.total_execution_time = time.time() - self.sim_start_time
        self.avg_wp_time = self.total_execution_time / self.total_commands if self.total_commands > 0 else 0.0
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baselines_dir = os.path.join(script_dir, "baselines")
        os.makedirs(baselines_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        perf_csv = os.path.join(baselines_dir, f"mpc_n{self.n_agent}_{timestamp}_performance_metrics.csv")
        with open(perf_csv, mode='w', newline='') as perf_file:
            writer = csv.writer(perf_file)
            writer.writerow([
                "Total Execution Time (s)",
                "Total Distance (m)",
                "Average Waypoint-to-Waypoint Time (s)",
                "Final Entropy",
                "Total Commands"
            ])
            writer.writerow([
                self.total_execution_time,
                self.total_distance,
                self.avg_wp_time,
                self.entropy_history[-1] if self.entropy_history else "N/A",
                self.total_commands
            ])

        vel_csv = os.path.join(baselines_dir, f"mpc_n{self.n_agent}_{timestamp}_velocity_commands.csv")
        with open(vel_csv, mode='w', newline='') as vel_file:
            writer = csv.writer(vel_file)
            writer.writerow(["Time (s)", "Tag", "x_velocity", "y_velocity", "yaw_velocity"])
            writer.writerows(self.velocity_command_log)

        rospy.loginfo("Performance metrics saved to %s", perf_csv)
        rospy.loginfo("Velocity command log saved to %s", vel_csv)

        plot_csv = os.path.join(baselines_dir, f"mpc_n{self.n_agent}_{timestamp}_plot_data.csv")
        with open(plot_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            tree_positions_flat = self.trees_pos.flatten().tolist()
            writer.writerow(["tree_positions"] + tree_positions_flat)
            header = ["time", "x", "y", "theta", "entropy"]
            if self.lambda_history and len(self.lambda_history[0]) > 0:
                num_trees = len(self.lambda_history[0])
                header += [f"lambda_{i}" for i in range(num_trees)]
            writer.writerow(header)
            for i in range(len(self.time_history)):
                time_val = self.time_history[i]
                x, y, theta = np.array(self.pose_history[i]).flatten()
                entropy_val = self.entropy_history[i]
                lambda_vals = self.lambda_history[i]
                row = [time_val, x, y, theta, entropy_val] + lambda_vals
                writer.writerow(row)
        rospy.loginfo("Plot data saved to %s", plot_csv)


    # ---------------------------
    # Plotting Function
    # ---------------------------
def plot_animated_trajectory_and_entropy_2d(self, all_trajectories, entropy_history, lambda_history, trees, lb, ub, computation_durations):
        # Reload the model for plotting.
        model = MultiLayerPerceptron(input_dim=self.nn_input_dim,
                                     hidden_size=self.hidden_size,
                                     hidden_layers=self.hidden_layers)
        model.load_state_dict(torch.load(self.get_latest_best_model(), weights_only=True))
        model.eval()
        g_nn = l4c.L4CasADi(model, name='plotting_f', batched=True, device='cuda')

        x_trajectory = np.array([traj[0] for traj in all_trajectories])
        y_trajectory = np.array([traj[1] for traj in all_trajectories])
        theta_trajectory = np.array([traj[2] for traj in all_trajectories])
        all_trajectories = np.array(all_trajectories)
        lambda_history = np.array(lambda_history)

        # Compute predicted entropy reduction.
        entropy_mpc_pred = []
        for k in range(all_trajectories.shape[0]):
            lambda_k = lambda_history[k]
            entropy_mpc_pred_k = [entropy_history[k]]
            for i in range(all_trajectories.shape[2]-1):
                relative_position_robot_trees = np.tile(all_trajectories[k, :2, i+1], (trees.shape[0], 1)) - trees
                distance_robot_trees = np.sqrt(np.sum(relative_position_robot_trees**2, axis=1))
                theta = np.tile(all_trajectories[k, 2, i+1], (trees.shape[0], 1))
                input_nn = ca.horzcat(relative_position_robot_trees, theta)
                z_k =  ca.fmax(g_nn(input_nn), 0.5)
                lambda_k = self.bayes(lambda_k, z_k)
                reduction = ca.sum1(self.entropy(lambda_k)).full().flatten()[0]
                entropy_mpc_pred_k.append(reduction)
            entropy_mpc_pred.append(entropy_mpc_pred_k)

        entropy_mpc_pred = np.array(entropy_mpc_pred)
        sum_entropy_history = entropy_history
        cumulative_durations = np.cumsum(computation_durations)

        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.6, 0.4],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )

        # MPC predicted trajectory.
        fig.add_trace(
            go.Scatter(
                x=x_trajectory[0],
                y=y_trajectory[0],
                mode="lines+markers",
                name="MPC Future Trajectory",
                line=dict(width=4),
                marker=dict(size=5)
            ),
            row=1, col=1
        )
        # Drone trajectory (empty initially).
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name="Drone Trajectory",
                line=dict(width=4),
                marker=dict(size=5)
            ),
            row=1, col=1
        )

        # Add tree markers.
        for i in range(trees.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=[trees[i, 0]],
                    y=[trees[i, 1]],
                    mode="markers+text",
                    marker=dict(size=10, colorscale=[[0, "#FF0000"], [1, "#00FF00"]]),
                    name=f"Tree {i}: {lambda_history[0][i]:.2f}",
                    text=[str(i)],
                    textposition="top center"
                ),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name="Sum of Entropies (Past)",
                line=dict(width=2),
                marker=dict(size=5)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name="Sum of Entropies (Future)",
                line=dict(width=2, dash="dot"),
                marker=dict(size=5)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name="Computation Durations",
                line=dict(width=2),
                marker=dict(size=5)
            ),
            row=2, col=2
        )

        # Create animation frames.
        frames = []
        for k in range(len(entropy_mpc_pred)):
            tree_data = []
            for i in range(trees.shape[0]):
                tree_data.append(
                    go.Scatter(
                        x=[trees[i, 0]],
                        y=[trees[i, 1]],
                        mode="markers+text",
                        marker=dict(size=10, color=[2*(lambda_history[k][i]-0.5)],
                                    colorscale=[[0, "#FF0000"], [1, "#00FF00"]]),
                        name=f"Tree {i}: {lambda_history[k][i]:.2f}",
                        text=[str(i)],
                        textposition="top center"
                    )
                )
            sum_entropy_past = sum_entropy_history[:k+1]
            sum_entropy_future = entropy_mpc_pred[k]
            computation_durations_past = computation_durations[:k+1]

            x_start = x_trajectory[k]
            y_start = y_trajectory[k]
            theta = theta_trajectory[k]
            x_end = x_start + 0.5 * np.cos(theta)
            y_end = y_start + 0.5 * np.sin(theta)
            list_of_actual_orientations = []
            for x0, y0, x1, y1 in zip(x_start, y_start, x_end, y_end):
                arrow = go.layout.Annotation(
                    dict(
                        x=x1, y=y1,
                        xref="x", yref="y",
                        showarrow=True,
                        ax=x0, ay=y0,
                        arrowhead=3, arrowwidth=1.5,
                        arrowcolor="red",
                    )
                )
                list_of_actual_orientations.append(arrow)
            for x0, y0, x1, y1 in zip(x_trajectory[:k+1, 0],
                                        y_trajectory[:k+1, 0],
                                        x_trajectory[:k+1, 0] + 0.5 * np.cos(theta_trajectory[:k+1, 0]),
                                        y_trajectory[:k+1, 0] + 0.5 * np.sin(theta_trajectory[:k+1, 0])):
                arrow = go.layout.Annotation(
                    dict(
                        x=x1, y=y1,
                        xref="x", yref="y",
                        showarrow=True,
                        ax=x0, ay=y0,
                        arrowhead=3, arrowwidth=1.5,
                        arrowcolor="orange",
                    )
                )
                list_of_actual_orientations.append(arrow)

            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=x_trajectory[k],
                        y=y_trajectory[k],
                        mode="lines+markers",
                        line=dict(width=4),
                        marker=dict(size=5)
                    ),
                    go.Scatter(
                        x=x_trajectory[:k+1, 0],
                        y=y_trajectory[:k+1, 0],
                        mode="lines+markers",
                        line=dict(width=4),
                        marker=dict(size=5)
                    ),
                    *tree_data,
                    go.Scatter(
                        x=np.arange(len(sum_entropy_past)),
                        y=sum_entropy_past,
                        mode="lines+markers",
                        line=dict(width=2),
                        marker=dict(size=5)
                    ),
                    go.Scatter(
                        x=np.arange(k, k+len(sum_entropy_future)),
                        y=sum_entropy_future,
                        mode="lines+markers",
                        line=dict(width=2, dash="dot"),
                        marker=dict(size=5)
                    ),
                    go.Scatter(
                        x=np.arange(len(computation_durations_past)),
                        y=computation_durations_past,
                        mode="lines+markers",
                        line=dict(width=2),
                        marker=dict(size=5)
                    )
                ],
                name=f"Frame {k}",
                layout=dict(annotations=list_of_actual_orientations)
            )
            frames.append(frame)

        fig.frames = frames
        fig.update_layout(
            title="Drone Trajectory, Sum of Entropies, and Computation Durations",
            xaxis=dict(title="X Position", range=[lb[0] - 3, ub[0] + 3]),
            yaxis=dict(title="Y Position", range=[lb[1] - 3, ub[1] + 3]),
            xaxis2=dict(title="Time Step"),
            yaxis2=dict(title="Sum of Entropies"),
            xaxis3=dict(title="Time Step"),
            yaxis3=dict(title="Computation Duration (s)"),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                        )
                    ],
                    showactive=True,
                    x=0.1,
                    y=0
                )
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 50, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(frames)
                ],
            }]
        )
        fig.show()
        fig.write_html('neural_mpc_results.html')
        return entropy_mpc_pred

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    rospy.init_node("neural_mpc_node", anonymous=True)
    mpc = NeuralMPC()
    sim_results = mpc.run_simulation()
    # Optionally, call the plotting method:
    # mpc.plot_animated_trajectory_and_entropy_2d(*sim_results[:-4], computation_durations=sim_results[3])
