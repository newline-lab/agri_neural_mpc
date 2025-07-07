#!/usr/bin/env python
import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension, Int32
import tf
from nmpc_ros.msg import Trajectory, MultiTraj
from std_msgs.msg import Header
import os


class RobotsPositionListener:
    def __init__(self):
        rospy.init_node('robots_position_listener', anonymous=True)
        rospy.on_shutdown(self.save_position_history)

        # Parametri
        self.num_robots = rospy.get_param('~num_robots', 3)
        self.alpha_elem = rospy.get_param('~alpha_elem', 1.0)
        self.R = rospy.get_param('~R', 3.0)
        self.sigma = np.sqrt((self.R**4) / np.log(2))

        # Sincronizzazione
        rospy.Subscriber("/mpc_ok", Trajectory, self.mpc_callback)
        self.current_step = 0
        self.curret_ok = [False] * self.num_robots # se tutti True step completo
        self.traj_x = [[] for _ in range(self.num_robots)]
        self.traj_y = [[] for _ in range(self.num_robots)]
        self.thetas = [[] for _ in range(self.num_robots)]
        self.history_pos = []

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publisher
        self.lambda2_pub = rospy.Publisher('/lambda2', Float64, queue_size=1)
        self.adjacency_pub = rospy.Publisher('/adjacency', Float64MultiArray, queue_size=1, latch=True)
        self.robot_states_pub = rospy.Publisher('/robot_states', Float64MultiArray, queue_size=1, latch=True)
        self.traj_pub = rospy.Publisher('/predictions', MultiTraj, queue_size=1)

        self.init_cmd()

        rospy.spin() 
    
    def init_cmd(self):
        while True: #not rospy.is_shutdown():
            positions = self.get_robot_positions()
            if positions is not None:
                GREEN = "\033[92m"
                RESET = "\033[0m"
                rospy.loginfo(f"{GREEN}NEXT STEP{RESET}")
                A = self.compute_adjacency_matrix(positions)
                self.publish_adjacency_matrix(A)
                lambda2 = self.compute_lambda2(A)
                self.lambda2_pub.publish(lambda2)
                self.publish_robot_positions(positions)
                return
        
    # metti a 1 l'id dell'mpc che ha fatto
    def mpc_callback(self, msg):
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        rospy.loginfo(f"{YELLOW}Received robot {msg.id}{RESET}") 
        self.curret_ok[msg.id-1] = True

        self.traj_x[msg.id - 1] = msg.positions_x
        self.traj_y[msg.id - 1] = msg.positions_y
        self.thetas[msg.id - 1] = msg.theta

        if all(self.curret_ok):
            # positions = self.get_robot_positions()
            positions = np.zeros((self.num_robots, 3))
            for i in range(self.num_robots):
                positions[i, 0] = self.traj_x[i][1]
                positions[i, 1] = self.traj_y[i][1]
                positions[i, 2] = self.thetas[i][1]
            self.history_pos.append(positions)
            GREEN = "\033[92m"
            RESET = "\033[0m"
            rospy.loginfo(f"{GREEN}NEXT STEP{RESET}")

            self.publish_trajectories()

            A = self.compute_adjacency_matrix(positions)
            self.publish_adjacency_matrix(A)
            lambda2 = self.compute_lambda2(A)
            self.lambda2_pub.publish(lambda2)
            self.publish_robot_positions(positions)

            self.curret_ok = [False] * self.num_robots
            self.traj_x = [[] for _ in range(self.num_robots)]
            self.traj_u = [[] for _ in range(self.num_robots)]



    def get_robot_positions(self):
        positions = np.zeros((self.num_robots, 3))
        for i in range(self.num_robots):
            try:
                frame_id = f'base_link_{i+1}'
                trans: TransformStamped = self.tf_buffer.lookup_transform('map', frame_id, rospy.Time(0))
                (_, _, yaw) = tf.transformations.euler_from_quaternion([
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w
                ])
                positions[i, 0] = trans.transform.translation.x
                positions[i, 1] = trans.transform.translation.y
                positions[i, 2] = yaw
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF lookup failed for robot {i+1}: {e}")
                return None
        return positions

    def compute_adjacency_matrix(self, positions):
        A = np.zeros((self.num_robots, self.num_robots))
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                d_squared = dx**2 + dy**2
                if d_squared < self.R**2:
                    term = ((self.R**2 - d_squared)**2) / (self.sigma**2)
                    A_ij = self.alpha_elem * (np.exp(term) - 1)
                else:
                    A_ij = 0.0
                A[i, j] = A[j, i] = A_ij
        return A

    def compute_lambda2(self, A):
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        eigvals = np.linalg.eigvalsh(L)
        eigvals.sort()
        return float(eigvals[1]) if len(eigvals) > 1 else 0.0

    def publish_adjacency_matrix(self, A):
        msg = Float64MultiArray()
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "rows"
        msg.layout.dim[0].size = self.num_robots
        msg.layout.dim[0].stride = self.num_robots * self.num_robots

        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[1].label = "cols"
        msg.layout.dim[1].size = self.num_robots
        msg.layout.dim[1].stride = self.num_robots

        msg.data = A.flatten().tolist()  # riga per riga (default)
        self.adjacency_pub.publish(msg)

    def publish_robot_positions(self, positions):
        msg = Float64MultiArray()
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "robots"
        msg.layout.dim[0].size = self.num_robots
        msg.layout.dim[0].stride = self.num_robots * 3  # 3 valori per robot

        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[1].label = "x_y_yaw"
        msg.layout.dim[1].size = 3
        msg.layout.dim[1].stride = 3

        msg.data = positions.flatten().tolist()
        self.robot_states_pub.publish(msg)

    def publish_trajectories(self):
        msg = MultiTraj()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()

        # Flatten le traiettorie
        msg.traj_x = [x for robot_x in self.traj_x for x in robot_x]
        msg.traj_y = [y for robot_y in self.traj_y for y in robot_y]
        msg.lengths = [len(robot_x) for robot_x in self.traj_x]  # o robot_y, sono uguali in lunghezza

        self.traj_pub.publish(msg)

    def save_position_history(self):
        try:
            if self.history_pos:
                # Ottieni la directory del file corrente
                script_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(script_dir, 'src/baselines/position_history.npy')
                
                # Converti in numpy array e salva
                history_array = np.array(self.history_pos)
                np.save(file_path, history_array)
                rospy.loginfo(f"Position history saved to {file_path}")
        except Exception as e:
            rospy.logerr(f"Error saving position history: {e}")

if __name__ == '__main__':
    try:
        RobotsPositionListener()
    except rospy.ROSInterruptException:
        pass
