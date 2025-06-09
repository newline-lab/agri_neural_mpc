#!/usr/bin/env python
import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension

class RobotsPositionListener:
    def __init__(self):
        rospy.init_node('robots_position_listener', anonymous=True)

        # Parametri
        self.num_robots = rospy.get_param('~num_robots', 3)
        self.alpha_elem = rospy.get_param('~alpha_elem', 1.0)
        self.R = rospy.get_param('~R', 3.0)
        self.sigma = np.sqrt((self.R**4) / np.log(2))

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publisher
        self.lambda2_pub = rospy.Publisher('/lambda2', Float64, queue_size=1)
        self.adjacency_pub = rospy.Publisher('/adjacency', Float64MultiArray, queue_size=1)

        # Loop principale
        self.main_loop()

    def main_loop(self):
        rate = rospy.Rate(30)  # 100 Hz
        while not rospy.is_shutdown():
            positions = self.get_robot_positions()
            if positions is not None:
                A = self.compute_adjacency_matrix(positions)
                self.publish_adjacency_matrix(A)
                lambda2 = self.compute_lambda2(A)
                self.lambda2_pub.publish(lambda2)
                # rospy.loginfo_throttle(1, f"Lambda2: {lambda2:.4f}")
            rate.sleep()

    def get_robot_positions(self):
        positions = np.zeros((self.num_robots, 2))
        for i in range(self.num_robots):
            try:
                frame_id = f'base_link_{i+1}'
                trans: TransformStamped = self.tf_buffer.lookup_transform('map', frame_id, rospy.Time(0))
                positions[i, 0] = trans.transform.translation.x
                positions[i, 1] = trans.transform.translation.y
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF lookup failed for robot {i}: {e}")
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

if __name__ == '__main__':
    try:
        RobotsPositionListener()
    except rospy.ROSInterruptException:
        pass
