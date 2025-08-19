from sensor_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs.msg import *
import numpy as np
from nmpc_ros.srv import GetTreesPoses
import tf
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import tf.transformations
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import rospy


create_pose_array = lambda data: PoseArray(
    poses=[
        Pose(
            position=Point(x=data[0, i, 0], y=data[1, i, 0], z=0),
            orientation=Quaternion(*tf.transformations.quaternion_from_euler(0, 0, data[2, i, 0]))
        ) for i in range(data.shape[1])
    ]
)

def create_path_from_mpc_prediction(mpc_prediction):
    path = Path()
    path.header.frame_id = "map"
    path.header.stamp = rospy.Time.now()

    for i in range(mpc_prediction.shape[1]):
        pose = PoseStamped()
        pose.header = path.header
        pose.header.stamp =  rospy.Time.now() + rospy.Duration(0.1*i)

        pose.pose.position.x = mpc_prediction[0, i]
        pose.pose.position.y = mpc_prediction[1, i]
        pose.pose.position.z = 1.5  # Assuming 2D motion

        # Assuming the third row is yaw, if available
        if mpc_prediction.shape[0] > 2:
            yaw = mpc_prediction[2, i]
            quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]
        else:
            pose.pose.orientation.w = 1.0

        path.poses.append(pose)

    return path

def create_tree_markers(trees_pos, lambda_values):
    markers = MarkerArray()
    lambda_values = np.array(lambda_values)

    BASE_TREE_DIAMETER = 0.4
    BASE_TREE_HEIGHT = 2.5
    BASE_TREE_COLOR_R, BASE_TREE_COLOR_G, BASE_TREE_COLOR_B, BASE_TREE_COLOR_A = 0.5, 0.35, 0.05, 1.0

    # Indicator Cylinders (Red/Green)
    INDICATOR_MIN_HEIGHT = 0.1
    INDICATOR_MAX_HEIGHT = .5
    INDICATOR_DIAMETER = 0.15
    INDICATOR_OFFSET_X = INDICATOR_DIAMETER * 0.6

    VARIABLE_HEIGHT_RANGE = INDICATOR_MAX_HEIGHT - INDICATOR_MIN_HEIGHT

    for i, (tree_pos, lambda_val) in enumerate(zip(trees_pos, lambda_values)):

        """# --- Base Tree Trunk Marker ---
        trunk_marker = Marker()
        trunk_marker.header.frame_id = 'map'
        trunk_marker.header.stamp = rospy.Time.now()
        trunk_marker.ns = "tree_trunks"
        trunk_marker.id = i * 3 # Unique ID base
        trunk_marker.type = Marker.CYLINDER
        trunk_marker.action = Marker.ADD
        # Position the base of the trunk at z=0, so its center is at z = BASE_TREE_HEIGHT / 2
        trunk_marker.pose.position = Point(x=tree_pos[0], y=tree_pos[1], z=BASE_TREE_HEIGHT / 2.0)
        trunk_marker.pose.orientation.w = 1.0
        trunk_marker.scale.x = BASE_TREE_DIAMETER
        trunk_marker.scale.y = BASE_TREE_DIAMETER
        trunk_marker.scale.z = BASE_TREE_HEIGHT
        trunk_marker.color.r = BASE_TREE_COLOR_R
        trunk_marker.color.g = BASE_TREE_COLOR_G
        trunk_marker.color.b = BASE_TREE_COLOR_B
        trunk_marker.color.a = BASE_TREE_COLOR_A
        markers.markers.append(trunk_marker)"""

        # --- Calculate Indicator Cylinder Heights ---
        red_height = INDICATOR_MIN_HEIGHT
        green_height = INDICATOR_MIN_HEIGHT

        indicator_base_z = BASE_TREE_HEIGHT

        base = INDICATOR_MIN_HEIGHT
        span = VARIABLE_HEIGHT_RANGE

        p_red   = max(INDICATOR_MIN_HEIGHT, min(1.0, lambda_val[0]))
        p_green = max(INDICATOR_MIN_HEIGHT, min(1.0, lambda_val[1]))

        red_height   = base + span * p_red
        green_height = base + span * p_green

        # --- Red Indicator Cylinder ---
        red_marker = Marker()
        red_marker.header.frame_id = 'map'
        red_marker.header.stamp = rospy.Time.now()
        red_marker.ns = "tree_lambda_indicators_red"
        red_marker.id = i * 3 + 1
        red_marker.type = Marker.CYLINDER
        red_marker.action = Marker.ADD
        red_marker.pose.position = Point(
            x=tree_pos[0] - INDICATOR_OFFSET_X,
            y=tree_pos[1],
            z=indicator_base_z + red_height / 2.0
        )
        red_marker.pose.orientation.w = 1.0
        red_marker.scale.x = INDICATOR_DIAMETER
        red_marker.scale.y = INDICATOR_DIAMETER
        red_marker.scale.z = red_height
        red_marker.color.r = 0.8588235294117647
        red_marker.color.g = 0.37254901960784315
        red_marker.color.b = 0.3411764705882353
        red_marker.color.a = 1.0
        markers.markers.append(red_marker)

        # --- Green Indicator Cylinder ---
        green_marker = Marker()
        green_marker.header.frame_id = 'map'
        green_marker.header.stamp = rospy.Time.now()
        green_marker.ns = "tree_lambda_indicators_green"
        green_marker.id = i * 3 + 2
        green_marker.type = Marker.CYLINDER
        green_marker.action = Marker.ADD
        green_marker.pose.position = Point(
            x=tree_pos[0] + INDICATOR_OFFSET_X,
            y=tree_pos[1],
            z=indicator_base_z + green_height / 2.0
        )
        green_marker.pose.orientation.w = 1.0
        green_marker.scale.x = INDICATOR_DIAMETER
        green_marker.scale.y = INDICATOR_DIAMETER
        green_marker.scale.z = green_height
        green_marker.color.r = 0.6980392156862745
        green_marker.color.g = 0.8745098039215686
        green_marker.color.b = 0.5411764705882353
        green_marker.color.a = 1.0
        markers.markers.append(green_marker)
    return markers

def update_robot_state(buffer):
    robot_pose = []
    while not len(robot_pose):
        try:
            trans = buffer.lookup_transform('map', 'drone_base_link', rospy.Time())
            (_, _, yaw) = tf.transformations.euler_from_quaternion([ trans.transform.rotation.x,  trans.transform.rotation.y,  trans.transform.rotation.z,  trans.transform.rotation.w])
            
            robot_pose = [[trans.transform.translation.x], [trans.transform.translation.y], [yaw]]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get transform: {e}")
        
    return robot_pose
