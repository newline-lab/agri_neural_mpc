import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from image_geometry import PinholeCameraModel
from scipy.spatial import cKDTree
from nmpc_ros.srv import GetTreesPoses
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf
import tf.transformations as tf_trans
import message_filters

def weight_value(n_elements, mean_score, midpoint=5., steepness=10.):
    val = (mean_score - 0.5) * (0.5 + 0.5*np.tanh(steepness*(n_elements - midpoint)))
    return np.round(val, 2)

class DataAssociationNode:
    def __init__(self):
        rospy.init_node('bounding_box_3d_pose', anonymous=True)

        self.bridge = CvBridge()
        self.camera_info = None
        self.camera_matrix = None
        self.depth_image = None
        self.tree_poses = None
        self.publish_visualization = rospy.get_param('~publish_visualization', True)

        detection_sub = message_filters.Subscriber("/yolov7/detect", Detection2DArray)
        depth_image_sub = message_filters.Subscriber("camera/depth/image/compressed", CompressedImage)

        self.ts = message_filters.ApproximateTimeSynchronizer([detection_sub, depth_image_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.synchronized_callback)

        self.camera_info_sub = rospy.Subscriber("camera/depth/camera_info", CameraInfo, self.camera_info_callback)
        self.scores_pub = rospy.Publisher("tree_scores", Float32MultiArray, queue_size=1)
        
        if self.publish_visualization:
            self.marker_scores_pub = rospy.Publisher("scores_markers", MarkerArray, queue_size=1)
            self.marker_fruits_pub = rospy.Publisher("fruits_markers", MarkerArray, queue_size=1)
        
        self.cam_model = PinholeCameraModel()
        self.tf_listener = tf.TransformListener()
        
        rospy.wait_for_service('/obj_pose_srv')
        self.get_trees_poses = rospy.ServiceProxy('/obj_pose_srv', GetTreesPoses)
        self.update_tree_poses()
        
        rospy.spin()

    def update_tree_poses(self):
        try:
            response = self.get_trees_poses()
            self.tree_poses = np.array([[pose.position.x, pose.position.y] for pose in response.trees_poses.poses])
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.camera_matrix = np.array(self.camera_info.K).reshape(3, 3)
        self.cam_model.fromCameraInfo(msg)
    

    def uint8_to_distance(self, value, min_dist, max_dist):
        value = max(0, min(value, 255))
        fraction = value / 255.0
        distance = max_dist - fraction * (max_dist - min_dist)
        return distance
    
    def associate_fruits_to_trees(self, fruit_positions, fruit_classes, fruit_scores):
        if self.tree_poses is None or len(fruit_positions) == 0:
            return {}
        
        tree_kdtree = cKDTree(self.tree_poses)
        distances, tree_indices = tree_kdtree.query(fruit_positions[:, :2])

        tree_fruit_dict = {i: {"ripe": [], "raw": []} for i in range(len(self.tree_poses))}
        
        for fruit_index, (tree_index, distance) in enumerate(zip(tree_indices, distances)):
            if distance <= 2.5:
                fruit_class = "ripe" if fruit_classes[fruit_index] == "ripe" else "raw"
                tree_fruit_dict[tree_index][fruit_class].append(fruit_scores[fruit_index])
        
        return tree_fruit_dict

    def transform_fruit_positions(self, fruit_positions, header):
        map_fruits_positions = []
        for fruit_pos in fruit_positions:
            point_camera = PointStamped()
            point_camera.point = Point(*fruit_pos)
            point_camera.header.frame_id = 'depth_camera_frame'
            try:
                point_map = self.tf_listener.transformPoint('map', point_camera)
                map_fruits_positions.append([point_map.point.x, point_map.point.y, point_map.point.z])
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(e)
                continue
        return np.array(map_fruits_positions)
    

    def synchronized_callback(self, detection_msg, depth_image_msg):
        if self.camera_matrix is None or self.tree_poses is None:
            return
        
        try:
            # Extract the depth channel
            self.depth_image = self.bridge.compressed_imgmsg_to_cv2(
                depth_image_msg,
                desired_encoding="passthrough"
            )[:, :, 0]
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        fruit_positions, fruit_scores, fruit_classes = [], [], []

        # Project detections into 3D
        for detection in detection_msg.detections:
            bbox = detection.bbox
            xmin = int(bbox.center.x - bbox.size_x / 2)
            xmax = int(bbox.center.x + bbox.size_x / 2)
            ymin = int(bbox.center.y - bbox.size_y / 2)
            ymax = int(bbox.center.y + bbox.size_y / 2)

            xmin, xmax = max(0, xmin), min(self.depth_image.shape[1], xmax)
            ymin, ymax = max(0, ymin), min(self.depth_image.shape[0], ymax)

            # Sample the center pixel for depth
            cx, cy = int(bbox.center.x), int(bbox.center.y)
            depth_roi = self.depth_image[cy, cx]
            non_zero_depths = depth_roi[depth_roi > 11]
            if len(non_zero_depths) == 0:
                continue

            median_depth = np.median(non_zero_depths)

            
            ray = np.array(self.cam_model.projectPixelTo3dRay((cx, cy)))
            distance = self.uint8_to_distance(median_depth, 0.05, 20)
            XYZ = ray * distance

            fruit_positions.append(XYZ)
            fruit_scores.append(detection.results[0].score)
            fruit_classes.append(
                "ripe" if detection.results[0].id == 2 else "raw"
            )

        fruit_positions = np.array(fruit_positions)

        # Transform to map frame and associate fruits to trees
        map_fruits_positions = self.transform_fruit_positions(
            fruit_positions, detection_msg.header
        )
        associated_fruits = self.associate_fruits_to_trees(
            map_fruits_positions, fruit_classes, fruit_scores
        )

        # Get drone position for distance-based scoring
        try:
            (drone_trans, drone_rot) = self.tf_listener.lookupTransform(
                'map', 'drone_base_link', rospy.Time()
            )
            drone_x, drone_y = drone_trans[0], drone_trans[1]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("Could not get drone position, skipping distance check.")
            drone_x, drone_y = None, None

        # Initialize base scores for each tree
        tree_scores = np.ones(len(self.tree_poses)) * 0.5

        # Update tree scores
        for i, fruits in associated_fruits.items():
            ripe_scores = fruits.get("ripe", [])
            raw_scores  = fruits.get("raw", [])
            tree_x, tree_y = self.tree_poses[i]

            if drone_x is not None:
                dist = np.hypot(drone_x - tree_x, drone_y - tree_y)
                if dist < 8:
                    ripe_value = weight_value(
                        len(ripe_scores), np.mean(ripe_scores) if ripe_scores else 0
                    )
                    raw_value = weight_value(
                        len(raw_scores), np.mean(raw_scores) if raw_scores else 0
                    )
                    tree_scores[i] = ripe_value - raw_value + 0.5

        # Build a 2-column array: [score, -score] per tree
        scores_with_neg = np.stack([tree_scores, 1-tree_scores], axis=1)  # shape (N,2)

        # Publish as a 2D multiarray
        msg = Float32MultiArray()
        dim0 = MultiArrayDimension(
            label="tree",
            size=scores_with_neg.shape[0],
            stride=scores_with_neg.shape[0] * scores_with_neg.shape[1]
        )
        dim1 = MultiArrayDimension(
            label="type",
            size=2,
            stride=scores_with_neg.shape[1]
        )
        msg.layout.dim = [dim0, dim1]
        msg.data = scores_with_neg.flatten().tolist()

        self.scores_pub.publish(msg)

        
        if self.publish_visualization:
            markers = MarkerArray()
            for i, (fruit_pos, score) in enumerate(zip(map_fruits_positions, fruit_scores)):
                fruit_marker = Marker()
                fruit_marker.header = detection_msg.header
                fruit_marker.header.frame_id = 'map'
                fruit_marker.ns = "fruit_markers"
                fruit_marker.id =  len(self.tree_poses) * 2 + i * 2
                fruit_marker.type = Marker.SPHERE
                fruit_marker.action = Marker.MODIFY
                fruit_marker.pose.position.x = fruit_pos[0]
                fruit_marker.pose.position.y = fruit_pos[1]
                fruit_marker.pose.position.z = fruit_pos[2]
                fruit_marker.pose.orientation.w = 1.0
                fruit_marker.lifetime = rospy.Duration(0.2)
                fruit_marker.scale.x = fruit_marker.scale.y = fruit_marker.scale.z = 0.1
                fruit_marker.color.a = 1.0
                fruit_marker.color.r = 1.0 if fruit_classes[i] == "ripe" else 0.0
                fruit_marker.color.g = 1.0 if fruit_classes[i] == "raw" else 0.0
                fruit_marker.color.b = 0.0
                markers.markers.append(fruit_marker)
            self.marker_fruits_pub.publish(markers)

if __name__ == '__main__':
    try:
        DataAssociationNode()
    except rospy.ROSInterruptException:
        pass
