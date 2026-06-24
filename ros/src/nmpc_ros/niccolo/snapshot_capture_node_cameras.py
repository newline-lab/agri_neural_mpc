#!/usr/bin/env python3

import os
import sys
import cv2
import yaml
import rospy
import select
import termios
import tty
from datetime import datetime

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, NavSatFix
from apriltag_ros.msg import AprilTagDetectionArray

from geometry_msgs.msg import PoseStamped
from cube_pose_estimator.msg import CubePoseArray
from std_msgs.msg import Float64

#from pynput import mouse


class SnapshotCaptureNode:
    def __init__(self):
        rospy.init_node("snapshot_capture_node")

        self.snapshot_counter = 1

        self.bridge = CvBridge()
        self.latest_msgs = {}
        self.capture_requested = False
        self.capture_in_progress = False
        self.capture_t_ref = None

        self.output_root = rospy.get_param(
            "~output_root",
            os.path.expanduser("~/apriltag_snapshots")
        )

        self.auto_capture_rate = rospy.get_param("~auto_capture_rate", 0.0)  # Hz, 0 = disabled
        self.last_auto_capture_time = rospy.Time(0)

        self.topics = {
            "/cam_up/color/image_raw": Image,
            "/cam_up/color/camera_info": CameraInfo,
            "/cam_up/depth/image_rect_raw": Image,
            "/cam_up/depth/camera_info": CameraInfo,
            "/tag_detections": AprilTagDetectionArray,
            "/cube_pose/tag10": PoseStamped,
            "/cube_pose/tag11": PoseStamped,
            "/cube_pose/tag12": PoseStamped,
            "/cube_pose/tag13": PoseStamped,
            "/cube_pose/tag14": PoseStamped,
            "/cube_pose/fused_pose": PoseStamped,
            "/cube_pose/azimuth_deg": Float64,
            "/fix": NavSatFix,
        }

        self.subs = []
        for topic, msg_type in self.topics.items():
            self.subs.append(
                rospy.Subscriber(topic, msg_type, self.snapshot_callback, callback_args=topic, queue_size=1)
            )


        """
        self.mouse_listener = mouse.Listener(
            on_click=self.on_click
        )
        self.mouse_listener.start()
        """

        os.makedirs(self.output_root, exist_ok=True)

        rospy.loginfo("snapshot_capture_node started.")

        if self.auto_capture_rate > 0:
            rospy.loginfo("Auto-capture enabled at %.2f Hz. Press 'q' to quit.", self.auto_capture_rate)
        else:
            rospy.loginfo("Press 'c' to capture snapshot, 'q' to quit.")

    def snapshot_callback(self, msg, topic_name):
        self.latest_msgs[topic_name] = msg

        # If user requested a capture, anchor it on the NEXT /cube_pose/all frame
        if topic_name == "/cube_pose/fused_pose" and self.capture_requested and not self.capture_in_progress:
            self.capture_in_progress = True
            self.capture_t_ref = msg.header.stamp
            self.capture_snapshot()
            self.capture_requested = False
            self.capture_in_progress = False

    def is_close_in_time(self, msg, t_ref, tol=0.05):
        if not hasattr(msg, "header"):
            return False
        dt = abs((msg.header.stamp - t_ref).to_sec())
        return dt <= tol

    def msg_to_yaml_file(self, msg, filepath):
        with open(filepath, "w") as f:
            f.write(str(msg))
            
    def write_metadata_yaml(self, snapshot_dir, timestamp, msgs_to_save):
        metadata = {
            "snapshot_timestamp": timestamp,
            "ros_time_now": rospy.Time.now().to_sec(),
            "available_topics": sorted(list(msgs_to_save.keys())),
            "visible_tag_ids": [],
            "num_visible_tags": 0,
            "color_image": {},
            "depth_image": {},
            "camera_info": {},
            "gps": {},
            "fused_pose": {},
            "azimuth_deg": None,
        }

        if "/tag_detections" in msgs_to_save:
            det_msg = msgs_to_save["/tag_detections"]
            ids = []
            for det in det_msg.detections:
                if len(det.id) > 0:
                    ids.append(int(det.id[0]))
            metadata["visible_tag_ids"] = ids

        metadata["num_visible_tags"] = len(metadata["visible_tag_ids"])

        if "/cube_pose/fused_pose" in msgs_to_save:
            fused_msg = msgs_to_save["/cube_pose/fused_pose"]
            metadata["fused_pose"] = {
                "frame_id": fused_msg.header.frame_id,
                "stamp": fused_msg.header.stamp.to_sec(),
                "position": {
                    "x": fused_msg.pose.position.x,
                    "y": fused_msg.pose.position.y,
                    "z": fused_msg.pose.position.z,
                },
                "orientation": {
                    "x": fused_msg.pose.orientation.x,
                    "y": fused_msg.pose.orientation.y,
                    "z": fused_msg.pose.orientation.z,
                    "w": fused_msg.pose.orientation.w,
                },
            }

        if "/cube_pose/azimuth_deg" in msgs_to_save:
            metadata["azimuth_deg"] = msgs_to_save["/cube_pose/azimuth_deg"].data


        # Color image info
        if "/cam_up/color/image_raw" in msgs_to_save:
            img_msg = msgs_to_save["/cam_up/color/image_raw"]
            metadata["color_image"] = {
                "width": img_msg.width,
                "height": img_msg.height,
                "encoding": img_msg.encoding,
                "frame_id": img_msg.header.frame_id,
                "stamp": img_msg.header.stamp.to_sec(),
            }

        # Depth image info
        if "/cam_up/depth/image_rect_raw" in msgs_to_save:
            depth_msg = msgs_to_save["/cam_up/depth/image_rect_raw"]
            metadata["depth_image"] = {
                "width": depth_msg.width,
                "height": depth_msg.height,
                "encoding": depth_msg.encoding,
                "frame_id": depth_msg.header.frame_id,
                "stamp": depth_msg.header.stamp.to_sec(),
            }

        # Camera info
        if "/cam_up/color/camera_info" in msgs_to_save:
            cam_info = msgs_to_save["/cam_up/color/camera_info"]
            metadata["camera_info"]["color"] = {
                "frame_id": cam_info.header.frame_id,
                "stamp": cam_info.header.stamp.to_sec(),
                "width": cam_info.width,
                "height": cam_info.height,
                "K": list(cam_info.K),
                "D": list(cam_info.D),
                "distortion_model": cam_info.distortion_model,
            }

        if "/cam_up/depth/camera_info" in msgs_to_save:
            cam_info = msgs_to_save["/cam_up/depth/camera_info"]
            metadata["camera_info"]["depth"] = {
                "frame_id": cam_info.header.frame_id,
                "stamp": cam_info.header.stamp.to_sec(),
                "width": cam_info.width,
                "height": cam_info.height,
                "K": list(cam_info.K),
                "D": list(cam_info.D),
                "distortion_model": cam_info.distortion_model,
            }
        
        if "/fix" in msgs_to_save:
            gps_msg = msgs_to_save["/fix"]
            metadata["gps"] = {
                "frame_id": gps_msg.header.frame_id,
                "stamp": gps_msg.header.stamp.to_sec(),
                "status": gps_msg.status.status,
                "service": gps_msg.status.service,
                "latitude": gps_msg.latitude,
                "longitude": gps_msg.longitude,
                "altitude": gps_msg.altitude,
                "position_covariance": list(gps_msg.position_covariance),
                "position_covariance_type": gps_msg.position_covariance_type,
            }

        metadata_path = os.path.join(snapshot_dir, "metadata.yaml")
        with open(metadata_path, "w") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)

    def save_color_image(self, msg, filepath):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(filepath, cv_img)

    def save_depth_image(self, msg, filepath_png, filepath_npy):
        cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(filepath_png, cv_depth)
        try:
            import numpy as np
            np.save(filepath_npy, cv_depth)
        except Exception as e:
            rospy.logwarn("Could not save depth .npy: %s", str(e))

    def capture_snapshot(self):
        if self.capture_t_ref is None:
            rospy.logwarn("No reference timestamp available for snapshot.")
            return

        t_ref = self.capture_t_ref
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snapshot_dir = os.path.join(self.output_root, f"snapshot_{self.snapshot_counter:04d}_{timestamp}")
        os.makedirs(snapshot_dir, exist_ok=True)

        msgs_to_save = {}
        
        if "/cube_pose/fused_pose" in self.latest_msgs:
            fused_msg = self.latest_msgs["/cube_pose/fused_pose"]
            if self.is_close_in_time(fused_msg, t_ref, tol=0.1):
                msgs_to_save["/cube_pose/fused_pose"] = fused_msg
            else:
                rospy.logwarn("Skipping /cube_pose/fused (timestamp too old/new).")

        if "/cube_pose/azimuth_deg" in self.latest_msgs:
            msgs_to_save["/cube_pose/azimuth_deg"] = self.latest_msgs["/cube_pose/azimuth_deg"]

        if "/tag_detections" in self.latest_msgs:
            det_msg = self.latest_msgs["/tag_detections"]
            if self.is_close_in_time(det_msg, t_ref, tol=0.03):
                msgs_to_save["/tag_detections"] = det_msg

        for cam_topic in [
            "/cam_up/color/image_raw",
            "/cam_up/color/camera_info",
            "/cam_up/depth/image_rect_raw",
            "/cam_up/depth/camera_info",
            "/fix",
        ]:
            if cam_topic in self.latest_msgs:
                msg = self.latest_msgs[cam_topic]

                # More permissive tolerance for sensors
                if hasattr(msg, "header"):
                    tol = 0.10 if "image" in cam_topic or "camera_info" in cam_topic else 0.20
                    if self.is_close_in_time(msg, t_ref, tol=tol):
                        msgs_to_save[cam_topic] = msg
                    else:
                        rospy.logwarn("Skipping %s (not close enough to snapshot time).", cam_topic)
                else:
                    msgs_to_save[cam_topic] = msg

        for topic, msg in msgs_to_save.items():
            safe_name = topic.strip("/").replace("/", "__")

            if topic == "/cam_up/color/image_raw":
                try:
                    self.save_color_image(msg, os.path.join(snapshot_dir, f"{safe_name}.png"))
                except Exception as e:
                    rospy.logwarn("Failed saving color image: %s", str(e))
                    self.msg_to_yaml_file(msg, os.path.join(snapshot_dir, f"{safe_name}.txt"))

            elif topic == "/cam_up/depth/image_rect_raw":
                try:
                    self.save_depth_image(
                        msg,
                        os.path.join(snapshot_dir, f"{safe_name}.png"),
                        os.path.join(snapshot_dir, f"{safe_name}.npy"),
                    )
                except Exception as e:
                    rospy.logwarn("Failed saving depth image: %s", str(e))
                    self.msg_to_yaml_file(msg, os.path.join(snapshot_dir, f"{safe_name}.txt"))

            else:
                self.msg_to_yaml_file(msg, os.path.join(snapshot_dir, f"{safe_name}.txt"))

        self.write_metadata_yaml(snapshot_dir, timestamp,msgs_to_save)

        rospy.loginfo("Synchronized snapshot saved in: %s", snapshot_dir)

        self.snapshot_counter += 1

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def on_click(self, x, y, button, pressed):
        if pressed and not self.capture_requested and not self.capture_in_progress:
            self.capture_requested = True
            rospy.loginfo("Capture requested via mouse click.")

    def run(self):
        rate = rospy.Rate(200)
        while not rospy.is_shutdown():
            key = self.get_key()

            if key is not None:
                if key.lower() == "c":
                    if not self.capture_requested and not self.capture_in_progress:
                        self.capture_requested = True
                        rospy.loginfo("Capture requested via keyboard.")
                elif key.lower() == "q":
                    rospy.loginfo("Exiting snapshot_capture_node.")
                    break

            # Auto-capture logic
            if self.auto_capture_rate > 0 and not self.capture_requested and not self.capture_in_progress:
                interval = rospy.Duration(1.0 / self.auto_capture_rate)
                if (rospy.Time.now() - self.last_auto_capture_time) >= interval:
                    self.capture_requested = True
                    self.last_auto_capture_time = rospy.Time.now()
                    rospy.loginfo("Auto-capture triggered.")

            rate.sleep()


if __name__ == "__main__":
    try:
        node = SnapshotCaptureNode()
        node.run()
    except rospy.ROSInterruptException:
        pass