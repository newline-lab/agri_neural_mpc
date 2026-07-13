#!/usr/bin/env python3
"""
car_mapper_node.py — ROS1 Node
================================
Derived from parking_map_node.py, but WITHOUT the slot map: instead of
voting on predefined parking slots, it simply localizes every detected car
in world coordinates and records its orientation class (front/back).

Reused parts from parking_map_node.py:
  - YOLO + depth median filtering        (get_box_median_depth, filter_detections_by_depth)
  - bearing gate near image center
  - project_detection_to_world
  - RGB/Depth sync + GPS subscriber structure

New parts:
  - CarTrack: online clustering of projected detections. Detections that
    fall within ~cluster_radius_m of an existing track are merged
    (confidence-weighted running average of position, class votes).
  - Robot trajectory logging.
  - Final XY matplotlib plot:
      * robot trajectory (line)
      * one point per recognized car
      * an arrow at each car:
          front → arrow points TOWARD the robot path (toward where the
                  robot was when it saw the car)
          back  → arrow points AWAY from the robot path

Published topics:
  /car_map/state             (std_msgs/String)  — JSON of all car tracks
  /car_map/detections_debug  (sensor_msgs/Image) — RGB + YOLO boxes overlay

Parameters:
  ~weights           : path to YOLO best.pt              (required)
  ~side              : "left" or "right"                 (default: "right")
  ~max_depth_m       : max valid depth in meters         (default: 4.5)
  ~gps_range_m       : max robot-to-car distance in m    (default: 3.0)
                       (world-frame gate; detections whose projected
                       position is farther than this from the robot's
                       GPS position are discarded)
  ~conf_thresh       : YOLO confidence threshold         (default: 0.5)
  ~max_bearing_deg   : bearing gate half-angle           (default: 15.0)
  ~cluster_radius_m  : merge radius for car tracks       (default: 1.0)
  ~min_track_score   : min summed confidence for a track (default: 2.0)
                       to be kept in the final map
  ~hfov_deg          : camera horizontal FOV             (default: 69.0)
  ~output_dir        : where to save final plot/JSON     (default: ".")

Output files (on shutdown):
  <output_dir>/car_map_final.png   — XY plot (trajectory + cars + arrows)
  <output_dir>/car_map_final.json  — car positions, classes, orientations

Requirements:
    pip install ultralytics opencv-python numpy matplotlib
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")          # headless-safe
import matplotlib.pyplot as plt

try:
    import rospy
    import message_filters
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Pose2D
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ── Class constants (same trained model as parking_map_node) ────────────────
CLS_BACK  = 0
CLS_FRONT = 1
CLS_NAMES = {CLS_BACK: "car_back", CLS_FRONT: "car_front"}


# ─────────────────────────────────────────────────────────────────────────────
# Reused pure functions (copied from parking_map_node.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_box_median_depth(depth_img: np.ndarray, x1: int, y1: int,
                         x2: int, y2: int,
                         depth_scale: float = 0.001) -> Optional[float]:
    """Median depth (m) of valid pixels inside the bounding box."""
    h, w = depth_img.shape[:2]
    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(w, x2); y2c = min(h, y2)
    crop = depth_img[y1c:y2c, x1c:x2c].astype(np.float32)
    valid = crop[crop > 0] * depth_scale
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def filter_detections_by_depth(detections: np.ndarray,
                               depth_img: np.ndarray,
                               max_depth_m: float = 4.5
                               ) -> List[Tuple[np.ndarray, float]]:
    """Keep detections with valid depth <= max_depth_m, closest first."""
    if len(detections) == 0:
        return []
    candidates = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        depth = get_box_median_depth(depth_img, int(x1), int(y1),
                                     int(x2), int(y2))
        if depth is None or depth > max_depth_m:
            continue
        candidates.append((det, depth))
    candidates.sort(key=lambda t: t[1])
    return candidates


def project_detection_to_world(det_row, depth_m,
                               robot_x, robot_y, robot_theta,
                               side, img_width,
                               hfov_deg=69.0):
    """World (x, y) of a detected car (camera perpendicular to travel)."""
    x1, _, x2, _ = det_row[:4]
    u = (x1 + x2) / 2.0
    bearing = math.radians(((u / img_width) - 0.5) * hfov_deg)

    side_offset = math.pi / 2 if side == "left" else -math.pi / 2
    angle = robot_theta + side_offset - (bearing if side == "left" else -bearing)

    cx = robot_x + depth_m * math.cos(angle)
    cy = robot_y + depth_m * math.sin(angle)
    return cx, cy


# ─────────────────────────────────────────────────────────────────────────────
# Car track: online clustering of projected detections
# ─────────────────────────────────────────────────────────────────────────────

class CarTrack:
    """
    One physical car, built from multiple detections. Position is a
    confidence-weighted running mean; class is decided by confidence-weighted
    majority between front/back votes. The robot pose at each observation is
    also averaged so the arrow can point toward/away from where the robot
    actually was when it saw the car.
    """
    _next_id = 0

    def __init__(self, x: float, y: float, cls_id: int, conf: float,
                 robot_x: float, robot_y: float):
        self.id = CarTrack._next_id
        CarTrack._next_id += 1

        self.x = x
        self.y = y
        self.weight = conf                 # sum of confidences (position weight)

        self.front_conf = conf if cls_id == CLS_FRONT else 0.0
        self.back_conf  = conf if cls_id == CLS_BACK  else 0.0
        self.n_obs = 1

        # Weighted mean of robot positions at observation time
        self.obs_rx = robot_x * conf
        self.obs_ry = robot_y * conf

    def add(self, x: float, y: float, cls_id: int, conf: float,
            robot_x: float, robot_y: float):
        # Confidence-weighted running mean of position
        w_new = self.weight + conf
        self.x = (self.x * self.weight + x * conf) / w_new
        self.y = (self.y * self.weight + y * conf) / w_new
        self.weight = w_new

        if cls_id == CLS_FRONT:
            self.front_conf += conf
        else:
            self.back_conf += conf
        self.n_obs += 1

        self.obs_rx += robot_x * conf
        self.obs_ry += robot_y * conf

    # ── Resolution ────────────────────────────────────────────────────────
    @property
    def cls_id(self) -> int:
        return CLS_FRONT if self.front_conf >= self.back_conf else CLS_BACK

    @property
    def score(self) -> float:
        return self.front_conf + self.back_conf

    @property
    def mean_conf(self) -> float:
        return self.score / self.n_obs

    def mean_robot_pos(self) -> Tuple[float, float]:
        return self.obs_rx / self.weight, self.obs_ry / self.weight

    def orientation_angle(self) -> float:
        """
        Arrow angle (rad, world frame):
          front → points from the car TOWARD the (mean) robot position
          back  → points AWAY from the robot position
        """
        rx, ry = self.mean_robot_pos()
        to_robot = math.atan2(ry - self.y, rx - self.x)
        return to_robot if self.cls_id == CLS_FRONT else to_robot + math.pi

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "x":           round(self.x, 3),
            "y":           round(self.y, 3),
            "class":       CLS_NAMES[self.cls_id],
            "orientation_rad": round(self.orientation_angle(), 4),
            "orientation_deg": round(math.degrees(self.orientation_angle()), 2),
            "n_obs":       self.n_obs,
            "score":       round(self.score, 3),
            "mean_conf":   round(self.mean_conf, 3),
            "votes": {"front_conf": round(self.front_conf, 3),
                      "back_conf":  round(self.back_conf, 3)},
        }


    def front_position(self, front_offset_m: float = 0.0) -> Tuple[float, float]:
        """
        Reference point at the car's FRONT, regardless of which side was
        detected. 'front' tracks are already correct (returned as-is).
        'back' tracks are shifted by front_offset_m further along the
        robot→car ray (the same direction used at projection time), since
        the camera saw the back face and the front is farther away.
        """
        if self.cls_id == CLS_FRONT or front_offset_m == 0.0:
            return self.x, self.y
        rx, ry = self.mean_robot_pos()
        ray_angle = math.atan2(self.y - ry, self.x - rx)   # robot → car direction
        fx = self.x + front_offset_m * math.cos(ray_angle)
        fy = self.y + front_offset_m * math.sin(ray_angle)
        return fx, fy


def add_detection_to_tracks(tracks: List[CarTrack],
                            cx: float, cy: float,
                            cls_id: int, conf: float,
                            robot_x: float, robot_y: float,
                            cluster_radius_m: float = 1.0) -> CarTrack:
    """Merge a projected detection into the nearest track, or spawn a new one."""
    if tracks:
        nearest = min(tracks, key=lambda t: math.hypot(t.x - cx, t.y - cy))
        if math.hypot(nearest.x - cx, nearest.y - cy) <= cluster_radius_m:
            nearest.add(cx, cy, cls_id, conf, robot_x, robot_y)
            return nearest
    t = CarTrack(cx, cy, cls_id, conf, robot_x, robot_y)
    tracks.append(t)
    return t



# ─────────────────────────────────────────────────────────────────────────────
# Final XY plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_car_map(trajectory: List[Tuple[float, float]],
                 tracks: List[CarTrack],
                 out_path: str,
                 min_track_score: float = 2.0,
                 arrow_len: float = 0.8):
    """
    XY plot: robot trajectory + one marker per recognized car, with an arrow
    pointing toward the robot path (front) or away from it (back).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if trajectory:
        traj = np.asarray(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], "-", color="tab:orange",
                lw=1.5, label="robot trajectory", zorder=1)
        ax.plot(traj[0, 0], traj[0, 1], "o", color="tab:orange", ms=8)
        ax.annotate("start", (traj[0, 0], traj[0, 1]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    kept = [t for t in tracks if t.score >= min_track_score]
    for t in kept:
        is_front = (t.cls_id == CLS_FRONT)
        col = "tab:green" if is_front else "tab:red"
        ax.plot(t.x, t.y, "s", color=col, ms=9, zorder=3)

        ang = t.orientation_angle()
        ax.annotate(
            "", xy=(t.x + arrow_len * math.cos(ang),
                    t.y + arrow_len * math.sin(ang)),
            xytext=(t.x, t.y),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=2),
            zorder=4)

        ax.annotate(f"#{t.id} {CLS_NAMES[t.cls_id]}\n({t.mean_conf:.2f})",
                    (t.x, t.y), textcoords="offset points",
                    xytext=(8, -14), fontsize=7)

    # Legend proxies
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="tab:orange", lw=1.5, label="robot trajectory"),
        Line2D([], [], marker="s", ls="", color="tab:green",
               label="car_front (arrow → robot)"),
        Line2D([], [], marker="s", ls="", color="tab:red",
               label="car_back (arrow ← robot)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Detected cars ({len(kept)}) and robot trajectory")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# ROS Node
# ─────────────────────────────────────────────────────────────────────────────

class CarMapperNode:

    def __init__(self):
        if not _ROS_AVAILABLE:
            raise RuntimeError("CarMapperNode requires ROS1 (rospy etc.).")
        if not _ULTRALYTICS_AVAILABLE:
            raise RuntimeError("CarMapperNode requires 'ultralytics'.")
        rospy.init_node("car_mapper_node", anonymous=False)

        weights_path        = rospy.get_param("~weights")
        self.side           = rospy.get_param("~side",             "right")
        self.max_depth_m    = rospy.get_param("~max_depth_m",      4.0)
        self.gps_range_m    = rospy.get_param("~gps_range_m",      5.0)
        self.conf_thresh    = rospy.get_param("~conf_thresh",      0.5)
        self.max_bearing_deg = rospy.get_param("~max_bearing_deg", 15.0)
        self.cluster_radius_m = rospy.get_param("~cluster_radius_m", 1.0)
        self.min_track_score  = rospy.get_param("~min_track_score",  50.0)
        self.hfov_deg       = rospy.get_param("~hfov_deg",         69.0)
        self.front_offset_m = rospy.get_param("~front_offset_m", 3.7)
        self.output_dir     = rospy.get_param("~output_dir",       ".")

        if self.side not in ("left", "right"):
            raise ValueError(f"~side must be 'left' or 'right', got {self.side!r}")

        rospy.loginfo(f"[CarMap] Loading YOLO weights: {weights_path}")
        self.model = YOLO(weights_path)
        rospy.loginfo("[CarMap] YOLO model loaded.")

        self.bridge = CvBridge()
        self.lock   = threading.Lock()
        self.robot_x = self.robot_y = self.robot_theta = None

        self.tracks: List[CarTrack] = []
        self.trajectory: List[Tuple[float, float]] = []
        self._traj_min_step = 0.05      # m: don't log duplicate poses

        self.pub_state = rospy.Publisher("/car_map/state", String, queue_size=1)
        self.pub_debug = rospy.Publisher("/car_map/detections_debug",
                                         Image, queue_size=1)

        rospy.Subscriber("/gps_data", Pose2D, self._gps_cb, queue_size=10)

        rgb_sub   = message_filters.Subscriber("/camera/color/image_raw", Image)
        depth_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self._rgbd_cb)

        rospy.Timer(rospy.Duration(1.0), self._publish_state)
        rospy.loginfo("[CarMap] Node ready. Waiting for data...")

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _gps_cb(self, msg: Pose2D):
        with self.lock:
            self.robot_x, self.robot_y, self.robot_theta = msg.x, msg.y, msg.theta
            if (not self.trajectory or
                    math.hypot(msg.x - self.trajectory[-1][0],
                               msg.y - self.trajectory[-1][1])
                    >= self._traj_min_step):
                self.trajectory.append((msg.x, msg.y))

    def _rgbd_cb(self, rgb_msg: Image, depth_msg: Image):
        try:
            rgb_img   = self.bridge.imgmsg_to_cv2(rgb_msg,   "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            rospy.logerr(f"[CarMap] Image conversion error: {e}")
            return

        results = self.model(rgb_img, conf=self.conf_thresh, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        valid = filter_detections_by_depth(detections, depth_img,
                                           max_depth_m=self.max_depth_m)

        # ── Debug overlay ────────────────────────────────────────────────
        if _CV2_AVAILABLE and self.pub_debug.get_num_connections() > 0:
            self._publish_debug(rgb_img, depth_img, detections, valid)

        with self.lock:
            rx, ry, rtheta = self.robot_x, self.robot_y, self.robot_theta
        if rx is None:
            rospy.logwarn_throttle(5.0, "[CarMap] Waiting for first GPS message...")
            return

        img_w = rgb_img.shape[1]
        for det_row, depth_m in valid:
            # Bearing gate: only trust detections near image center
            x1, _, x2, _ = det_row[:4]
            u = (x1 + x2) / 2.0
            bearing_deg = ((u / img_w) - 0.5) * self.hfov_deg
            if abs(bearing_deg) > self.max_bearing_deg:
                continue

            cx, cy = project_detection_to_world(
                det_row, depth_m, rx, ry, rtheta,
                self.side, img_w, hfov_deg=self.hfov_deg)

            # GPS-range gate: discard cars whose projected world position
            # is too far from the robot's current GPS position. Combined
            # with the depth gate (max_depth_m) in filter_detections_by_depth,
            # only cars inside BOTH limits are considered.
            robot_dist = math.hypot(cx - rx, cy - ry)
            if robot_dist > self.gps_range_m:
                rospy.logdebug(
                    f"[CarMap] Detection discarded: robot_dist={robot_dist:.2f}m "
                    f"> gps_range_m={self.gps_range_m:.2f}m")
                continue

            cls_id = int(det_row[5])
            conf   = float(det_row[4])
            with self.lock:
                track = add_detection_to_tracks(
                    self.tracks, cx, cy, cls_id, conf, rx, ry,
                    cluster_radius_m=self.cluster_radius_m)
            rospy.loginfo(
                f"[CarMap] Track {track.id}: +{CLS_NAMES[cls_id]} "
                f"(conf={conf:.2f}) → pos=({track.x:.2f},{track.y:.2f}), depth={depth_m:.2f}m, "
                f"n_obs={track.n_obs}, score={track.score:.2f}")

    def _publish_debug(self, rgb_img, depth_img, detections, valid):
        dbg = rgb_img.copy()
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            col = (0, 200, 0) if cls_id == CLS_FRONT else (0, 0, 220)
            depth = get_box_median_depth(depth_img, int(x1), int(y1),
                                         int(x2), int(y2))
            depth_str = f"{depth:.2f}m" if depth is not None else "no depth"
            label = f"{CLS_NAMES.get(cls_id, '?')} {conf:.2f} {depth_str}"
            cv2.rectangle(dbg, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(dbg, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        for det_row, _ in valid:
            bx1, by1, bx2, by2 = det_row[:4].astype(int)
            u = (bx1 + bx2) / 2.0
            bearing_deg = ((u / rgb_img.shape[1]) - 0.5) * self.hfov_deg
            gated = abs(bearing_deg) > self.max_bearing_deg
            col = (128, 128, 128) if gated else (255, 255, 255)
            tag = f"{'GATED' if gated else 'VALID'} {bearing_deg:+.0f}deg"
            cv2.rectangle(dbg, (bx1, by1), (bx2, by2), col, 3)
            cv2.putText(dbg, tag, (bx1, by2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        try:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))
        except Exception as e:
            rospy.logerr(f"[CarMap] Debug publish error: {e}")

    # ── Periodic state publishing ──────────────────────────────────────────

    def _publish_state(self, event):
        with self.lock:
            state = {
                "robot": {"x": self.robot_x, "y": self.robot_y},
                "cars": [t.to_dict() for t in self.tracks],
            }
        self.pub_state.publish(String(data=json.dumps(state)))

    # ── Shutdown: save plot + JSON ─────────────────────────────────────────

    def save_final_map(self):
        with self.lock:
            tracks = list(self.tracks)
            traj   = list(self.trajectory)

        front_offset_m = self.front_offset_m   # set this param, value TBD
        for t in tracks:
            t.x, t.y = t.front_position(front_offset_m)

        # ── Reassign contiguous IDs to KEPT cars only ──────────────────────
        # Track ids are originally assigned in discovery order
        # (CarTrack._next_id, incrementing as new tracks are spawned during
        # the run). Once low-score tracks are discarded below, the surviving
        # cars can end up with sparse ids (e.g. {0, 3, 4} if ids 1, 2 were
        # discarded) — these sparse ids then get consumed downstream by
        # car_data_association_node, where they used to cause the score
        # array to be sized by max(id)+1 instead of the actual car count,
        # leaving permanent zero-filled phantom columns for the missing ids.
        # Renumbering here, once, at the point of finalization, means kept
        # cars always have clean, contiguous ids 0..N-1 with no gaps.
        # Sorted by x for a stable, intuitive left-to-right ordering along
        # the row (discarded cars are NOT renumbered — they're excluded from
        # every downstream consumer, so their original ids are irrelevant
        # and kept only for cross-referencing in the JSON's
        # "discarded_low_score" section).
        kept = [t for t in tracks if t.score >= self.min_track_score]
        kept.sort(key=lambda t: t.x)
        for new_id, t in enumerate(kept):
            t.id = new_id

        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        png_path = str(out / "car_map_final.png")
        plot_car_map(traj, tracks, png_path,
                     min_track_score=self.min_track_score)
        rospy.loginfo(f"[CarMap] Final map plot saved: {png_path}")

        state = {
            "trajectory": [[round(x, 3), round(y, 3)] for x, y in traj],
            "cars": [t.to_dict() for t in kept],
            #"discarded_low_score": [t.to_dict() for t in tracks
            #                        if t.score < self.min_track_score],
        }
        json_path = str(out / "car_map_final.json")
        with open(json_path, "w") as f:
            json.dump(state, f, indent=2)
        rospy.loginfo(f"[CarMap] Final map JSON saved: {json_path} "
                      f"({len(kept)} cars kept, "
                      f"{len(tracks) - len(kept)} discarded)")


if __name__ == "__main__":
    node = CarMapperNode()
    rospy.on_shutdown(node.save_final_map)
    rospy.spin()
