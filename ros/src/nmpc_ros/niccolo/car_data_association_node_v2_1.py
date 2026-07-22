#!/usr/bin/env python3
"""

Online loop: at every step k, detect car+person with YOLOv7 (COCO),
check person-in-car containment, PROJECT each live detection to world
coordinates (same depth + bearing geometry as car_mapper_node) and match
it to the nearest MAPPED CAR by position (within ~assoc_max_m), then
update the per-car score array in [-1, +1]:

    person in car : scores[id] = +(conf_car + conf_person)/2   in (0, +1]
    car only      : scores[id] = -conf_car                     in [-1, 0)
    no detection  : score unchanged (optional EMA via ~ema_alpha)

Published topics:
  /parking/scores        std_msgs/Float32MultiArray (per-car scores, index=car id)
  /parking/scores_json   std_msgs/String  (rich: id, score, tag, position)
  /gps_data              geometry_msgs/Pose2D   (only if ~gps_source:=serial)
  /gps/rtk_quality       std_msgs/Int32         (only if ~gps_source:=serial)
  /data_association/debug_image  sensor_msgs/Image

Required params:
  ~weights        COCO yolov7 .pt
  ~origin_lat, ~origin_lon   local-frame anchor (only for serial GPS mode;
                  must be the SAME anchor used by car_mapper_node:
                  41.85626425142204, 12.469038428190489 per your file)
  One of:
    ~car_map_source:=file        + ~car_map_file  path to car_map_final.json
    ~car_map_source:=hardcoded   uses HARDCODED_CARS below, no I/O at all
    ~car_map_source:=topic       (default) subscribes to /car_map/state

Main optional params:
  ~car_map_source     "topic" | "file" | "hardcoded"   (default "topic")
  ~car_map_file       path to car_map_final.json  (if source=file)
  ~gps_source         "serial" (default) or "topic"
  ~gps_topic          /gps_data
  ~side               "left" | "right"   (default "right", as mapper)
  ~hfov_deg 69.0, ~max_bearing_deg 15.0  (same values as car_mapper_node)
  ~assoc_max_m 1.0    max projected-detection→mapped-car match distance
  ~containment_thresh 0.7
  ~conf_thresh 0.4, ~iou_thresh 0.45, ~img_size 640, ~device cuda
  ~gps_range_m 4.5, ~max_depth_m 4.5     (site-validated values)
  ~ema_alpha 1.0   (1.0 = pure overwrite; <1 = exponential smoothing)
  ~rgb_topic /cam_up/color/image_raw
  ~depth_topic /cam_up/aligned_depth_to_color/image_raw
  ~serial_port /dev/ttyUSB0, ~baud 115200, ~ntrip_* (as in your client)
"""

import base64
import json
import math
import socket
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

import rospy
import message_filters
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32, String, MultiArrayDimension

import utm

# yolov7 repo root must be importable as top-level 'models' / 'utils'.
# Adjust this path if the yolov7-ros checkout moves.
_YOLOV7_REPO_ROOT = ("/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/yolov7-ros/src")
if _YOLOV7_REPO_ROOT not in sys.path:
    sys.path.insert(0, _YOLOV7_REPO_ROOT)

# yolov7 repo utilities (same imports as your yolov7 ROS node)
from models.experimental import attempt_load
from utils.general import non_max_suppression

# COCO ids for the standard yolov7 weights
COCO_PERSON = 0
COCO_CAR_CLASSES = {2,7}       # car, bus, truck


HARDCODED_CARS: List[dict] = [
    {"id": 0, "x": 42.36, "y": -6.16, "class": "car_back", "orientation_rad": 1.57},
    {"id": 1, "x": 49.53, "y": -7.62, "class": "car_front", "orientation_rad": -1.57},
    {"id": 2, "x": 52.48, "y":  -8.08, "class": "car_front",  "orientation_rad":  -1.57},
]


# ============================================================================
# Mapped cars — replaces the Slot list; loaded from car_mapper_node output
# ============================================================================

class MappedCar:
    """One car from the offline car map (car_mapper_node)."""

    def __init__(self, car_id: int, x: float, y: float,
                 tag: str = "unknown",
                 orientation_rad: Optional[float] = None,
                 lat: Optional[float] = None,
                 lon: Optional[float] = None,
                 visitable: bool = True):
        self.id = car_id
        self.x, self.y = x, y
        self.tag = tag                  
        self.orientation_rad = orientation_rad
        self.lat = lat
        self.lon = lon
        self.visitable = visitable

        self.score_col: int = 0

    def distance_to(self, rx, ry):
        return math.hypot(self.x - rx, self.y - ry)


def cars_from_state(car_dicts: List[dict]) -> List[MappedCar]:
    """Build MappedCar list from car_mapper_node JSON ('cars' array).
    Accepts live /car_map/state, saved car_map_final.json, AND the
    HARDCODED_CARS constant above — all three use the same dict shape."""
    cars = []
    for d in car_dicts:
        cls = d.get("class") or ""
        tag = cls.replace("car_", "") if cls else "unknown"
        visitable = d.get("visitable", True)
        
        cars.append(MappedCar(
            car_id=int(d["id"]),
            x=float(d["x"]),
            y=float(d["y"]),
            tag=tag,
            orientation_rad=d.get("orientation_rad"),
            visitable=visitable
        ))
    cars.sort(key=lambda c: c.id)
    return cars

def local_xy_to_latlon(x: float, y: float,
                       ox: float, oy: float,
                       zone_n: int, zone_l: str) -> Tuple[float, float]:
    easting = x + ox
    northing = y + oy
    lat, lon = utm.to_latlon(easting, northing, zone_n, zone_l)
    return lat, lon

# ============================================================================
# GPS / RTK backend (UM982 + NTRIP), publishes /gps_data Pose2D — unchanged
# ============================================================================

class GpsRtkBackend:
    """Owns the serial port. RTCM download -> serial, GGA upload -> NTRIP.
    Parses GGA (position+quality) and THS/HDT (heading). Publishes
    /gps_data (Pose2D: x, y local metric, theta = ENU yaw rad) so that
    car_mapper_node can use the same topic."""

    def __init__(self, origin_lat: float, origin_lon: float):
        import serial as pyserial
        import utm
        self._serial_mod = pyserial
        self._utm = utm

        self.lock = threading.Lock()
        self.x = self.y = None
        self.theta = 0.0
        self.quality = 0
        self.stamp = rospy.Time(0)

        self.ox, self.oy, self.zone_n, self.zone_l = utm.from_latlon(
            origin_lat, origin_lon)

        port = rospy.get_param("~serial_port", "/dev/ttyUSB0")
        baud = rospy.get_param("~baud", 115200)
        self.server = rospy.get_param("~ntrip_server",
                                      "gnss-rtk.regione.abruzzo.it")
        self.nport = rospy.get_param("~ntrip_port", 2101)
        self.mount = rospy.get_param("~ntrip_mount", "0_RTCM_MSM")
        self.user = rospy.get_param("~ntrip_user", "newline")
        self.password = rospy.get_param("~ntrip_pass", "Newline!")

        self.pub_pose = rospy.Publisher("/gps_data", Pose2D, queue_size=10)
        self.pub_quality = rospy.Publisher("/gps/rtk_quality", Int32,
                                           queue_size=10)

        self.ser = pyserial.Serial(port, baud, timeout=1)
        self.ser.reset_input_buffer()
        self._configure_receiver()
        self.sock = self._connect_ntrip()
        threading.Thread(target=self._serial_loop, daemon=True).start()
        threading.Thread(target=self._rtcm_loop, daemon=True).start()

    def _configure_receiver(self):
        for cmd in ["UNLOGALL\r\n", "GNGGA 0.1\r\n", "GPHDT 0.1\r\n",
                    "INTERFACEMODE COM1 RTCM3 NONE OFF\r\n",
                    "INTERFACEMODE COM2 RTCM3 NONE OFF\r\n",
                    "INTERFACEMODE COM3 RTCM3 NONE OFF\r\n",
                    "SAVECONFIG\r\n"]:
            self.ser.write(cmd.encode("utf-8"))
            time.sleep(0.1)
        rospy.loginfo("[GPS] UM982 configured (GGA + HDT)")

    def _connect_ntrip(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.server, self.nport))
        auth = base64.b64encode(
            f"{self.user}:{self.password}".encode()).decode()
        req = (f"GET /{self.mount} HTTP/1.1\r\n"
               f"User-Agent: NTRIP PythonClient/1.0\r\n"
               f"Authorization: Basic {auth}\r\n"
               f"Accept: */*\r\n\r\n")
        sock.sendall(req.encode())
        resp = sock.recv(1024).decode(errors="ignore")
        if "200 OK" not in resp:
            raise RuntimeError(f"NTRIP refused: {resp.strip()}")
        rospy.loginfo("[GPS] NTRIP connected (%s/%s)", self.server, self.mount)
        return sock

    @staticmethod
    def _nmea_to_deg(value, direction):
        if not value:
            return None
        split = 2 if direction in ("N", "S") else 3
        dec = int(value[:split]) + float(value[split:]) / 60.0
        return -dec if direction in ("S", "W") else dec

    def _serial_loop(self):
        while not rospy.is_shutdown():
            try:
                raw = self.ser.readline()
            except self._serial_mod.SerialException as e:
                rospy.logerr_throttle(5, "[GPS] serial error: %s", e)
                time.sleep(0.5)
                continue
            if b"GGA" in raw:
                self._handle_gga(raw)
                try:
                    self.sock.sendall(raw)          # VRS position upload
                except OSError:
                    pass
            elif b"HDT" in raw or b"THS" in raw:
                self._handle_heading(raw)

    def _handle_gga(self, raw):
        parts = raw.decode("ascii", errors="ignore").strip().split(",")
        if len(parts) < 10:
            return
        lat = self._nmea_to_deg(parts[2], parts[3])
        lon = self._nmea_to_deg(parts[4], parts[5])
        if lat is None or lon is None:
            return
        try:
            quality = int(parts[6]) if parts[6] else 0
        except ValueError:
            return
        ux, uy, _, _ = self._utm.from_latlon(
            lat, lon, force_zone_number=self.zone_n,
            force_zone_letter=self.zone_l)
        with self.lock:
            self.x, self.y = ux - self.ox, uy - self.oy
            self.quality = quality
            self.stamp = rospy.Time.now()
            pose = Pose2D(x=self.x, y=self.y, theta=self.theta)
        self.pub_quality.publish(Int32(quality))
        self.pub_pose.publish(pose)

    def _handle_heading(self, raw):
        parts = raw.decode("ascii", errors="ignore").strip().split(",")
        if len(parts) < 2 or not parts[1]:
            return
        if b"THS" in raw and len(parts) > 2 and \
                parts[2].split("*")[0] == "V":
            return
        try:
            h = float(parts[1])                     # deg CW from true North
        except ValueError:
            return
        with self.lock:
            self.theta = math.radians(90 - h)     # NED -> ENU yaw [rad]

    def _rtcm_loop(self):
        while not rospy.is_shutdown():
            try:
                data = self.sock.recv(2048)
                if not data:
                    rospy.logwarn("[GPS] NTRIP closed; reconnecting...")
                    self.sock = self._connect_ntrip()
                    continue
                self.ser.write(data)
            except OSError as e:
                rospy.logwarn("[GPS] NTRIP error (%s); retry in 2s", e)
                time.sleep(2)
                try:
                    self.sock = self._connect_ntrip()
                except Exception:
                    pass

    def get_state(self):
        with self.lock:
            return self.x, self.y, self.theta, self.quality, self.stamp


# ============================================================================
# YOLOv7 (COCO) — same pipeline as your yolov7 ROS node — unchanged
# ============================================================================

class YoloV7:
    def __init__(self, weights, conf_thresh, iou_thresh, img_size, device):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.device = device
        self.model = attempt_load(weights, map_location=device)
        self.model.eval()
        rospy.loginfo("[YOLO] weights loaded on %s", device)

    @torch.no_grad()
    def detect(self, bgr):
        h0, w0 = bgr.shape[:2]
        resized = cv2.resize(bgr, (self.img_size, self.img_size))
        img = resized.transpose((2, 0, 1))[::-1]        # HWC->CHW, BGR->RGB
        img = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        pred = self.model(img)[0]
        det = non_max_suppression(pred, conf_thres=self.conf_thresh,
                                  iou_thres=self.iou_thresh)
        det = det[0] if det else torch.empty((0, 6))
        det = det.cpu().numpy()
        if len(det):                                    # rescale to original
            det[:, [0, 2]] *= w0 / self.img_size
            det[:, [1, 3]] *= h0 / self.img_size
        return det                                      # [N,6] x1y1x2y2 conf cls


# ============================================================================
# Geometry helpers (median depth identical in spirit to car_mapper_node)
# ============================================================================

def get_box_median_depth(depth_img, x1, y1, x2, y2, depth_scale=0.001):
    h, w = depth_img.shape[:2]
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(w, int(x2)), min(h, int(y2))
    crop = depth_img[y1c:y2c, x1c:x2c].astype(np.float32)
    valid = crop[crop > 0] * depth_scale
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def project_detection_to_world(det_row, depth_m,
                               robot_x, robot_y, robot_theta,
                               side, img_width,
                               hfov_deg=69.0):

    x1, _, x2, _ = det_row[:4]
    u = (x1 + x2) / 2.0
    bearing = math.radians(((u / img_width) - 0.5) * hfov_deg)

    side_offset = math.pi / 2 if side == "left" else -math.pi / 2
    angle = robot_theta + side_offset - (bearing if side == "left" else -bearing)

    cx = robot_x + depth_m * math.cos(angle)
    cy = robot_y + depth_m * math.sin(angle)
    return cx, cy


def containment_ratio(person_box, car_box):
    """area(person ∩ car) / area(person)."""
    px1, py1, px2, py2 = person_box
    cx1, cy1, cx2, cy2 = car_box
    iw = max(0.0, min(px2, cx2) - max(px1, cx1))
    ih = max(0.0, min(py2, cy2) - max(py1, cy1))
    p_area = max(1e-6, (px2 - px1) * (py2 - py1))
    return (iw * ih) / p_area

def match_detection_to_car(car_det, car_depth,
                           cars_snapshot: List[MappedCar],
                           robot_x, robot_y, robot_theta,
                           side, img_width, hfov_deg,
                           gps_range_m, depth_tolerance_m, car_length_m,
                           logger=None):
    # ── Reachability ──────────────────────────────────────────────────
    # IMPORTANT: reachability must be judged against the car's NEAREST
    # physical endpoint (the interval's lower bound), NOT the raw distance
    # to the stored reference position. For a "back"-tagged car, the
    # stored (x, y) is the FRONT-shifted reference point (car_length_m
    # farther from the robot than the car's actual visible face). Using
    # c.distance_to(robot_x, robot_y) directly here rejects a back car
    # that is genuinely within range — reproducing the exact bug this
    # interval mechanism exists to fix, just one step earlier.
    reachable = []
    intervals = {}
    for c in cars_snapshot:
        lo, hi = car_depth_interval(c, robot_x, robot_y, car_length_m)
        intervals[c.id] = (lo, hi)
        if lo <= gps_range_m:
            reachable.append(c)
    if not reachable:
        if logger:
            logger(
                "[Assoc] no reachable cars: nearest physical point of every "
                "mapped car exceeds gps_range_m=%.2fm" % gps_range_m)
        return None

    plausible = []
    for c in reachable:
        lo, hi = intervals[c.id]
        if (lo - depth_tolerance_m) <= car_depth <= (hi + depth_tolerance_m):
            err = 0.0 if lo <= car_depth <= hi else min(abs(car_depth-lo), abs(car_depth-hi))
            plausible.append((c, err))

    if not plausible:
        return None

    if len(plausible) == 1:
        return plausible[0][0], plausible[0][1]

    cx, cy = project_detection_to_world(
        car_det, car_depth, robot_x, robot_y, robot_theta,
        side, img_width, hfov_deg=hfov_deg)
    #target = min((c for c, _ in plausible), key=lambda c: c.distance_to(cx, cy))
    target = min(
    (c for c, _ in plausible),
    key=lambda c: math.hypot(cx - car_near_point(c, robot_x, robot_y, car_length_m)[0],
                              cy - car_near_point(c, robot_x, robot_y, car_length_m)[1])
    )
    match_quality = target.distance_to(cx, cy)
    return target, match_quality

def update_score(old_score, new_score, alpha_person, alpha_no_person):
    alpha = alpha_person if new_score >= 0 else alpha_no_person
    return alpha * new_score + (1 - alpha) * old_score

def car_depth_interval(c, robot_x, robot_y, car_length_m):
    d_ref = c.distance_to(robot_x, robot_y)
    if c.orientation_rad is None:
        half = car_length_m / 2.0
        return max(0.0, d_ref - half), d_ref + half
    ox = c.x - car_length_m * math.cos(c.orientation_rad)
    oy = c.y - car_length_m * math.sin(c.orientation_rad)
    d_other = math.hypot(ox - robot_x, oy - robot_y)
    return min(d_ref, d_other), max(d_ref, d_other)

def car_near_point(c, robot_x, robot_y, car_length_m):
    """World (x,y) of the car endpoint nearest the robot — the physically
    visible face — instead of the raw stored reference point."""
    if c.orientation_rad is None:
        return c.x, c.y
    ox = c.x - car_length_m * math.cos(c.orientation_rad)
    oy = c.y - car_length_m * math.sin(c.orientation_rad)
    d_ref = math.hypot(c.x - robot_x, c.y - robot_y)
    d_other = math.hypot(ox - robot_x, oy - robot_y)
    return (c.x, c.y) if d_ref <= d_other else (ox, oy)

# ============================================================================
# Data association node — car-oriented
# ============================================================================

class DataAssociationNode:
    def __init__(self):
        rospy.init_node("data_association_node")

        # --- params
        weights = rospy.get_param("~weights", '/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/yolov7-ros/weights/yolov7.pt')
        self.cont_thresh = rospy.get_param("~containment_thresh", 0.7)
        conf_thresh = rospy.get_param("~conf_thresh", 0.4)
        iou_thresh = rospy.get_param("~iou_thresh", 0.45)
        img_size = rospy.get_param("~img_size", 640)
        device = rospy.get_param("~device", "cuda")
        self.gps_range_m = rospy.get_param("~gps_range_m", 4.5)
        self.max_depth_m = rospy.get_param("~max_depth_m", 4.5)
        self.side = rospy.get_param("~side", "right")
        self.hfov_deg = rospy.get_param("~hfov_deg", 69.0)
        #self.max_bearing_deg = rospy.get_param("~max_bearing_deg", 15.0)
        #self.assoc_max_m = rospy.get_param("~assoc_max_m", 1.0)
        #self.front_offset_m = rospy.get_param("~front_offset_m", 3.7)  
        self.car_length_m = rospy.get_param("~car_length_m", 3.7)  
        self.depth_tolerance_m = rospy.get_param("~depth_tolerance_m", 1.5)
        if self.side not in ("left", "right"):
            raise ValueError(f"~side must be 'left' or 'right', got {self.side!r}")
        self.min_quality = rospy.get_param("~min_rtk_quality", 4)
        #self.ema_alpha = rospy.get_param("~ema_alpha", 1.0)
        rgb_topic = rospy.get_param("~rgb_topic", "/cam_up/color/image_raw")
        depth_topic = rospy.get_param(
            "~depth_topic", "/cam_up/aligned_depth_to_color/image_raw")
        origin_lat = rospy.get_param("~origin_lat", 41.85626425142204)
        origin_lon = rospy.get_param("~origin_lon", 12.469038428190489)
        self._ox, self._oy, self._zone_n, self._zone_l = utm.from_latlon(
            origin_lat, origin_lon)
        self.ema_alpha_person = rospy.get_param("~ema_alpha_person", 0.6)
        self.ema_alpha_no_person = rospy.get_param("~ema_alpha_no_person", 0.05)
        self.close_range_m = rospy.get_param("~close_range_m", 1.75)
        self.person_depth_tolerance_m = rospy.get_param("~person_depth_tolerance_m", 2.0)

        self.cars_lock = threading.Lock()
        self.cars: List[MappedCar] = []
        self.scores = np.zeros((2, 0), dtype=np.float32)

        self.car_map_source = rospy.get_param("~car_map_source", "topic")
        if self.car_map_source == "file":
            self._load_car_map_file(rospy.get_param("~car_map_file"))
        elif self.car_map_source == "hardcoded":
            self._apply_car_map(HARDCODED_CARS)
            rospy.loginfo("[Assoc] car map HARDCODED (%d cars, no file/topic "
                          "used) — edit HARDCODED_CARS in this script to "
                          "change positions/tags.", len(self.cars))
        else:
            rospy.Subscriber("/car_map/state", String,
                             self._car_map_cb, queue_size=1)

        # --- GPS
        self.gps_source = rospy.get_param("~gps_source", "topic")
        if self.gps_source == "serial":
            self.gps = GpsRtkBackend(rospy.get_param("~origin_lat"),
                                     rospy.get_param("~origin_lon"))
        else:
            self.gps = None
            self._gps_lock = threading.Lock()
            self._gx = self._gy = self._gtheta = None
            self._gstamp = rospy.Time(0)
            rospy.Subscriber(rospy.get_param("~gps_topic", "/gps_data"),
                             Pose2D, self._gps_cb, queue_size=10)

        # --- detector
        self.yolo = YoloV7(weights, conf_thresh, iou_thresh, img_size, device)

        # --- pubs
        self.pub_scores = rospy.Publisher("/parking/scores",
                                          Float32MultiArray, queue_size=5)
        self.pub_scores_json = rospy.Publisher("/parking/scores_json",
                                               String, queue_size=5)
        self.pub_debug = rospy.Publisher("/data_association/debug_image",
                                         Image, queue_size=2)
        self.bridge = CvBridge()

        # --- synchronized RGB + depth (same style as car_mapper_node)
        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.05)
        sync.registerCallback(self.step)

        rospy.loginfo("[Assoc] node ready (car-oriented, source=%s).",
                      self.car_map_source)

    # --------------------------------------------------------------- inputs
    def _gps_cb(self, msg: Pose2D):
        with self._gps_lock:
            self._gx, self._gy, self._gtheta = msg.x, msg.y, msg.theta
            self._gstamp = rospy.Time.now()

    def _get_gps(self):
        """Returns (x, y, theta, ok)."""
        if self.gps is not None:
            x, y, theta, quality, stamp = self.gps.get_state()
            ok = (x is not None and quality >= self.min_quality
                  and (rospy.Time.now() - stamp).to_sec() < 1.0)
            return x, y, theta, ok
        with self._gps_lock:
            ok = (self._gx is not None
                  and (rospy.Time.now() - self._gstamp).to_sec() < 1.0)
            return self._gx, self._gy, self._gtheta, ok

    # --- car map ingestion (replaces occupancy priors) ---------------------
    def _car_map_cb(self, msg: String):
        try:
            state = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._apply_car_map(state.get("cars", []))

    def _load_car_map_file(self, path):
        with open(path) as f:
            state = json.load(f)
        self._apply_car_map(state.get("cars", []))
        rospy.loginfo("[Assoc] car map loaded from %s (%d cars)",
                      path, len(self.cars))

    def _apply_car_map(self, car_dicts):
        """
        Rebuild the mapped-car list; preserve existing scores by ORIGINAL id
        (not by array position, since positions can shift as the loaded set
        changes). The scores array is sized by the ACTUAL number of loaded
        cars (len(new_cars)), not by max(id)+1 — car_mapper_node ids can be
        sparse (e.g. {0, 3, 4} if ids 1, 2 were discarded as low-confidence
        tracks), and sizing by max id would silently allocate permanent
        zero-filled phantom columns for the missing ids.
        """
        new_cars = cars_from_state(car_dicts)
        if not new_cars:
            return
        for c in new_cars:
            c.lat, c.lon = local_xy_to_latlon(
                c.x, c.y, self._ox, self._oy, self._zone_n, self._zone_l)

        with self.cars_lock:
            # Snapshot old scores keyed by ORIGINAL id before rebuilding.
            old_scores_by_id = {}
            for old_c in self.cars:
                if old_c.score_col < self.scores.shape[1]:
                    old_scores_by_id[old_c.id] = (
                        self.scores[0, old_c.score_col],
                        self.scores[1, old_c.score_col],
                    )

            n = len(new_cars)
            new_scores = np.zeros((2, n), dtype=np.float32)
            for idx, c in enumerate(new_cars):
                c.score_col = idx
                if c.id in old_scores_by_id:
                    new_scores[0, idx], new_scores[1, idx] = \
                        old_scores_by_id[c.id]

            self.cars = new_cars
            self.scores = new_scores

    # ----------------------------------------------------------------- step
    def step(self, rgb_msg: Image, depth_msg: Image):
        """
        One iteration k. Wrapped in try/except: any exception in the
        processing below is logged with a full traceback via rospy.logerr
        rather than potentially failing silently — this was added
        specifically because a prior symptom (front cars stopped matching
        right after a back car failed to match) could not be conclusively
        explained by the reachability bug alone, and a swallowed exception
        was the leading alternative explanation. If this fires, the
        traceback will show up in the node's log output.
        """
        try:
            self._step_impl(rgb_msg, depth_msg)
        except Exception:
            rospy.logerr("[Assoc] EXCEPTION in step():\n%s",
                        traceback.format_exc())

    def _step_impl(self, rgb_msg: Image, depth_msg: Image):
        with self.cars_lock:
            cars_snapshot = list(self.cars)
        if not cars_snapshot:
            rospy.logwarn_throttle(
                10, "[Assoc] no car map yet (waiting for /car_map/state "
                    "or ~car_map_file)")
            return

        rx, ry, rtheta, gps_ok = self._get_gps()
        if not gps_ok or rtheta is None:
            rospy.logwarn_throttle(
                5, "[Assoc] no valid GPS pose (need RTK quality >= %d "
                   "and heading)", self.min_quality)
            return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        img_w = rgb.shape[1]

        det = self.yolo.detect(rgb)
        cars_det = [d for d in det if int(d[5]) in COCO_CAR_CLASSES]
        persons = [d for d in det if int(d[5]) == COCO_PERSON]

        # depth gate on cars (same policy as car_mapper_node)
        gated = []
        for c in cars_det:
            z = get_box_median_depth(depth, *c[:4])
            if z is not None and z <= self.max_depth_m:
                gated.append((c, z))
        """
        if not gated:
            self._publish_scores()
            return
        """

        matched = []
        for car, car_depth in gated:
            result = match_detection_to_car(
                car, car_depth, cars_snapshot,
                rx, ry, rtheta, self.side, img_w, self.hfov_deg,
                gps_range_m=self.gps_range_m,
                depth_tolerance_m=self.depth_tolerance_m,car_length_m=self.car_length_m,
                logger=rospy.logdebug)
            if result is None:
                continue
            target, match_quality = result
            matched.append((car, car_depth, target, match_quality))

        """
        if not matched:
            self._publish_scores()
            return
        """
        # if multiple detections match the SAME mapped car, keep the one
        # with the smallest association distance
        best_by_id = {}
        for m in matched:
            tid = m[2].id
            if tid not in best_by_id or m[3] < best_by_id[tid][3]:
                best_by_id[tid] = m

        last_target = None
        any_occupied = False
        best_car_box = None
        for car, car_depth, target, match_quality in best_by_id.values():
            # person-in-car containment (best person for THIS car box)
            best_person_conf = None
            for p in persons:
                if containment_ratio(p[:4], car[:4]) >= self.cont_thresh:
                    pc = float(p[4])
                    if best_person_conf is None or pc > best_person_conf:
                        best_person_conf = pc

            # score update — indexed by mapped-car id as a COLUMN, ROW 0 ONLY.
            car_conf = float(car[4])
            is_unvisitable_back = (target.tag == "back" and not target.visitable)

            if best_person_conf is not None:
                new = +(car_conf + best_person_conf) / 2.0   # (0, +1]
            else:
                if is_unvisitable_back:
                    new = 0.0
                else:
                    new = -car_conf                          # [-1, 0)
                                          
            with self.cars_lock:
                if is_unvisitable_back and best_person_conf is None:
                    # Forza lo score a 0 saltando l'EMA
                    self.scores[0, target.score_col] = 0.0
                else:
                    old = self.scores[0, target.score_col]
                    self.scores[0, target.score_col] = update_score(
                        old, new, self.ema_alpha_person, self.ema_alpha_no_person)
                
                score_now = self.scores[0, target.score_col]
            
            rospy.loginfo(
                "[Assoc] car %d (%s, visitable=%r): score=%+.2f (car=%.2f person=%s "
                "depth=%.2fm car_pos=(%.2f,%.2f) match_err=%.2fm)",
                target.id, target.tag, target.visitable, score_now, car_conf,
                f"{best_person_conf:.2f}" if best_person_conf else "none",
                car_depth, target.x, target.y, match_quality)

            last_target = target
            if best_person_conf is not None:
                any_occupied = True
                best_car_box = car
            elif best_car_box is None:
                best_car_box = car

        # ── Person-only fallback ─────────────────────────────────────────
        matched_ids = set(best_by_id.keys())
        near_cars = [c for c in cars_snapshot
                    if c.distance_to(rx, ry) <= self.close_range_m]
        for near_car in near_cars:
            if near_car.id in matched_ids:
                continue

            near_x, near_y = car_near_point(near_car, rx, ry, self.car_length_m)
            expected_dist = math.hypot(near_x - rx, near_y - ry)
            best_person_conf = None
            for p in persons:
                pz = get_box_median_depth(depth, *p[:4])
                if pz is None:
                    continue
                if abs(pz - expected_dist) > self.person_depth_tolerance_m:
                    continue
                pc = float(p[4])
                if best_person_conf is None or pc > best_person_conf:
                    best_person_conf = pc

            if best_person_conf is None:
                continue

            new = +best_person_conf
            with self.cars_lock:
                old = self.scores[0, near_car.score_col]
                self.scores[0, near_car.score_col] = update_score(
                    old, new, self.ema_alpha_person, self.ema_alpha_no_person)
                score_now = self.scores[0, near_car.score_col]

            rospy.loginfo(
                "[Assoc] car %d (%s): FALLBACK score=%+.2f (no car box "
                "detected; person_conf=%.2f, expected_dist=%.2fm)",
                near_car.id, near_car.tag, score_now, best_person_conf,
                expected_dist)

            last_target = near_car
            any_occupied = True

        if last_target is None:
            self._publish_scores()
            return

        self._publish_scores(last_target)
        if best_car_box is not None:
            self._publish_debug(rgb, best_car_box, persons, any_occupied)

    # ------------------------------------------------------------ publishing
    def _publish_scores(self, updated_car: Optional[MappedCar] = None):
        with self.cars_lock:
            scores = self.scores.copy()
            cars_snapshot = list(self.cars)

        n_rows, n_cols = scores.shape   # (2, N)

        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="rows", size=n_rows,
                                stride=n_rows * n_cols),
            MultiArrayDimension(label="cols", size=n_cols,
                                stride=n_cols),
        ]
        msg.layout.data_offset = 0
        msg.data = scores.flatten().tolist()
        self.pub_scores.publish(msg)

        payload = {
            "updated": updated_car.id if updated_car else None,
            "cars": [{"id": c.id,
                      "score": float(scores[0, c.score_col]),
                      "score_row1": float(scores[1, c.score_col]),
                      "tag": c.tag,
                      "x": round(c.x, 3),
                      "y": round(c.y, 3),
                      "lat": round(c.lat, 8) if c.lat is not None else None,
                      "lon": round(c.lon, 8) if c.lon is not None else None,
                      "orientation_rad": c.orientation_rad,
                      "visitable": c.visitable}
                     for c in cars_snapshot],
        }
        self.pub_scores_json.publish(String(json.dumps(payload)))

    def _publish_debug(self, rgb, car, persons, occupied):
        if self.pub_debug.get_num_connections() == 0:
            return
        img = rgb.copy()
        x1, y1, x2, y2 = map(int, car[:4])
        col = (0, 0, 255) if occupied else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"car {car[4]:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        for p in persons:
            x1, y1, x2, y2 = map(int, p[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))


if __name__ == "__main__":
    try:
        DataAssociationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass