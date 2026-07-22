#!/usr/bin/env python3
import os
import time
import json
import math
import csv
import threading
import traceback
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import casadi as ca

import rospy
import message_filters
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32, String, MultiArrayDimension

import utm
import sys

# Path al repository YOLOv7 (Adatta se necessario)
_YOLOV7_REPO_ROOT = ("/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/yolov7-ros/src")
if _YOLOV7_REPO_ROOT not in sys.path:
    sys.path.insert(0, _YOLOV7_REPO_ROOT)

from models.experimental import attempt_load
from utils.general import non_max_suppression

COCO_PERSON = 0
COCO_CAR_CLASSES = {2, 7}

HARDCODED_CARS: List[dict] = [
    {"id": 0, "x": 42.36, "y": -6.16, "class": "car_back", "orientation_rad": 1.57},
    {"id": 1, "x": 49.53, "y": -7.62, "class": "car_front", "orientation_rad": -1.57},
    {"id": 2, "x": 52.48, "y":  -8.08, "class": "car_front",  "orientation_rad":  -1.57},
]

class MappedCar:
    def __init__(self, car_id: int, x: float, y: float, tag: str = "unknown",
                 orientation_rad: Optional[float] = None, lat: Optional[float] = None,
                 lon: Optional[float] = None, visitable: bool = True):
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
    cars = []
    for d in car_dicts:
        cls = d.get("class") or ""
        tag = cls.replace("car_", "") if cls else "unknown"
        visitable = d.get("visitable", True)
        cars.append(MappedCar(
            car_id=int(d["id"]), x=float(d["x"]), y=float(d["y"]),
            tag=tag, orientation_rad=d.get("orientation_rad"), visitable=visitable
        ))
    cars.sort(key=lambda c: c.id)
    return cars

def local_xy_to_latlon(x: float, y: float, ox: float, oy: float, zone_n: int, zone_l: str) -> Tuple[float, float]:
    easting = x + ox
    northing = y + oy
    lat, lon = utm.to_latlon(easting, northing, zone_n, zone_l)
    return lat, lon

class YoloV7:
    def __init__(self, weights, conf_thresh, iou_thresh, img_size, device):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.device = device
        self.model = attempt_load(weights, map_location=device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, bgr):
        h0, w0 = bgr.shape[:2]
        resized = cv2.resize(bgr, (self.img_size, self.img_size))
        img = resized.transpose((2, 0, 1))[::-1]
        img = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        pred = self.model(img)[0]
        det = non_max_suppression(pred, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh)
        det = det[0] if det else torch.empty((0, 6))
        det = det.cpu().numpy()
        if len(det):
            det[:, [0, 2]] *= w0 / self.img_size
            det[:, [1, 3]] *= h0 / self.img_size
        return det

def get_box_median_depth(depth_img, x1, y1, x2, y2, depth_scale=0.001):
    h, w = depth_img.shape[:2]
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(w, int(x2)), min(h, int(y2))
    crop = depth_img[y1c:y2c, x1c:x2c].astype(np.float32)
    valid = crop[crop > 0] * depth_scale
    if len(valid) == 0: return None
    return float(np.median(valid))

def project_detection_to_world(det_row, depth_m, robot_x, robot_y, robot_theta, side, img_width, hfov_deg=69.0):
    x1, _, x2, _ = det_row[:4]
    u = (x1 + x2) / 2.0
    bearing = math.radians(((u / img_width) - 0.5) * hfov_deg)
    side_offset = math.pi / 2 if side == "left" else -math.pi / 2
    angle = robot_theta + side_offset - (bearing if side == "left" else -bearing)
    cx = robot_x + depth_m * math.cos(angle)
    cy = robot_y + depth_m * math.sin(angle)
    return cx, cy

def containment_ratio(person_box, car_box):
    px1, py1, px2, py2 = person_box
    cx1, cy1, cx2, cy2 = car_box
    iw = max(0.0, min(px2, cx2) - max(px1, cx1))
    ih = max(0.0, min(py2, cy2) - max(py1, cy1))
    p_area = max(1e-6, (px2 - px1) * (py2 - py1))
    return (iw * ih) / p_area

def car_near_point(c, robot_x, robot_y, car_length_m):
    if c.orientation_rad is None: return c.x, c.y
    ox = c.x - car_length_m * math.cos(c.orientation_rad)
    oy = c.y - car_length_m * math.sin(c.orientation_rad)
    d_ref = math.hypot(c.x - robot_x, c.y - robot_y)
    d_other = math.hypot(ox - robot_x, oy - robot_y)
    return (c.x, c.y) if d_ref <= d_other else (ox, oy)

def car_depth_interval(c, robot_x, robot_y, car_length_m):
    d_ref = c.distance_to(robot_x, robot_y)
    if c.orientation_rad is None:
        half = car_length_m / 2.0
        return max(0.0, d_ref - half), d_ref + half
    ox = c.x - car_length_m * math.cos(c.orientation_rad)
    oy = c.y - car_length_m * math.sin(c.orientation_rad)
    d_other = math.hypot(ox - robot_x, oy - robot_y)
    return min(d_ref, d_other), max(d_ref, d_other)

def match_detection_to_car(car_det, car_depth, cars_snapshot, robot_x, robot_y, robot_theta, side, img_width, hfov_deg, gps_range_m, depth_tolerance_m, car_length_m):
    reachable = []
    intervals = {}
    for c in cars_snapshot:
        lo, hi = car_depth_interval(c, robot_x, robot_y, car_length_m)
        intervals[c.id] = (lo, hi)
        if lo <= gps_range_m: reachable.append(c)
    if not reachable: return None

    plausible = []
    for c in reachable:
        lo, hi = intervals[c.id]
        if (lo - depth_tolerance_m) <= car_depth <= (hi + depth_tolerance_m):
            err = 0.0 if lo <= car_depth <= hi else min(abs(car_depth-lo), abs(car_depth-hi))
            plausible.append((c, err))

    if not plausible: return None
    if len(plausible) == 1: return plausible[0][0], plausible[0][1]

    cx, cy = project_detection_to_world(car_det, car_depth, robot_x, robot_y, robot_theta, side, img_width, hfov_deg=hfov_deg)
    target = min((c for c, _ in plausible), key=lambda c: math.hypot(cx - car_near_point(c, robot_x, robot_y, car_length_m)[0], cy - car_near_point(c, robot_x, robot_y, car_length_m)[1]))
    return target, target.distance_to(cx, cy)

def update_score(old_score, new_score, alpha_person, alpha_no_person):
    alpha = alpha_person if new_score >= 0 else alpha_no_person
    return alpha * new_score + (1 - alpha) * old_score

class DataAssocAndLoggingNode:
    def __init__(self):
        rospy.init_node("data_assoc_logging_node")

        # Configurazione Parametri
        weights = rospy.get_param("~weights", '/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/yolov7-ros/weights/yolov7.pt')
        self.cont_thresh = rospy.get_param("~containment_thresh", 0.7)
        self.gps_range_m = rospy.get_param("~gps_range_m", 4.5)
        self.max_depth_m = rospy.get_param("~max_depth_m", 4.5)
        self.side = rospy.get_param("~side", "right")
        self.hfov_deg = rospy.get_param("~hfov_deg", 69.0)
        self.car_length_m = rospy.get_param("~car_length_m", 3.7)  
        self.depth_tolerance_m = rospy.get_param("~depth_tolerance_m", 1.5)
        self.ema_alpha_person = rospy.get_param("~ema_alpha_person", 0.6)
        self.ema_alpha_no_person = rospy.get_param("~ema_alpha_no_person", 0.1)
        self.close_range_m = rospy.get_param("~close_range_m", 1.75)
        self.person_depth_tolerance_m = rospy.get_param("~person_depth_tolerance_m", 2.0)
        self.baselines_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../baselines")
        
        # Inizializzazione Logica Bayesiana (dal NMPC)
        self.beliefs_k = None
        self.entropy_entire_field = None
        self.num_total_trees = 0
        
        # Variabili di Logging
        self.sim_start_time = time.time()
        self.lambda_history = []
        self.entropy_history = []
        self.time_history = []
        self.pose_history = []

        self.cars_lock = threading.Lock()
        self.cars: List[MappedCar] = []
        self.scores = np.zeros((2, 0), dtype=np.float32)

        # -------------------------------------------------------------
        # Inizializzazione Mappa (Sorgente: File)
        # -------------------------------------------------------------
        self.car_map_source = rospy.get_param("~car_map_source", "file")
        if self.car_map_source == "file":
            car_map_file_default = "/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/nmpc_ros/niccolo/map_results/car_map_final.json"
            file_path = rospy.get_param("~car_map_file", car_map_file_default)
            self._load_car_map_file(file_path)
        elif self.car_map_source == "hardcoded":
            self._apply_car_map(HARDCODED_CARS)
        else:
            rospy.Subscriber("/car_map/state", String, self._car_map_cb, queue_size=1)

        # GPS Setup (Solo topic per esecuzione da ROS Bag)
        self._gps_lock = threading.Lock()
        self._gx = self._gy = self._gtheta = None
        self._gstamp = rospy.Time(0)
        rospy.Subscriber(rospy.get_param("~gps_topic", "/gps_data"), Pose2D, self._gps_cb, queue_size=10)

        # YOLO Setup
        self.yolo = YoloV7(weights, rospy.get_param("~conf_thresh", 0.4), rospy.get_param("~iou_thresh", 0.45), rospy.get_param("~img_size", 640), rospy.get_param("~device", "cuda"))
        self.bridge = CvBridge()

        # -------------------------------------------------------------
        # Modifica ai topic della camera basata sul contenuto del rosbag
        # -------------------------------------------------------------
        rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        
        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        sync.registerCallback(self.step)
        
        rospy.on_shutdown(self.save_performance_data)
        rospy.loginfo("[Assoc+Log] Nodo inizializzato. In attesa di dati...")

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

    def _load_car_map_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._apply_car_map(state.get("cars", []))
            rospy.loginfo(f"[Assoc] Car map caricata correttamente dal file: {path}")
        except Exception as e:
            rospy.logerr(f"[Assoc] Errore caricamento car map da {path}: {e}")

    def _apply_car_map(self, car_dicts):
        new_cars = cars_from_state(car_dicts)
        if not new_cars: return

        with self.cars_lock:
            n = len(new_cars)
            self.scores = np.zeros((2, n), dtype=np.float32)
            self.cars = new_cars
            
            # Setup del filtro per le macchine presenti
            self.num_total_trees = n
            self.beliefs_k = ca.DM.ones(self.num_total_trees, 2) * 0.5
            self.entropy_entire_field = self.entropy_f(self.num_total_trees)
            
            for idx, c in enumerate(self.cars):
                c.score_col = idx

    def _car_map_cb(self, msg: String):
        try:
            state = json.loads(msg.data)
            self._apply_car_map(state.get("cars", []))
        except json.JSONDecodeError:
            pass

    def _gps_cb(self, msg: Pose2D):
        with self._gps_lock:
            self._gx, self._gy, self._gtheta = msg.x, msg.y, msg.theta
            self._gstamp = rospy.Time.now()

    def step(self, rgb_msg: Image, depth_msg: Image):
        try:
            self._step_impl(rgb_msg, depth_msg)
        except Exception:
            rospy.logerr("[Assoc+Log] EXCEPTION:\n%s", traceback.format_exc())

    def _step_impl(self, rgb_msg: Image, depth_msg: Image):
        with self.cars_lock:
            cars_snapshot = list(self.cars)
        if not cars_snapshot or self.beliefs_k is None: return

        with self._gps_lock:
            rx, ry, rtheta = self._gx, self._gy, self._gtheta
            valid_gps = rx is not None
        if not valid_gps: return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        img_w = rgb.shape[1]

        det = self.yolo.detect(rgb)
        cars_det = [d for d in det if int(d[5]) in COCO_CAR_CLASSES]
        persons = [d for d in det if int(d[5]) == COCO_PERSON]

        gated = []
        for c in cars_det:
            z = get_box_median_depth(depth, *c[:4])
            if z is not None and z <= self.max_depth_m:
                gated.append((c, z))

        matched = []
        for car, car_depth in gated:
            result = match_detection_to_car(car, car_depth, cars_snapshot, rx, ry, rtheta, self.side, img_w, self.hfov_deg, self.gps_range_m, self.depth_tolerance_m, self.car_length_m)
            if result:
                matched.append((car, car_depth, result[0], result[1]))

        best_by_id = {}
        for m in matched:
            tid = m[2].id
            if tid not in best_by_id or m[3] < best_by_id[tid][3]:
                best_by_id[tid] = m

        for car, car_depth, target, match_quality in best_by_id.values():
            best_person_conf = None
            for p in persons:
                if containment_ratio(p[:4], car[:4]) >= self.cont_thresh:
                    pc = float(p[4])
                    if best_person_conf is None or pc > best_person_conf:
                        best_person_conf = pc

            car_conf = float(car[4])
            is_unvisitable_back = (target.tag == "back" and not target.visitable)

            if best_person_conf is not None:
                new = 1.0  
            else:
                if is_unvisitable_back:
                    new = 0.0
                else:
                    new = -car_conf
                                          
            with self.cars_lock:
                if is_unvisitable_back and best_person_conf is None:
                    self.scores[0, target.score_col] = 0.0
                else:
                    old = self.scores[0, target.score_col]
                    self.scores[0, target.score_col] = update_score(old, new, self.ema_alpha_person, self.ema_alpha_no_person)

        matched_ids = set(best_by_id.keys())
        near_cars = [c for c in cars_snapshot if c.distance_to(rx, ry) <= self.close_range_m]
        
        for near_car in near_cars:
            if near_car.id in matched_ids: continue
            near_x, near_y = car_near_point(near_car, rx, ry, self.car_length_m)
            expected_dist = math.hypot(near_x - rx, near_y - ry)
            
            best_person_conf = None
            for p in persons:
                pz = get_box_median_depth(depth, *p[:4])
                if pz is None or abs(pz - expected_dist) > self.person_depth_tolerance_m: continue
                pc = float(p[4])
                if best_person_conf is None or pc > best_person_conf: best_person_conf = pc

            if best_person_conf is None: continue
            
            new = 1.0
            
            with self.cars_lock:
                old = self.scores[0, near_car.score_col]
                self.scores[0, near_car.score_col] = update_score(old, new, self.ema_alpha_person, self.ema_alpha_no_person)

        # -------------------------------------------------------------
        # CALCOLO ENTROPIA E SALVATAGGIO STORICO (LOGICA NMPC)
        # -------------------------------------------------------------
        with self.cars_lock:
            current_scores = self.scores.copy()
            
        prob_scores = np.zeros_like(current_scores.T) 
        prob_scores[:, 0] = 0.5 + 0.5 * current_scores[0, :]
        prob_scores[:, 1] = 1.0 - prob_scores[:, 0]

        self.beliefs_k = self.bayes(self.beliefs_k, ca.DM(prob_scores))
        
        entropy_k = self.entropy_entire_field(self.beliefs_k)
        current_sim_time = time.time() - self.sim_start_time

        self.pose_history.append([rx, ry, rtheta])
        self.time_history.append(current_sim_time)
        self.lambda_history.append(self.beliefs_k.full().flatten().tolist())
        self.entropy_history.append(ca.sum1(entropy_k).full().flatten()[0])

        rospy.loginfo_throttle(1, f"Aggiornamento completato. Entropia globale: {self.entropy_history[-1]:.4f}")

    def save_performance_data(self):
        rospy.loginfo("Arresto del nodo. Inizio salvataggio dati...")
        os.makedirs(self.baselines_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        plot_csv = os.path.join(self.baselines_dir, f"husky_mpc_{timestamp}_plot_data.csv")
        with open(plot_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            
            # --- RIGA 1: Posizioni delle auto ---
            tree_positions_flat = []
            with self.cars_lock:
                for c in self.cars:
                    # Usa 0.0 se orientation_rad è None
                    ori = c.orientation_rad if c.orientation_rad is not None else 0.0
                    tree_positions_flat.extend([c.x, c.y, ori])
            writer.writerow(["tree_positions"] + tree_positions_flat)
            
            # --- RIGA 2: Ground Truth IDs (Dato fittizio per compatibilità script di plotting) ---
            writer.writerow(["trees_gt_id", 0, 1])
            
            # --- RIGA 3: Intestazione Temporale ---
            header = ["time", "x", "y", "theta", "entropy"]
            if self.lambda_history:
                header += [f"lambda_{i}" for i in range(len(self.lambda_history[0]))]
            writer.writerow(header)
            
            # --- RIGHE DATI ---
            for i in range(len(self.time_history)):
                x, y, theta = self.pose_history[i]
                row = [self.time_history[i], x, y, theta, self.entropy_history[i]] + self.lambda_history[i]
                writer.writerow(row)
                
        rospy.loginfo(f"Dati di entropia e lambda salvati con successo in: {plot_csv}")

if __name__ == "__main__":
    try:
        node = DataAssocAndLoggingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass