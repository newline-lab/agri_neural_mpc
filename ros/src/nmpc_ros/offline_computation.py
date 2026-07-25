#!/usr/bin/env python3
"""
Script di Data Association e aggiornamento Bayesiano Offline da Rosbag.
"""

import os
import json
import math
import csv
import time
import numpy as np
import cv2
import torch
import rosbag
from cv_bridge import CvBridge
import casadi as ca
import utm

import sys
# Modifica questo percorso se necessario
_YOLOV7_REPO_ROOT = ("/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/yolov7-ros/src")
if _YOLOV7_REPO_ROOT not in sys.path:
    sys.path.insert(0, _YOLOV7_REPO_ROOT)

from models.experimental import attempt_load
from utils.general import non_max_suppression

COCO_PERSON = 0
COCO_CAR_CLASSES = {2,7}

# ============================================================================
# CLASSI E FUNZIONI DI SUPPORTO (Tratte dal tuo Data Association)
# ============================================================================

class MappedCar:
    def __init__(self, car_id: int, x: float, y: float, orientation_rad=None, visitable=True):
        self.id = car_id
        self.x, self.y = x, y
        self.orientation_rad = orientation_rad
        self.visitable = visitable
        self.score_col: int = 0

    def distance_to(self, rx, ry):
        return math.hypot(self.x - rx, self.y - ry)

def cars_from_state(car_dicts) -> list:
    cars = []
    for d in car_dicts:
        cars.append(MappedCar(
            car_id=int(d["id"]),
            x=float(d["x"]),
            y=float(d["y"]),
            orientation_rad=d.get("orientation_rad"),
            visitable=d.get("visitable", True)
        ))
    cars.sort(key=lambda c: c.id)
    return cars

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
    if len(valid) == 0:
        return None
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
    if c.orientation_rad is None:
        return c.x, c.y
    ox = c.x - car_length_m * math.cos(c.orientation_rad)
    oy = c.y - car_length_m * math.sin(c.orientation_rad)
    d_ref = math.hypot(c.x - robot_x, c.y - robot_y)
    d_other = math.hypot(ox - robot_x, oy - robot_y)
    return (c.x, c.y) if d_ref <= d_other else (ox, oy)

def match_detection_to_car(car_det, car_depth, cars_snapshot, robot_x, robot_y, robot_theta, side, img_width, hfov_deg, gps_range_m, depth_tolerance_m, car_length_m):
    reachable = []
    intervals = {}
    for c in cars_snapshot:
        lo, hi = car_depth_interval(c, robot_x, robot_y, car_length_m)
        intervals[c.id] = (lo, hi)
        if lo <= gps_range_m:
            reachable.append(c)
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

def update_score(old_score, new_score, alpha_person, alpha_no_person, uncertain=False):
    if uncertain:
        return 0
    else:
        alpha = alpha_person if new_score >= 0 else alpha_no_person
    return alpha * new_score + (1 - alpha) * old_score

def point_in_driver_half(c, px, py, car_length_m):
    if c.orientation_rad is None: return False
    ux, uy = math.cos(c.orientation_rad), math.sin(c.orientation_rad)
    back_x = c.x - car_length_m * ux
    back_y = c.y - car_length_m * uy
    along = (px - back_x) * ux + (py - back_y) * uy
    return along >= (car_length_m / 2.0)

def is_looking_at_driver_half(det_row, depth_m, c, robot_x, robot_y, robot_theta, side, img_width, hfov_deg, car_length_m):
    px, py = project_detection_to_world(det_row, depth_m, robot_x, robot_y, robot_theta, side, img_width, hfov_deg=hfov_deg)
    return point_in_driver_half(c, px, py, car_length_m)


# ============================================================================
# CLASSE PRINCIPALE PER L'ELABORAZIONE DEL ROSBAG
# ============================================================================

class BagProcessor:
    def __init__(self, bag_path, map_file_path, output_dir, yolo_weights):
        self.bag_path = bag_path
        self.output_dir = output_dir
        
        # Parametri YOLO e DA
        self.cont_thresh = 0.7
        self.hfov_deg = 69.0
        self.gps_range_m = 4.5
        self.max_depth_m = 4.0
        self.side = "right"
        self.car_length_m = 3.7
        self.depth_tolerance_m = 1.5
        self.ema_alpha_person = 0.6
        self.ema_alpha_no_person = 0.1
        self.close_range_m = 1.75
        self.person_depth_tolerance_m = 2.0
        
        # Inizializzazione YOLO
        print(f"Caricamento YOLOv7 da {yolo_weights}...")
        self.yolo = YoloV7(yolo_weights, conf_thresh=0.4, iou_thresh=0.45, img_size=640, device="cuda")
        self.bridge = CvBridge()

        # Inizializzazione Mappa e Score
        with open(map_file_path, "r", encoding="utf-8") as f:
            cars_data = json.load(f).get("cars", [])
        self.cars = cars_from_state(cars_data)
        for idx, c in enumerate(self.cars):
            c.score_col = idx
        
        self.num_total_cars = len(self.cars)
        self.da_scores = np.zeros((2, self.num_total_cars), dtype=np.float32)
        
        # Inizializzazione Bayes e CasADi
        self.beliefs_k = ca.DM.ones(self.num_total_cars, 2) * 0.5
        self.entropy_entire_field = self.entropy_f(self.num_total_cars)

        # Variabili di stato per la sincronizzazione del bag
        self.curr_gps = None
        self.curr_depth = None
        
        # Strutture dati per l'export CSV
        self.time_history = []
        self.pose_history = []
        self.entropy_history = []
        self.lambda_history = []

    @staticmethod
    def bayes(prior, likelihood):
        unnorm = prior * likelihood
        norm = ca.repmat(ca.sum2(unnorm), 1, 2)
        return ca.fmax(unnorm / norm, 0.01)

    @staticmethod
    def entropy_f(num_targets):
        p = ca.MX.sym(f'input_entropy_f{num_targets}_dim', num_targets, 2)
        eps = 1e-6
        p_clipped = ca.fmax(eps, ca.fmin(1 - eps, p))
        entropy_per_target = -ca.sum2(p_clipped * (ca.log(p_clipped)/ca.log(2)))
        return ca.Function(f'entropy_f_{num_targets}_dim', [p], [entropy_per_target])

    def run(self):
        print(f"Apertura del bag: {self.bag_path}")
        bag = rosbag.Bag(self.bag_path, 'r')
        
        # Topici target
        rgb_topic = "/camera/color/image_raw/compressed" # Sostituisci se diverso
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        gps_topic = "/gps_data"
        
        count = 0
        for topic, msg, t in bag.read_messages(topics=[rgb_topic, depth_topic, gps_topic]):
            
            if topic == gps_topic:
                # Salvataggio Posa [x, y, theta]
                self.curr_gps = (msg.x, msg.y, msg.theta)
                
            elif topic == depth_topic:
                self.curr_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                
            elif topic == rgb_topic:
                # Esegui lo step di elaborazione solo quando arriva un'immagine RGB (come nel nodo live)
                if self.curr_gps is None or self.curr_depth is None:
                    continue
                
                rgb_img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                
                # --- 1. DATA ASSOCIATION ---
                self._process_da_step(rgb_img, self.curr_depth, self.curr_gps)
                
                # --- 2. AGGIORNAMENTO BAYESIANO E MAPPA SCORES ---
                # Estraiamo gli score grezzi correnti
                raw_scores = self.da_scores[0, :]
                
                # Mapping da [0, 1] a [0.5, 1] come specificato dal nodo NMPC
                mapped_scores = np.zeros((self.num_total_cars, 2))
                mapped_scores[:, 0] = 0.5 + 0.5 * raw_scores
                mapped_scores[:, 1] = 1.0 - mapped_scores[:, 0]
                
                # Esegui Bayes Update
                self.beliefs_k = self.bayes(self.beliefs_k, ca.DM(mapped_scores))
                
                # --- 3. CALCOLO ENTROPIA ---
                entropy_k = self.entropy_entire_field(self.beliefs_k)
                entropy_val = float(ca.sum1(entropy_k).full().flatten()[0])
                
                # --- 4. SALVATAGGIO LOG ---
                self.time_history.append(t.to_sec())
                self.pose_history.append(self.curr_gps)
                self.entropy_history.append(entropy_val)
                self.lambda_history.append(self.beliefs_k.full().flatten().tolist())
                
                count += 1
                if count % 10 == 0:
                    print(f"Elaborati {count} frame. Entropia Globale Corrente: {entropy_val:.3f}")

        bag.close()
        self.save_csv()
        print("Elaborazione Offline Completata!")

    def _process_da_step(self, rgb, depth, gps):
        rx, ry, rtheta = gps
        img_w = rgb.shape[1]
        
        det = self.yolo.detect(rgb)
        cars_det = [d for d in det if int(d[5]) in COCO_CAR_CLASSES]
        persons = [d for d in det if int(d[5]) == COCO_PERSON]

        gated = []
        for c in cars_det:
            z = get_box_median_depth(depth, *c[:4])
            if z is not None and z <= self.max_depth_m and ((c[2] - c[0]) * (c[3] - c[1]) / float(rgb.shape[0]*rgb.shape[1])) >= 0.3:
                gated.append((c, z))

        matched = []
        for car, car_depth in gated:
            result = match_detection_to_car(
                car, car_depth, self.cars, rx, ry, rtheta, self.side, img_w, self.hfov_deg,
                self.gps_range_m, self.depth_tolerance_m, self.car_length_m)
            if result is not None:
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
            driver_half_ok = is_looking_at_driver_half(
                car, car_depth, target, rx, ry, rtheta, self.side, img_w,
                self.hfov_deg, self.car_length_m)
            
            uncertain = (target.visitable and not driver_half_ok) or not target.visitable

            if uncertain and best_person_conf is None:
                new = 0.0
            elif best_person_conf is not None:
                new = +(car_conf + best_person_conf) / 2.0
            else:
                new = -car_conf

            old = self.da_scores[0, target.score_col]
            self.da_scores[0, target.score_col] = update_score(
                old, new, self.ema_alpha_person, self.ema_alpha_no_person, uncertain=uncertain)

        # Fallback persone senza box auto
        matched_ids = set(best_by_id.keys())
        near_cars = [c for c in self.cars if c.distance_to(rx, ry) <= self.close_range_m]
        for near_car in near_cars:
            if near_car.id in matched_ids:
                continue

            near_x, near_y = car_near_point(near_car, rx, ry, self.car_length_m)
            expected_dist = math.hypot(near_x - rx, near_y - ry)

            if near_car.visitable:
                center_det = (img_w / 2.0, 0.0, img_w / 2.0, 0.0)
                driver_half_ok = is_looking_at_driver_half(
                    center_det, expected_dist, near_car, rx, ry, rtheta,
                    self.side, img_w, self.hfov_deg, self.car_length_m)
                if not driver_half_ok:
                    old = self.da_scores[0, near_car.score_col]
                    self.da_scores[0, near_car.score_col] = update_score(
                        old, 0.0, self.ema_alpha_person, self.ema_alpha_no_person, uncertain=True)
                    continue

            best_person_conf = None
            for p in persons:
                pz = get_box_median_depth(depth, *p[:4])
                if pz is None or abs(pz - expected_dist) > self.person_depth_tolerance_m:
                    continue
                pc = float(p[4])
                if best_person_conf is None or pc > best_person_conf:
                    best_person_conf = pc

            if best_person_conf is not None:
                old = self.da_scores[0, near_car.score_col]
                self.da_scores[0, near_car.score_col] = update_score(
                    old, +best_person_conf, self.ema_alpha_person, self.ema_alpha_no_person)

    def save_csv(self):
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_csv = os.path.join(self.output_dir, f"offline_da_mpc_{timestamp}_plot_data.csv")
        
        with open(plot_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            
            # Formattazione degli header coerente con NeuralMPCHusky
            trees_pos = np.array([[c.x, c.y, c.orientation_rad] for c in self.cars]).astype(np.float32)
            writer.writerow(["tree_positions"] + trees_pos.flatten().tolist())
            
            header = ["time", "x", "y", "theta", "entropy"]
            if self.lambda_history:
                header += [f"lambda_{i}" for i in range(len(self.lambda_history[0]))]
            writer.writerow(header)
            
            for i in range(len(self.time_history)):
                x, y, theta = self.pose_history[i]
                row = [self.time_history[i], x, y, theta, self.entropy_history[i]] + self.lambda_history[i]
                writer.writerow(row)
                
        print(f"Report delle metriche salvato correttamente in: {plot_csv}")

if __name__ == "__main__":
    
    # --- CONFIGURAZIONI UTENTE ---
    BAG_FILE = "/home/andre/esperimento_parcheggio_ws/nmpc.bag" 
    JSON_MAP = "/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/nmpc_ros/niccolo/map_results/car_map_final.json"
    YOLO_W = "/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/yolov7-ros/weights/yolov7.pt"
    OUTPUT_DIR = "./offline_results"
    
    processor = BagProcessor(
        bag_path=BAG_FILE,
        map_file_path=JSON_MAP,
        output_dir=OUTPUT_DIR,
        yolo_weights=YOLO_W
    )
    processor.run()