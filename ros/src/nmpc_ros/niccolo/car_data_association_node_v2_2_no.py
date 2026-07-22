#!/usr/bin/env python3
import os
import re
import json
import math
import sys
import numpy as np
import threading

import rospy
import cv2
import torch
import tf
from cv_bridge import CvBridge

import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

# ============================================================================
# YOLOv7 Import & Class[cite: 1]
# ============================================================================
_YOLOV7_REPO_ROOT = "/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/yolov7-ros/src"
if _YOLOV7_REPO_ROOT not in sys.path:
    sys.path.insert(0, _YOLOV7_REPO_ROOT)

from models.experimental import attempt_load
from utils.general import non_max_suppression

COCO_PERSON = 0
COCO_CAR_CLASSES = {2, 7}

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


# ============================================================================
# Rete Neurale Surrogata (Dal Mock)[cite: 2]
# ============================================================================
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=64, hidden_layers=3):
        super().__init__()
        in_features = input_dim if input_dim != 3 else input_dim + 1
        self.input_layer = torch.nn.Linear(in_features, hidden_size)
        self.hidden_layer = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        self.out_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.shape[-1] == 3:
            sin_cos = torch.cat([torch.sin(x[..., -1:]), torch.cos(x[..., -1:])], dim=-1)
            x = torch.cat([x[..., :-1], sin_cos], dim=-1)
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x


# ============================================================================
# Helper Geometrici (Dal Data Association)[cite: 1]
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

def update_score(old_score, new_score, alpha_person, alpha_no_person, uncertain=False):
    if uncertain:
        return 0.0
    else:
        alpha = alpha_person if new_score >= 0 else alpha_no_person
        return alpha * new_score + (1 - alpha) * old_score


# ============================================================================
# Main Node
# ============================================================================
class HybridDataAssociation:
    def __init__(self):
        rospy.init_node('hybrid_association_node', anonymous=True)

        # ----------------------------------------------------------------------
        # 1. Caricamento Mappa Auto (Stile Mock)[cite: 2]
        # ----------------------------------------------------------------------
        file_path = "/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/nmpc_ros/niccolo/map_results/car_map_final.json"
        
        self.cars_data = [] # Lista di dizionari per mantenere gli ID
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                cars = json.load(f).get("cars", [])
            cars_sorted = sorted(cars, key=lambda c: c["id"])
            for c in cars_sorted:
                self.cars_data.append({
                    "id": c["id"],
                    "x": c["x"],
                    "y": c["y"],
                    "orientation_rad": c["orientation_rad"]
                })
            rospy.loginfo(f"[Node] Caricate {len(self.cars_data)} auto dal JSON.")
        else:
            # Fallback hardcoded se il file non esiste
            self.cars_data = [
                {"id": 0, "x": 43.04, "y": -2.149, "orientation_rad": 1.4337},
                {"id": 1, "x": 49.701, "y": -7.093, "orientation_rad": -1.6995},
                {"id": 2, "x": 52.743, "y": -7.233, "orientation_rad": -1.7331}
            ]
            rospy.logwarn("[Node] JSON non trovato. Usato Fallback Hardcoded.")

        self.num_cars = len(self.cars_data)
        
        # Popolazione matrice (2, N) - verrà aggiornata solo la riga 0
        self.scores = np.zeros((2, self.num_cars), dtype=np.float32)

        # ----------------------------------------------------------------------
        # 2. Configurazione Stato Robot e Parametri
        # ----------------------------------------------------------------------
        self.robot_pos = None
        self.robot_yaw = 0.0
        
        self.side = rospy.get_param("~side", "right")
        self.hfov_deg = rospy.get_param("~hfov_deg", 69.0)
        self.max_depth_m = rospy.get_param("~max_depth_m", 10)
        self.assoc_max_m = rospy.get_param("~assoc_max_m", 5)
        self.cont_thresh = rospy.get_param("~containment_thresh", 0.7)
        self.ema_alpha_person = rospy.get_param("~ema_alpha_person", 1.0)
        self.ema_alpha_no_person = rospy.get_param("~ema_alpha_no_person", 0.1)

        # ----------------------------------------------------------------------
        # 3. Caricamento Modelli AI (MLP + YOLO)[cite: 1, 2]
        # ----------------------------------------------------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # MLP (Filtro visibilità auto)
        self.mlp_model = MultiLayerPerceptron(input_dim=3, hidden_size=64, hidden_layers=3)
        try:
            model_path = self.get_latest_best_model('car')
            self.mlp_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.mlp_model.to(self.device)
            self.mlp_model.eval()
            rospy.loginfo("[Node] Rete MLP caricata con successo.")
        except Exception as e:
            rospy.logerr(f"[Node] Impossibile caricare rete MLP: {e}")

        # YOLOv7
        yolo_weights = rospy.get_param("~weights", '/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/yolov7-ros/weights/yolov7.pt')
        self.yolo = YoloV7(yolo_weights, conf_thresh=0.4, iou_thresh=0.45, img_size=640, device=self.device)

        # ----------------------------------------------------------------------
        # 4. Subscribers e Publishers[cite: 1, 2]
        # ----------------------------------------------------------------------
        self.sub_gps = rospy.Subscriber('/gps_data', Pose2D, self.gps_callback)
        self.pub_scores = rospy.Publisher('/parking/scores', Float32MultiArray, queue_size=10)
        
        self.bridge = CvBridge()
        rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")

        self.pub_debug = rospy.Publisher("/data_association/debug_image", Image, queue_size=2)
        
        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.camera_callback)
        
        rospy.loginfo("[+] Nodo ibrido avviato con successo. In attesa di immagini e odometria...")

    def get_latest_best_model(self, cls=''):
        """Stessa logica di caricamento del mock[cite: 2]"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, "models", cls)
        if not os.path.exists(model_dir):
            model_dir = os.path.join(script_dir, "..", "models", cls) 
            
        model_files = [f for f in os.listdir(model_dir) if re.match(r"best_model_epoch_(\d+)\.pth", f)]
        if not model_files:
            raise FileNotFoundError(f"Nessun modello trovato in {model_dir}")
        latest_model = max(model_files, key=lambda x: int(re.match(r"best_model_epoch_(\d+)\.pth", x).group(1)))
        return os.path.join(model_dir, latest_model)

    def gps_callback(self, msg):
        """Salvataggio asincrono dei dati GPS[cite: 2]"""
        self.robot_pos = np.array([msg.x, msg.y])
        self.robot_yaw = msg.theta

    def camera_callback(self, rgb_msg: Image, depth_msg: Image):
        """Motore principale unito: MLP + Detection + Data Association"""
        if self.robot_pos is None:
            return

        X_r = self.robot_pos[0]
        Y_r = self.robot_pos[1]
        theta_r = self.robot_yaw

        active_cars = []
        visible_status = {}

        # --- A. Inferenza MLP sulle macchine ---
        for car in self.cars_data:
            X_t = car["x"]
            Y_t = car["y"]
            theta_t = car["orientation_rad"]

            dX = X_r - X_t
            dY = Y_r - Y_t
            x_rel = dX * math.cos(theta_t) + dY * math.sin(theta_t)
            y_rel = -dX * math.sin(theta_t) + dY * math.cos(theta_t)
            
            theta_y_robot = theta_r + (math.pi / 2.0)
            azimuth_raw = theta_y_robot - theta_t + math.pi
            azimuth_norm = math.atan2(math.sin(azimuth_raw), math.cos(azimuth_raw))
            
            nn_input = torch.tensor([x_rel, y_rel, azimuth_norm], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logit = self.mlp_model(nn_input)
                p_correct = logit.item()

            # Usiamo una soglia conservativa per la rete
            if p_correct > 0.4:
                active_cars.append(car)
                visible_status[car["id"]] = True
            else:
                visible_status[car["id"]] = False

        # --- B. Detection YOLO sull'immagine ---
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        img_w = rgb.shape[1]

        det = self.yolo.detect(rgb)
        cars_det = [d for d in det if int(d[5]) in COCO_CAR_CLASSES]
        persons = [d for d in det if int(d[5]) == COCO_PERSON]

        gated = []
        for c in cars_det:
            z = get_box_median_depth(depth, *c[:4])
            # MODIFICA CHIAVE: Se la camera di profondità fallisce (z = None) 
            # o l'auto è lontana, non la scartiamo. Le diamo una profondità "fittizia"
            # o usiamo la massima, così la rete neurale può comunque fare l'associazione!
            if z is None or z <= 0:
                z_eff = 3.5 # Fallback di salvataggio
            else:
                z_eff = min(z, self.max_depth_m)
            gated.append((c, z_eff))

        # --- C. Associazione guidata dall'IA (No limiti rigidi) ---
        matched = []
        for car_box, car_depth in gated:
            if not active_cars:
                break
                
            cx, cy = project_detection_to_world(
                car_box, car_depth, X_r, Y_r, theta_r, self.side, img_w, hfov_deg=self.hfov_deg)
            
            # Ci fidiamo della rete: cerchiamo la macchina più vicina SOLO tra quelle 
            # che l'intelligenza artificiale ha etichettato come "Visibili"
            target = min(active_cars, key=lambda c: math.hypot(c["x"] - cx, c["y"] - cy))
            match_quality = math.hypot(target["x"] - cx, target["y"] - cy)
            
            # Il limite di associazione ora è larghissimo (es. 5 metri) per 
            # assecondare la certezza della rete neurale al posto della geometria pura
            if match_quality <= self.assoc_max_m:
                matched.append((car_box, car_depth, target, match_quality))

        best_by_id = {}
        for m in matched:
            tid = m[2]["id"]
            if tid not in best_by_id or m[3] < best_by_id[tid][3]:
                best_by_id[tid] = m

        # --- D. Aggiornamento Punteggi SULLA PRIMA RIGA ---
        matched_ids = set(best_by_id.keys())
        
        # Variabili per il debug visivo
        best_car_box = None
        any_occupied = False

        for idx, car in enumerate(self.cars_data):
            c_id = car["id"]
            
            if c_id in matched_ids:
                car_box, car_depth, target, match_quality = best_by_id[c_id]
                
                best_person_conf = None
                for p in persons:
                    if containment_ratio(p[:4], car_box[:4]) >= self.cont_thresh:
                        pc = float(p[4])
                        if best_person_conf is None or pc > best_person_conf:
                            best_person_conf = pc

                car_conf = float(car_box[4])
                new = +(car_conf + best_person_conf) / 2.0 if best_person_conf is not None else -car_conf
                
                old = self.scores[0, idx]
                self.scores[0, idx] = update_score(old, new, self.ema_alpha_person, self.ema_alpha_no_person, uncertain=False)
                
                # Salviamo i dati per il debug
                if best_person_conf is not None:
                    any_occupied = True
                    best_car_box = car_box
                elif best_car_box is None:
                    best_car_box = car_box
            
            # Se la rete la vede (active_cars) ma YOLO ha ciccato l'auto...
            elif visible_status.get(c_id, False) and persons:
                 best_person_conf = max([float(p[4]) for p in persons])
                 old = self.scores[0, idx]
                 self.scores[0, idx] = update_score(old, best_person_conf, self.ema_alpha_person, self.ema_alpha_no_person, uncertain=False)

            # Decadimento
            elif not visible_status.get(c_id, False):
                old = self.scores[0, idx]
                self.scores[0, idx] = update_score(old, 0.0, self.ema_alpha_person, self.ema_alpha_no_person, uncertain=True)

        self._publish_scores()
        
        # PUBBLICAZIONE IMMAGINE DI DEBUG
        if best_car_box is not None:
            self._publish_debug(rgb, best_car_box, persons, any_occupied)   

    def _publish_scores(self):
        """Formatta l'array bidimensionale per l'MP[cite: 1, 2]C"""
        n_rows, n_cols = self.scores.shape
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="rows", size=n_rows, stride=n_rows * n_cols),
            MultiArrayDimension(label="cols", size=n_cols, stride=n_cols),
        ]
        msg.layout.data_offset = 0
        msg.data = self.scores.flatten().tolist()
        self.pub_scores.publish(msg)

    def _publish_debug(self, rgb, car, persons, occupied):
        if self.pub_debug.get_num_connections() == 0:
            return
        img = rgb.copy()
        
        # Disegna l'auto
        x1, y1, x2, y2 = map(int, car[:4])
        col = (0, 0, 255) if occupied else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"car {car[4]:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                    
        # Disegna le persone
        for p in persons:
            px1, py1, px2, py2 = map(int, p[:4])
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 1)
            
        # Pubblica su ROS
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))

if __name__ == '__main__':
    try:
        node = HybridDataAssociation()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass