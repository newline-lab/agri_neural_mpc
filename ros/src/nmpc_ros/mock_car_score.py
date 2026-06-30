#!/usr/bin/env python
import os
import re
import rospy
import torch
import math
import numpy as np
import tf
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose, Point, Quaternion
import threading
from geometry_msgs.msg import Pose, Point, Quaternion, Pose2D

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
        # x = torch.sigmoid(self.out_layer(x))   # bound to (0,1)
        return x

class CarScoresMock:
    def __init__(self):
        rospy.init_node('car_scores_mock_node', anonymous=True)
        
        # [X, Y, Theta_target]
        self.cars_pos = np.array([
            [-4.0, -1.0, 0.0],
            [-4.0, 5.0, 0.0],
            [-3.5, 10.0, 0.0],
            [4.0, 15.0, -np.pi]
        ], dtype=np.float32)
        
        self.gt_ids = [0, 0, 0, 0] # 1: Ripe (Verde), 0: Raw (Rossa)
        self.num_cars = len(self.cars_pos)
        
        self.robot_pos = None
        self.robot_yaw = 0.0
        self.dt = 0.2 # 5 Hz
        
        # Inizializzazione Reti Neurali
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        rospy.loginfo("[Mock] Caricamento reti neurali...")
        
        # Modello
        for label in ['car', 'car']:
            model = MultiLayerPerceptron(input_dim=3, hidden_size=64, hidden_layers=3)
            model_path = self.get_latest_best_model(label)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[label] = model
        
        # Publisher e Subscriber
        self.sub_gps = rospy.Subscriber('/gps_data', Pose2D, self.gps_callback)
        self.pub_scores = rospy.Publisher('tree_scores', Float32MultiArray, queue_size=10)
        
        rospy.loginfo("[+] Nodo Mock Auto-Percettivo avviato. In attesa di odometria...")

        # Spawn in Gazebo su un thread separato
        threading.Thread(target=self.spawn_cars_in_gazebo).start()

    def get_latest_best_model(self, cls=''):
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
        self.robot_pos = np.array([msg.x, msg.y])
        self.robot_yaw = msg.theta
        
    def run(self):
        rate = rospy.Rate(int(1/self.dt))
        scores = np.zeros((self.num_cars, 2), dtype=np.float32)
        
        while not rospy.is_shutdown():
            if self.robot_pos is not None:
                # Dati globali del Robot
                X_r = self.robot_pos[0]
                Y_r = self.robot_pos[1]
                theta_r = self.robot_yaw

                with torch.no_grad():
                    for i in range(self.num_cars):
                        # Coordinate e orientamento globale della macchina (Target)
                        X_t = self.cars_pos[i, 0]
                        Y_t = self.cars_pos[i, 1]
                        theta_t = self.cars_pos[i, 2] # Orientamento asse X della macchina in terna mondo
                        # Distanza globale tra macchina e centro del robot
                        dX = X_t - X_r
                        dY = Y_t - Y_r
                        # Trasformazione in terna robot
                        x_rel = dX * math.cos(theta_r) + dY * math.sin(theta_r)
                        y_rel = -dX * math.sin(theta_r) + dY * math.cos(theta_r)
                        # Calcolo azimut
                        theta_y_robot = theta_r + (math.pi / 2.0)
                        azimuth_raw = theta_y_robot - theta_t + math.pi
                        azimuth_norm = math.atan2(math.sin(azimuth_raw), math.cos(azimuth_raw))
                        
                        # Input Rete Neurale
                        nn_input = torch.tensor([x_rel, y_rel, azimuth_norm], dtype=torch.float32).unsqueeze(0).to(self.device)
                        
                        logit = self.models['car'](nn_input)
                        p_correct = logit.item()
                        
                        rospy.loginfo(f"[Mock] Auto {i} | Terna Robot -> dX: {x_rel:.2f}m, dY: {y_rel:.2f}m | Azimuth: {azimuth_norm:.2f}rad | Output: {p_correct:.4f}")
            
                        if p_correct > 0.81:
                            scores[i, 0] = 1.0
                        else:
                            scores[i, 0] = 0.0
            
            else:
                rospy.logwarn_throttle(2.0, "[Mock] Nessun dato odometrico in arrivo.")

            # Standardizzazione output array per NMPC
            msg = Float32MultiArray()
            msg.layout.dim.append(MultiArrayDimension(label="rows", size=self.num_cars, stride=self.num_cars * 2))
            msg.layout.dim.append(MultiArrayDimension(label="cols", size=2, stride=2))
            
            # scores[i, 0] = p_correct (inserire qui logica assegnazione risultati)
            
            msg.data = scores.flatten().tolist()
            
            self.pub_scores.publish(msg)
            rate.sleep()

 # scores = np.zeros((self.num_cars, 2), dtype=np.float32)

    def spawn_cars_in_gazebo(self):
        rospy.loginfo("[Mock] In attesa del servizio di spawn di Gazebo...")
        service_name = '/gazebo/spawn_sdf_model'
        try:
            rospy.wait_for_service(service_name, timeout=10.0)
            spawn_model_client = rospy.ServiceProxy(service_name, SpawnModel)
            
            # Dimensioni
            l_car = 3.46
            w_car = 1.62
            h_car = 1.46
            
            # Il centro X dell'intera macchina è a -1.83 per avere la punta a -0.10
            # (Punta = -0.10, Centro = -0.10 - (3.46/2) = -1.83)
            center_x = -0.10 - (l_car / 2.0)
            
            for i, pos in enumerate(self.cars_pos):
                model_name = f"mock_citroen_c1_detailed_{i}"
                
                # Scegliamo il colore del "paraurti anteriore" per indicare la classe
                id_color = "0 1 0 1" if self.gt_ids[i] == 1 else "1 0 0 1" # Verde Ripe, Rosso Raw
                
                sdf_string = f"""
                <sdf version='1.6'>
                  <model name='{model_name}'>
                    <static>true</static>
                    <link name='chassis'>
                    
                      <collision name='car_collision'>
                        <pose>{center_x} 0 {h_car/2.0} 0 0 0</pose>
                        <geometry><box><size>{l_car} {w_car} {h_car}</size></box></geometry>
                      </collision>

                      <visual name='body_translucent'>
                        <pose>{center_x} 0 {h_car/2.0} 0 0 0</pose>
                        <geometry><box><size>{l_car} {w_car} {h_car}</size></box></geometry>
                        <material>
                          <ambient>1 1 1 0.2</ambient> <diffuse>1 1 1 0.2</diffuse>
                        </material>
                      </visual>
                      
                      <visual name='interior_compartment'>
                        <pose>-1.6 0 0.5 0 0 0</pose>
                        <geometry><box><size>1.8 1.4 0.6</size></box></geometry>
                        <material>
                          <ambient>0.2 0.2 0.2 1</ambient> <diffuse>0.2 0.2 0.2 1</diffuse>
                        </material>
                      </visual>

                      <visual name='driver_torso'>
                        <pose>-1.3 0.4 0.8 0 0 0</pose>
                        <geometry><cylinder><radius>0.1</radius><length>0.6</length></cylinder></geometry>
                        <material>
                          <ambient>0 1 0 1</ambient> <diffuse>0 1 0 1</diffuse>
                        </material>
                      </visual>
                      <visual name='driver_head'>
                        <pose>-1.3 0.4 1.15 0 0 0</pose>
                        <geometry><box><size>0.15 0.15 0.15</size></box></geometry>
                        <material>
                          <ambient>0 1 0 1</ambient> <diffuse>0 1 0 1</diffuse>
                        </material>
                      </visual>

                      <visual name='steering_wheel'>
                        <pose>-0.9 0.4 0.8 0 1.57 0</pose>
                        <geometry><cylinder><radius>0.18</radius><length>0.05</length></cylinder></geometry>
                        <material>
                          <ambient>0 0 0 1</ambient> <diffuse>0 0 0 1</diffuse>
                        </material>
                      </visual>
                      <visual name='dashboard'>
                        <pose>-0.7 0 0.7 0 0 0</pose>
                        <geometry><box><size>0.1 1.4 0.2</size></box></geometry>
                        <material>
                          <ambient>0.1 0.1 0.1 1</ambient> <diffuse>0.1 0.1 0.1 1</diffuse>
                        </material>
                      </visual>

                      <visual name='front_indicator_class'>
                        <pose>-0.10 0 0.5 0 0 0</pose> 
                        <geometry><box><size>0.01 1.5 0.3</size></box></geometry>
                        <material>
                          <ambient>{id_color}</ambient> <diffuse>{id_color}</diffuse>
                        </material>
                      </visual>
                    </link>
                  </model>
                </sdf>
                """
                
                # --- Configurazione Posa (Posizione + Orientamento) ---
                pose = Pose()
                pose.position = Point(x=pos[0], y=pos[1], z=0.0)
                
                # Conversione dell'angolo di imbardata (yaw) in quaternione
                # Roll=0.0, Pitch=0.0, Yaw=pos[2]
                q = tf.transformations.quaternion_from_euler(0.0, 0.0, pos[2])
                pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                
                spawn_model_client(
                    model_name=model_name,
                    model_xml=sdf_string,
                    robot_namespace="",
                    initial_pose=pose,
                    reference_frame="world"
                )
            rospy.loginfo("[Mock] Le Citroën C1 simulate sono state posizionate con l'orientamento corretto.")
            
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr(f"[Mock] Impossibile spawnare le auto: {e}")

if __name__ == '__main__':
    try:
        mock = CarScoresMock()
        mock.run()
    except rospy.ROSInterruptException:
        pass