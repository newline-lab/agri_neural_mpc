#!/usr/bin/env python
import rospy
import numpy as np
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose, Point
import threading

class TreeScoresMock:
    def __init__(self):
        rospy.init_node('tree_scores_mock_node', anonymous=True)
        
        # 1. Configurazione Alberi (Deve combaciare ESATTAMENTE con l'NMPC)
        #self.trees_pos = np.array([
        #    [-3.5,  2], [-3.5, 7.5], [3.5, 2], [3.5, 7.5]
        # ], dtype=np.float32)
        self.trees_pos = np.array([
            [-6.0, 0]
        ], dtype=np.float32)
        
        self.gt_ids = [0, 1, 0, 0]
        self.num_trees = len(self.trees_pos)
        
        # 2. Parametri di prossimità e accumulo informativo
        self.PROXIMITY_THRESHOLD = 2.8   # Raggio di attivazione (metri). Sotto questa distanza il robot "vede" l'albero.
        self.TIME_FOR_FULL_INFO = 5.0    # Tempo necessario (secondi) stando vicino all'albero per scansionarlo al 100%
        
        # Array per tracciare il tempo speso vicino a ciascun albero (inizializzato a 0)
        self.proximity_time = np.zeros(self.num_trees, dtype=np.float32)
        
        self.robot_pos = None
        self.dt = 0.2 # 5 Hz
        
        # 3. Publisher e Subscriber ROS
        self.sub_odom = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.pub_scores = rospy.Publisher('tree_scores', Float32MultiArray, queue_size=10)
        
        rospy.loginfo("[+] Nodo Mock Prossimità avviato. In attesa di odometria...")

        threading.Thread(target=self.spawn_trees_in_gazebo).start()


    def odom_callback(self, msg):
        """ Riceve la posizione in tempo reale dell'Husky """
        self.robot_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])

    def run(self):
        rate = rospy.Rate(int(1/self.dt))
        
        while not rospy.is_shutdown():
            scores = np.ones((self.num_trees, 2), dtype=np.float32) * 0.5
            
            if self.robot_pos is not None:
                # Calcola la distanza tra il robot e TUTTI gli alberi contemporaneamente
                distances = np.linalg.norm(self.trees_pos - self.robot_pos, axis=1)
                
                # Aggiorna i contatori di prossimità
                for i in range(self.num_trees):
                    if distances[i] < self.PROXIMITY_THRESHOLD:
                        self.proximity_time[i] += self.dt
                        rospy.loginfo_throttle(1.0, f"[Mock] Robot vicino all'albero {i} (Distanza: {distances[i]:.2f}m). Progresso scansione: {min(100.0, (self.proximity_time[i]/self.TIME_FOR_FULL_INFO)*100):.1f}%")
                    
                    # Calcola il livello di certezza per questo specifico albero [0.0 , 1.0]
                    factor = min(1.0, self.proximity_time[i] / self.TIME_FOR_FULL_INFO)
                    
                    # Converte il fattore in probabilità (da 0.5 a 0.98 di certezza)
                    p_correct = 0.5 + 0.5 * factor
                    p_wrong = 1.0 - p_correct
                    
                    if self.gt_ids[i] == 1: # Ripe
                        scores[i] = [p_wrong, p_correct]
                    else:                  # Raw
                        scores[i] = [p_correct, p_wrong]
            else:
                rospy.logwarn_throttle(2.0, "[Mock] Nessun dato odometrico ricevuto. Invio score vuoti [0.5, 0.5]")

            # Costruzione del messaggio Float32MultiArray standardizzato per l'NMPC
            msg = Float32MultiArray()
            msg.layout.dim.append(MultiArrayDimension(label="rows", size=self.num_trees, stride=self.num_trees * 2))
            msg.layout.dim.append(MultiArrayDimension(label="cols", size=2, stride=2))


            # NON CAMBIA
            scores = np.ones((self.num_trees, 2), dtype=np.float32) * 0.5 


            msg.data = scores.flatten().tolist()
            
            self.pub_scores.publish(msg)
            rate.sleep()

    def spawn_trees_in_gazebo(self):
        """Richiama il servizio di Gazebo per spawnare cilindri nelle posizioni degli alberi"""
        rospy.loginfo("[Mock] In attesa del servizio di spawn di Gazebo...")
        service_name = '/gazebo/spawn_sdf_model'
        try:
            rospy.wait_for_service(service_name, timeout=10.0)
            spawn_model_client = rospy.ServiceProxy(service_name, SpawnModel)
            
            for i, pos in enumerate(self.trees_pos):
                model_name = f"mock_tree_{i}"
                
                # Definizione di un cilindro semplice in formato SDF (altezza 1.5m, raggio 0.2m)
                # Include sia la geometria visiva (rossa/verde a seconda del tipo) che quella di collisione
                color = "0 1 0 1" if self.gt_ids[i] == 1 else "1 0 0 1" # Verde per Ripe, Rosso per Raw
                sdf_string = f"""
                <sdf version='1.6'>
                  <model name='{model_name}'>
                    <static>true</static>
                    <link name='link'>
                      <collision name='collision'>
                        <geometry><cylinder><radius>0.2</radius><length>1.5</length></cylinder></geometry>
                      </collision>
                      <visual name='visual'>
                        <geometry><cylinder><radius>0.2</radius><length>1.5</length></cylinder></geometry>
                        <material>
                          <ambient>{color}</ambient>
                          <diffuse>{color}</diffuse>
                        </material>
                      </visual>
                    </link>
                  </model>
                </sdf>
                """
                
                # Posizionamento del cilindro (Z = 0.75m per appoggiarlo perfettamente sul terreno)
                pose = Pose()
                pose.position = Point(x=pos[0], y=pos[1], z=0.75)
                
                # Chiamata al servizio
                spawn_model_client(
                    model_name=model_name,
                    model_xml=sdf_string,
                    robot_namespace="",
                    initial_pose=pose,
                    reference_frame="world"
                )
            rospy.loginfo("[Mock] Tutti i cilindri-albero sono stati spawnati in Gazebo.")
            
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr(f"[Mock] Impossibile spawnare i modelli in Gazebo: {e}")

if __name__ == '__main__':
    try:
        mock = TreeScoresMock()
        mock.run()
    except rospy.ROSInterruptException:
        pass