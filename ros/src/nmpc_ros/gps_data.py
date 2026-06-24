#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose2D
import tf.transformations

class FakeRTKGps:
    def __init__(self):
        # Inizializza il nodo ROS
        rospy.init_node('fake_rtk_gps_node', anonymous=True)

        # Nome del modello su Gazebo (di default 'husky', ma puoi cambiarlo)
        self.robot_model_name = rospy.get_param('~robot_name', 'husky') 
        # NOTA: spesso in Gazebo l'Husky viene chiamato '/' se è l'unico robot, 
        # oppure 'husky' / 'husky_robot'. Modifica di conseguenza.

        # Publisher per i dati GPS simulati
        self.gps_pub = rospy.Publisher('gps_data', Pose2D, queue_size=10)

        # Subscriber agli stati di Gazebo
        self.gazebo_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_callback)

        rospy.loginfo("Nodo Fake RTK GPS avviato. In attesa del modello '%s' su Gazebo...", self.robot_model_name)

    def gazebo_callback(self, msg):
        try:
            # Trova l'indice dell'Husky all'interno della lista dei modelli di Gazebo
            index = msg.name.index(self.robot_model_name)
            
            # Estrai la posa (posizione e orientamento)
            husky_pose = msg.pose[index]

            # 1. Estrai le coordinate X e Y
            x = husky_pose.position.x
            y = husky_pose.position.y

            # 2. Estrai l'orientamento
            # Gazebo fornisce l'orientamento in Quaternioni, ma a te serve in Radianti sul piano (Yaw)
            quaternion = (
                husky_pose.orientation.x,
                husky_pose.orientation.y,
                husky_pose.orientation.z,
                husky_pose.orientation.w
            )
            
            # Converti il quaternione in angoli di Eulero (roll, pitch, yaw)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            yaw = euler[2] # Lo 'yaw' è la rotazione attorno all'asse Z (il tuo orientamento sul piano)

            # 3. Costruisci il messaggio e pubblicalo
            gps_msg = Pose2D()
            gps_msg.x = x
            gps_msg.y = y
            gps_msg.theta = yaw

            self.gps_pub.publish(gps_msg)

        except ValueError:
            # Questo blocco viene eseguito se il modello non è ancora stato caricato in Gazebo
            # per evitare che il nodo crashi all'avvio.
            pass

if __name__ == '__main__':
    try:
        FakeRTKGps()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass