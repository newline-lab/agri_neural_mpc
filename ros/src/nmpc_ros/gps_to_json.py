#!/usr/bin/env python3

import rospy
import json
import sys
import select
import termios
import tty
from geometry_msgs.msg import Pose2D

class GpsToJsonNode:
    def __init__(self):
        # Inizializza il nodo
        rospy.init_node('gps_to_json_node', anonymous=True)
        
        # Sottoscrizione al topic
        self.subscriber = rospy.Subscriber('/gps_data', Pose2D, self.gps_callback)
        
        # Variabili di stato
        self.latest_pose = None
        self.cars_list = []
        self.current_id = 0
        self.json_filename = "/home/andre/esperimento_parcheggio_ws/src/agri_neural_mpc/ros/src/nmpc_ros/niccolo/map_results/car_map_final.json"
        
        # Salva le impostazioni del terminale per la lettura dei tasti
        self.settings = termios.tcgetattr(sys.stdin)

    def gps_callback(self, msg):
        # Aggiorna costantemente l'ultima posa ricevuta
        self.latest_pose = msg

    def get_key(self):
        # Funzione per leggere la pressione di un singolo tasto in modo non bloccante
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def save_to_json(self):
        # Costruisce il dizionario secondo la struttura richiesta
        data = {"cars": self.cars_list}
        
        # Salva su file con indentazione per renderlo leggibile
        with open(self.json_filename, 'w') as f:
            json.dump(data, f, indent=2)

    def run(self):
        rospy.loginfo("Nodo avviato. In attesa di dati su /gps_data...")
        rospy.loginfo("Premi la BARRA SPAZIATRICE per salvare la macchina come VISITABLE.")
        rospy.loginfo("Premi il tasto 'M' per salvare la macchina come NON VISITABLE.")
        rospy.loginfo("Premi 'q' o Ctrl+C per uscire.")
        
        while not rospy.is_shutdown():
            key = self.get_key()
            
            # Gestisce sia lo spazio (visitable) che 'm'/'M' (non visitable)
            if key == ' ' or key.lower() == 'm':
                if self.latest_pose is not None:
                    is_visitable = (key == ' ')
                    
                    # Estrae i dati e aggiunge il flag visitable
                    car_data = {
                        "id": self.current_id,
                        "x": round(self.latest_pose.x, 3),
                        "y": round(self.latest_pose.y, 3),
                        "orientation_rad": round(self.latest_pose.theta, 4),
                        "visitable": is_visitable
                    }
                    
                    self.cars_list.append(car_data)
                    self.current_id += 1
                    
                    self.save_to_json()
                    
                    stato = "VISITABLE" if is_visitable else "NON VISITABLE"
                    rospy.loginfo(f"Dato salvato [{stato}]! ID: {car_data['id']} | X: {car_data['x']} | Y: {car_data['y']}")
                else:
                    rospy.logwarn("Nessun dato ancora ricevuto sul topic /gps_data! Aspetta il primo messaggio.")
            
            elif key.lower() == 'q' or key == '\x03': # 'q' o Ctrl+C per uscire in modo pulito
                rospy.loginfo("Uscita in corso. File JSON finale salvato.")
                break

if __name__ == '__main__':
    try:
        node = GpsToJsonNode()
        node.run()
    except rospy.ROSInterruptException:
        pass