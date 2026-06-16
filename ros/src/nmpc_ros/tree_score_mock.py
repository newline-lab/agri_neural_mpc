#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

def tree_scores_simulator():
    rospy.init_node('tree_scores_mock_node', anonymous=True)
    pub = rospy.Publisher('tree_scores', Float32MultiArray, queue_size=10)
    
    rate = rospy.Rate(5) 
    
    # Deve corrispondere esattamente al numero di alberi del codice principale
    num_trees = 2 
    
    # Ground truth definito nell'NMPC (0: raw, 1: ripe)
    # trees_gt_id 
    gt_ids = [0, 1]
    
    rospy.loginfo("[-] Nodo Mock tree_scores avviato con successo.")
    start_time = rospy.get_time()

    while not rospy.is_shutdown():
        elapsed = rospy.get_time() - start_time
        scores = np.zeros((num_trees, 2), dtype=np.float32)
        
        # ----------------------------------------------------------------------
        # SCEGLI LA MODALITÀ DI TEST QUI SOTTO
        # ----------------------------------------------------------------------
        mode = "statica"  # Cambia in "statica" se vuoi zero informazioni costanti
        
        if mode == "statica":
            # L'entropia rimarrà massima (1.0). Il robot si muoverà senza mai fermarsi.
            scores = np.ones((num_trees, 2), dtype=np.float32) * 0.5
            
        elif mode == "dinamica":
            # Simula una convergenza lineare verso il valore reale in 45 secondi.
            # Questo permette di testare se l'MPC riduce l'entropia e si arresta.
            tempo_di_convergenza = 45.0 
            factor = min(1.0, elapsed / tempo_di_convergenza)
            
            # La probabilità corretta sale da 0.5 (buio totale) a 0.95 (certezza quasi assoluta)
            p_correct = 0.5 + 0.45 * factor
            p_wrong = 1.0 - p_correct
            
            for i, gt in enumerate(gt_ids):
                if gt == 1:  # Ripe (Maturo)
                    scores[i] = [p_wrong, p_correct]
                else:        # Raw (Grezzo)
                    scores[i] = [p_correct, p_wrong]
        
        # ----------------------------------------------------------------------
        # Costruzione del messaggio ROS conforme alla callback dell'NMPC
        # ----------------------------------------------------------------------
        msg = Float32MultiArray()
        
        # Definiamo esplicitamente il layout bidimensionale [10 righe x 2 colonne]
        msg.layout.dim.append(MultiArrayDimension(label="rows", size=num_trees, stride=num_trees * 2))
        msg.layout.dim.append(MultiArrayDimension(label="cols", size=2, stride=2))
        
        # Appiattiamo la matrice in una lista monodimensionale di 20 elementi
        msg.data = scores.flatten().tolist()
        
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        tree_scores_simulator()
    except rospy.ROSInterruptException:
        pass