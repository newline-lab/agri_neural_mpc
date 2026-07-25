import rosbag
import cv2
from cv_bridge import CvBridge

input_bag = '/home/andre/esperimento_parcheggio_ws/nmpc.bag'
output_bag = '/home/andre/esperimento_parcheggio_ws/nmpc_decompressed.bag'
target_topic = '/camera/color/image_raw/compressed'
new_topic = '/camera/color/image_raw'

bridge = CvBridge()

print(f"Inizio la conversione da {input_bag} a {output_bag}...")
print("Questa operazione richiederà del tempo date le dimensioni del file.")

with rosbag.Bag(output_bag, 'w') as outbag:
    for topic, msg, t in rosbag.Bag(input_bag).read_messages():
        if topic == target_topic:
            try:
                # Decomprime l'immagine da CompressedImage a un array OpenCV
                cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                
                # Riconverte in sensor_msgs/Image
                raw_msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                raw_msg.header = msg.header
                
                # Scrive l'immagine raw sul nuovo topic
                outbag.write(new_topic, raw_msg, t)
            except Exception as e:
                print(f"Errore nella decompressione di un frame: {e}")
        
        # Filtra i metadata dinamici del vecchio topic compresso, ormai inutili
        elif topic in [target_topic + '/parameter_descriptions', target_topic + '/parameter_updates']:
            continue
            
        else:
            # Ricopia tutti gli altri topic (depth, gps, log, ecc.) inalterati
            outbag.write(topic, msg, t)

print("Conversione completata con successo!")