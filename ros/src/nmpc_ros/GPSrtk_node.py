#!/usr/bin/env python3
# GPSrtk_node.py — il tuo GPSrtk.py originale + SOLO l'aggiunta della
# pubblicazione ROS. Struttura, stampe (❌🛰️🟡🟢🟠, 🧭, contatore byte)
# e flusso invariati.
#
# Topics pubblicati (aggiunta):
#   /gps_data         geometry_msgs/Pose2D  (x,y locali in metri; theta = yaw ENU rad)
#   /gps/fix          sensor_msgs/NavSatFix (lat/lon/alt grezzi + stato RTK)
#   /gps/rtk_quality  std_msgs/Int32        (campo qualita' GGA: 0,1,2,4,5)
#
# Parametri ROS (~privati, tutti con default = valori originali):
#   ~origin_lat, ~origin_lon  : ancora del frame locale (default: riferimento
#                               usato per parking_slots.json)
#   ~serial_port, ~baud, ~ntrip_server, ~ntrip_port, ~ntrip_mount,
#   ~ntrip_user, ~ntrip_pass

import os
import sys
import socket
import base64
import math
import time
from threading import Thread, Lock
from serial import Serial

import utm

# --- AGGIUNTA ROS ---
import rospy
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Int32
# --------------------

# --- CONFIGURAZIONE ---
PORTA_SERIALE = "/dev/ttyUSB0"
BAUD_RATE = 115200

NTRIP_SERVER = "gnss-rtk.regione.abruzzo.it"
NTRIP_PORT = 2101
MOUNTPOINT = "0_RTCM_MSM"
USER = "newline"
PASSWORD = "Newline!"

# Ancora del frame locale (stesso riferimento di parking_slots.json)
ORIGIN_LAT = 41.85626425142204
ORIGIN_LON = 12.469038428190489
# ----------------------

# --- AGGIUNTA ROS: stato condiviso per la pubblicazione ---
_pub_pose = None
_pub_fix = None
_pub_quality = None
_origin_x = None
_origin_y = None
_zone_n = None
_zone_l = None
_heading_lock = Lock()
_heading_deg = None      # gradi CW dal Nord vero (doppia antenna)
# ----------------------------------------------------------

def nmeatodecimal(valore, direzione):
    """Converte le coordinate NMEA (DDMM.MMMMM) in Gradi Decimali (DD.DDDDDD)"""
    if not valore or not direzione:
        return "N/D"
    try:
        split_idx = 2 if direzione in ['N', 'S'] else 3
        gradi = int(valore[:split_idx])
        minuti = float(valore[split_idx:])
        decimale = gradi + (minuti / 60.0)
        if direzione in ['S', 'W']:
            decimale = -decimale
        return f"{decimale:.8f}"
    except Exception:
        return "Errore Conversione"

def interpreta_stato_rtk(linea_nmea):
    """Analizza la stringa GNGGA/GPGGA per estrarre coordinate e stato del Fix RTK"""
    try:
        if b'GGA' in linea_nmea:
            stringa = linea_nmea.decode('utf-8', errors='ignore').strip()
            parti = stringa.split(',')

            if len(parti) > 9:
                ora = parti[1]
                lat_raw = parti[2]
                lat_dir = parti[3]
                lon_raw = parti[4]
                lon_dir = parti[5]
                qualita = parti[6]
                satelliti = parti[7]
                altitudine = parti[9]

                Stati = {
                    '0': "❌ NO FIX (Nessun satellite o segnale insufficiente)",
                    '1': "🛰️ SINGLE FIX (Posizione autonoma, precisione ~2-5 metri)",
                    '2': "🟡 DGPS/FLOAT (Correzioni in corso, convergenza RTK...)",
                    '4': "🟢 RTK FIXED (Precisione Centimetrica Agganciata!)",
                    '5': "🟠 RTK FLOAT (Flusso RTK instabile o scarso segnale)"
                }

                stato_attuale = Stati.get(qualita, f"Codice sconosciuto ({qualita})")
                lat_dec = nmeatodecimal(lat_raw, lat_dir)
                lon_dec = nmeatodecimal(lon_raw, lon_dir)

                print("\n" + "="*60)
                print(f" [UM982 RTK DATA]")
                print(f" STATO FIX:  {stato_attuale}")
                print(f" SATELLITI:  {satelliti}")
                print(f" LATITUDINE: {lat_dec}  (NMEA: {lat_raw} {lat_dir})")
                print(f" LONGITUDINE:{lon_dec}  (NMEA: {lon_raw} {lon_dir})")
                print(f" ALTITUDINE: {altitudine} metri")

                # --- AGGIUNTA ROS: pubblicazione ---
                pubblica_ros(lat_dec, lon_dec, qualita, altitudine)
                # ------------------------------------
    except Exception:
        pass

def interpreta_heading(linea_nmea):
    """Analizza la stringa HDT per estrarre la direzione (Heading) della doppia antenna"""
    global _heading_deg
    try:
        if b'HDT' in linea_nmea or b'THS' in linea_nmea:
            stringa = linea_nmea.decode('utf-8', errors='ignore').strip()
            parti = stringa.split(',')
            # Verifica che il campo del valore esista e non sia vuoto
            if len(parti) > 1 and parti[1]:
                heading = parti[1]
                print(f" 🧭 ORIENTAMENTO: {heading}° rispetto al Nord")
                print("="*60)

                # --- AGGIUNTA ROS: memorizza heading per /gps_data.theta ---
                try:
                    with _heading_lock:
                        _heading_deg = float(heading)
                except ValueError:
                    pass
                # -----------------------------------------------------------
            else:
                # Disattiva questa stampa se diventa troppo invadente in fase di avvio
                print(f" 🧭 ORIENTAMENTO: In calcolo... (Attesa segnale su entrambe le antenne)")
                print("="*60)
    except Exception:
        pass

# --- AGGIUNTA ROS: funzione di pubblicazione -------------------------------
def pubblica_ros(lat_dec, lon_dec, qualita, altitudine):
    """Pubblica /gps_data (Pose2D locale), /gps/fix e /gps/rtk_quality.
    Chiamata da interpreta_stato_rtk() per ogni GGA valida."""
    if _pub_pose is None:
        return
    try:
        lat = float(lat_dec)
        lon = float(lon_dec)
    except (ValueError, TypeError):
        return   # "N/D" o "Errore Conversione": non pubblicare
    try:
        q = int(qualita) if qualita else 0
    except ValueError:
        q = 0
    try:
        alt = float(altitudine) if altitudine else 0.0
    except ValueError:
        alt = 0.0

    stamp = rospy.Time.now()

    # qualita' RTK
    _pub_quality.publish(Int32(q))

    # NavSatFix (geodetico grezzo)
    fix = NavSatFix()
    fix.header.stamp = stamp
    fix.header.frame_id = "gps_link"
    fix.latitude, fix.longitude, fix.altitude = lat, lon, alt
    fix.status.status = (NavSatStatus.STATUS_GBAS_FIX if q in (4, 5)
                         else NavSatStatus.STATUS_FIX if q > 0
                         else NavSatStatus.STATUS_NO_FIX)
    fix.status.service = NavSatStatus.SERVICE_GPS
    _pub_fix.publish(fix)

    # Pose2D nel frame locale (stesso frame di parking_slots.json)
    x_utm, y_utm, _, _ = utm.from_latlon(lat, lon,
                                         force_zone_number=_zone_n,
                                         force_zone_letter=_zone_l)
    pose = Pose2D()
    pose.x = x_utm - _origin_x
    pose.y = y_utm - _origin_y
    with _heading_lock:
        h = _heading_deg
    pose.theta = math.radians(90-h) if h is not None else 0.0  # NED->ENU
    pose.theta = math.atan2(math.sin(pose.theta), math.cos(pose.theta))                     
    _pub_pose.publish(pose)
# ----------------------------------------------------------------------------

def leggi_da_hardware_e_invia_a_ntrip(ser, sock):
    """Legge l'output dell'UM982, estrae i dati e invia la posizione al server NTRIP"""
    try:
        while True:
            if ser.in_waiting:
                linea = ser.readline()

                if b'GGA' in linea:
                    interpreta_stato_rtk(linea)
                    sock.sendall(linea)
                elif b'HDT' in linea or b'THS' in linea:
                    interpreta_heading(linea)

            time.sleep(0.02)
    except Exception as e:
        print(f"\n[ERRORE] Interruzione canale di lettura hardware: {e}")

def main():
    global _pub_pose, _pub_fix, _pub_quality, _origin_x, _origin_y, _zone_n, _zone_l
    global PORTA_SERIALE, BAUD_RATE, NTRIP_SERVER, NTRIP_PORT, MOUNTPOINT, USER, PASSWORD

    print("--- Avvio Client RTK Python Master per UM982 ---")

    # --- AGGIUNTA ROS: init nodo, parametri e publisher ---
    rospy.init_node("gps_rtk_node", disable_signals=True)
    PORTA_SERIALE = rospy.get_param("~serial_port", PORTA_SERIALE)
    BAUD_RATE     = rospy.get_param("~baud", BAUD_RATE)
    NTRIP_SERVER  = rospy.get_param("~ntrip_server", NTRIP_SERVER)
    NTRIP_PORT    = rospy.get_param("~ntrip_port", NTRIP_PORT)
    MOUNTPOINT    = rospy.get_param("~ntrip_mount", MOUNTPOINT)
    USER          = rospy.get_param("~ntrip_user", USER)
    PASSWORD      = rospy.get_param("~ntrip_pass", PASSWORD)
    origin_lat    = rospy.get_param("~origin_lat", ORIGIN_LAT)
    origin_lon    = rospy.get_param("~origin_lon", ORIGIN_LON)

    _origin_x, _origin_y, _zone_n, _zone_l = utm.from_latlon(origin_lat, origin_lon)
    print(f"Origine frame locale: lat={origin_lat:.8f} lon={origin_lon:.8f} "
          f"(UTM {_origin_x:.2f}, {_origin_y:.2f}, zona {_zone_n}{_zone_l})")

    _pub_pose    = rospy.Publisher("/gps_data", Pose2D, queue_size=10)
    _pub_fix     = rospy.Publisher("/gps/fix", NavSatFix, queue_size=10)
    _pub_quality = rospy.Publisher("/gps/rtk_quality", Int32, queue_size=10)
    # -------------------------------------------------------

    if not os.access(PORTA_SERIALE, os.R_OK | os.W_OK):
        print(f"Errore permessi su {PORTA_SERIALE}. Esegui: sudo chmod 666 {PORTA_SERIALE}")
        sys.exit(1)

    # 1. Apertura connessione seriale e configurazione
    try:
        ser = Serial(PORTA_SERIALE, BAUD_RATE, timeout=1)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"Connesso all'UM982 su {PORTA_SERIALE}")

        # COMANDI CORRETTI PER UNICORE UM982
        print("Configurazione RTK + Doppia Antenna (Heading) in corso...")
        comandi_rtk = [
            "UNLOGALL\r\n",
            "GNGGA 0.1\r\n",                              # Output di posizione
            "GPHDT 0.1\r\n",                              # Output di Heading rispetto al Nord Vero
            "INTERFACEMODE COM1 RTCM3 NONE OFF\r\n",
            "INTERFACEMODE COM2 RTCM3 NONE OFF\r\n",
            "INTERFACEMODE COM3 RTCM3 NONE OFF\r\n",
            "SAVECONFIG\r\n"
        ]
        for cmd in comandi_rtk:
            ser.write(cmd.encode('utf-8'))
            time.sleep(0.1)
        print("Configurazione hardware completata!")

    except Exception as e:
        print(f"Impossibile aprire la seriale o configurare: {e}")
        sys.exit(1)

    # 2. Connessione NTRIP
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((NTRIP_SERVER, NTRIP_PORT))

        auth_str = f"{USER}:{PASSWORD}"
        auth_encoded = base64.b64encode(auth_str.encode('utf-8')).decode('utf-8')

        richiesta = (
            f"GET /{MOUNTPOINT} HTTP/1.1\r\n"
            f"User-Agent: NTRIP PythonClient/1.0\r\n"
            f"Authorization: Basic {auth_encoded}\r\n"
            f"Accept: */*\r\n\r\n"
        )
        sock.sendall(richiesta.encode('utf-8'))

        risposta = sock.recv(1024).decode('utf-8', errors='ignore')
        if "200 OK" not in risposta:
            print(f"Errore Server NTRIP:\n{risposta.strip()}")
            sock.close()
            ser.close()
            sys.exit(1)
        print("Connessione NTRIP stabilita con successo!")

    except Exception as e:
        print(f"Errore di rete NTRIP: {e}")
        ser.close()
        sys.exit(1)

    # Risveglio rete NTRIP
    print("In attesa della prima stringa di posizione dall'UM982 per attivare l'RTK...")
    timeout_sveglia = time.time()
    while True:
        if ser.in_waiting:
            prima_linea = ser.readline()
            if b'GGA' in prima_linea:
                sock.sendall(prima_linea)
                print("[SBLOCCO] Prima posizione inviata al server! Il flusso sta per partire.")
                break
        if time.time() - timeout_sveglia > 10:
            print("[AVVISO] L'UM982 non risponde ancora. Invio posizione backup...")
            posizione_backup = b"$GNGGA,120000.00,4225.0000,N,01412.0000,E,1,12,1.0,100.0,M,48.0,M,,*61\r\n"
            sock.sendall(posizione_backup)
            break
        time.sleep(0.1)

    # 3. Avvio Thread parallelo
    thread_hardware = Thread(target=leggi_da_hardware_e_invia_a_ntrip, args=(ser, sock), daemon=True)
    thread_hardware.start()

    print("\n--- SISTEMA RTK OPERATIVO ---")
    try:
        byte_ricevuti = 0
        ultimo_controllo = time.time()

        while True:
            dati_rtcm = sock.recv(2048)
            if not dati_rtcm:
                print("\n[AVVISO] Il server NTRIP ha chiuso la connessione.")
                break

            ser.write(dati_rtcm)
            byte_ricevuti += len(dati_rtcm)

            ora_attuale = time.time()
            if ora_attuale - ultimo_controllo >= 4:
                bps = (byte_ricevuti * 8) / (ora_attuale - ultimo_controllo)
                print(f" -> Ricezione Correzioni: {byte_ricevuti} Bytes (~{int(bps)} bps) alimentati all'hardware", end='\r')
                byte_ricevuti = 0
                ultimo_controllo = ora_attuale

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente. Chiusura in corso...")
    finally:
        sock.close()
        ser.close()
        print("\nPorte chiuse correttamente. Script terminato.")

if __name__ == "__main__":
    main()
