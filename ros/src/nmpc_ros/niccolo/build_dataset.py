"""

Processing dei dati estratti:

Per ogni cartella di snapshot all’interno di --data_dir:
  - Legge x, y dal file cube_pose__fused_pose.txt (posizione del cubo rispetto alla
    telecamera, espressa nel sistema di riferimento robot/telecamera)
  - Legge theta (in gradi) dal file cube_pose__azimuth_deg.txt 
  - Ruota (x, y) in modo che il risultato della detection di ogni tag
    sia sempre espresso nel sistema di riferimento del tag10 stesso: x = distanza lungo la
    normale, y = offset laterale lungo la faccia del tag10. Ciò rende il set di dati
    coerente indipendentemente dal tag di quale faccia la telecamera abbia visto
    (tag10/11/12/13), poiché senza questa correzione x/y sono espressi in un
    sistema di riferimento ruotato diverso per ciascuna faccia, disperdendo punti che dovrebbero
    giacere su un unico arco coerente attorno al cubo.
  - Inverte (cos_theta, sin_theta) — equivalente ad azimut+180°
    — in modo che il vettore theta punti dalla telecamera verso il cubo
    anziché dal cubo verso la telecamera, spostando quindi il riferimento dell'angolo sulla camera
    --> posi<ione + orientamento espressi rispetto alla telecamera

Colonne finali dataset:
  x | y | cos_theta | sin_theta | yolo_conf

"""

import os
import re
import math
import argparse
import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_fused_pose(filepath):
    """
    Extract position x and y from cube_pose__fused_pose.txt.
    Looks for lines like:  '    x: 3.471886...'  under the 'position:' block.
    """
    x, y = None, None
    in_position_block = False

    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()

            if stripped == 'position:':
                in_position_block = True
                continue

            if in_position_block:
                # Exit position block when we hit theta or another top-level key
                if stripped == 'theta:' or (stripped.endswith(':') and not stripped.startswith('x') and not stripped.startswith('y') and not stripped.startswith('z')):
                    in_position_block = False
                    continue

                m = re.match(r'^x:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', stripped)
                if m:
                    x = float(m.group(1))

                m = re.match(r'^y:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', stripped)
                if m:
                    y = float(m.group(1))

            if x is not None and y is not None:
                break

    if x is None or y is None:
        raise ValueError(f"Could not parse x, y from {filepath}")

    return x, y


def parse_azimuth(filepath):

    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            m = re.match(r'^data:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', stripped)
            if m:
                return float(m.group(1))

    raise ValueError(f"Could not parse azimuth from {filepath}")


def parse_person_confidence(label_filepath):

    best_conf = 0.0

    with open(label_filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if int(parts[0]) == 0:  # person class
                conf = float(parts[5])
                if conf > best_conf:
                    best_conf = conf

    return best_conf


def correct_xy_to_tag10_frame(x, y, az_deg):

    az_rad = math.radians(az_deg)
    c, s = math.cos(az_rad), math.sin(az_rad)
    x_corr = x * c - y * s
    y_corr = x * s + y * c
    return x_corr, y_corr


def theta_toward_cube(az_deg):

    az_rad = math.radians(az_deg)
    return -math.cos(az_rad), -math.sin(az_rad)


def find_label_file(folder_name, labels_dir):

    for f in labels_dir.glob('*.txt'):
        if folder_name in f.name:
            return f
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build training dataset CSV from pose files and filtered YOLO labels."
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/simulator/Desktop/neural_mpc_arlotta/paper_extension/'
                'agri_neural_mpc/surrogate_model_training/fronte/',
        help='Root folder containing one subfolder per snapshot'
    )
    parser.add_argument(
        '--labels_dir',
        type=str,
        default='/home/simulator/Desktop/neural_mpc_arlotta/paper_extension/'
                'agri_neural_mpc/surrogate_model_training/fronte_detections/labels_filtered/',
        help='Folder containing filtered YOLO label .txt files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/simulator/Desktop/neural_mpc_arlotta/paper_extension/'
                'agri_neural_mpc/surrogate_model_training/dataset.csv',
        help='Output CSV file path'
    )
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    labels_dir = Path(args.labels_dir)
    output     = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    if not folders:
        print(f"No subfolders found in {data_dir}")
        return

    rows = []
    skipped = []

    for folder in folders:
        folder_name = folder.name

        # --- Pose files ---
        pose_file    = folder / 'cube_pose__fused_pose.txt'
        azimuth_file = folder / 'cube_pose__azimuth_deg.txt'

        if not pose_file.exists():
            print(f"  [SKIP] Missing pose file in: {folder_name}")
            skipped.append(folder_name)
            continue

        if not azimuth_file.exists():
            print(f"  [SKIP] Missing azimuth file in: {folder_name}")
            skipped.append(folder_name)
            continue

        try:
            x, y    = parse_fused_pose(pose_file)
            az_deg  = parse_azimuth(azimuth_file)
        except ValueError as e:
            print(f"  [SKIP] Parse error in {folder_name}: {e}")
            skipped.append(folder_name)
            continue

        cos_orient, sin_orient = theta_toward_cube(az_deg)

        x, y = correct_xy_to_tag10_frame(x, y, az_deg)

        # --- Label file ---
        label_file = find_label_file(folder_name, labels_dir)

        if label_file is None:
            print(f"  [WARN] No label file found for folder: {folder_name} → yolo_conf = 0.0")
            yolo_conf = 0.0
        else:
            yolo_conf = parse_person_confidence(label_file)

        rows.append({
            'x':               x,
            'y':               y,
            'cos_theta': cos_orient,
            'sin_theta': sin_orient,
            'yolo_conf':       yolo_conf,
        })

    # --- Write CSV ---
    fieldnames = ['x', 'y', 'cos_theta', 'sin_theta', 'yolo_conf']
    with open(output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*50}")
    print(f"Folders processed : {len(rows) + len(skipped)}")
    print(f"Rows written      : {len(rows)}")
    print(f"Skipped (errors)  : {len(skipped)}")
    conf_nonzero = sum(1 for r in rows if r['yolo_conf'] > 0)
    print(f"Person detected   : {conf_nonzero}  (yolo_conf > 0)")
    print(f"No person         : {len(rows) - conf_nonzero}  (yolo_conf = 0)")
    print(f"Output saved to   : {output}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()