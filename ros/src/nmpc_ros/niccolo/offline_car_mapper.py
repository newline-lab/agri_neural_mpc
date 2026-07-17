#!/usr/bin/env python3
"""
offline_car_mapper.py
======================
Batch/offline version of car_mapper_node.py.

Instead of running the ROS node live against `rosbag play`, this script reads
one or more .bag files directly (rosbag Python API), replays the RGB/Depth/GPS
messages through the SAME detection + projection + clustering pipeline, and
accumulates ALL cars from ALL bags into a single set of CarTracks.

Why this is faster:
  - No roscore, no `rosbag play`, no real-time wall-clock throttling.
  - Bags are read back-to-back as fast as YOLO inference allows.
  - No per-bag JSON + manual merge/sort step: tracks live in one place and are
    finalized once, at the very end, exactly like `save_final_map()` does.

Usage:
    python3 offline_car_mapper.py \
        --weights best.pt \
        --bags left_side.bag right_side.bag \
        --output-dir ./out

Notes:
  - `--side` is intentionally NOT per-bag: per our earlier discussion, the
    camera mount side is fixed to the robot chassis, so it should be constant
    across every bag ("left" if that's the physical mount). The row flip
    between sides of the parking lot is handled by robot_theta (heading),
    which differs bag-to-bag as you drive the opposite direction.
  - GPS is tracked as "most recent message at or before the image timestamp",
    same semantics as the live node's self.robot_x/y/theta.

python3 offline_car_mapper.py  --bags /home/andre/esperimento_parcheggio_ws/bag_esperimenti/gruppo_a_sinistra.bag /home/andre/esperimento_parcheggio_ws/bag_esperimenti/gruppo_a_destra.bag /home/andre/esperimento_parcheggio_ws/bag_esperimenti/gruppo_b_sinistra.bag /home/andre/esperimento_parcheggio_ws/bag_esperimenti/gruppo_b_destra.bag   --side left   --output-dir map_results/ --weights results/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rosbag
from cv_bridge import CvBridge

# Reuse everything pure from the live node unchanged.
from car_map_node import (
    CLS_FRONT, CLS_NAMES,
    get_box_median_depth, filter_detections_by_depth,
    project_detection_to_world, CarTrack, add_detection_to_tracks,
    plot_car_map,
)

from ultralytics import YOLO


def sync_rgbd_gps(bag_path: str, rgb_topic: str, depth_topic: str,
                   gps_topic: str, slop: float = 0.05):
    """
    Single pass over one bag, yielding (rgb_msg, depth_msg, robot_x, robot_y,
    robot_theta) tuples using a simple nearest-neighbor sync: for every RGB
    message, pick the closest depth message within `slop` seconds, and the
    most recent GPS message at or before that time (mirrors the live node's
    ApproximateTimeSynchronizer + latest-GPS-wins behaviour).
    """
    depth_buffer: List[Tuple[float, object]] = []
    gps_latest: Optional[Tuple[float, float, float]] = None
    pending_rgb: List[Tuple[float, object]] = []

    with rosbag.Bag(bag_path) as bag:
        for topic, msg, t in bag.read_messages(
                topics=[rgb_topic, depth_topic, gps_topic]):
            ts = t.to_sec()

            if topic == gps_topic:
                gps_latest = (msg.x, msg.y, msg.theta)
                continue

            if topic == depth_topic:
                depth_buffer.append((ts, msg))
                # Keep buffer small; only need a short recent window.
                if len(depth_buffer) > 50:
                    depth_buffer.pop(0)
                continue

            if topic == rgb_topic:
                if gps_latest is None:
                    continue  # no pose yet, can't project
                # Find closest depth frame within slop
                best = None
                best_dt = slop
                for dts, dmsg in reversed(depth_buffer):
                    dt = abs(dts - ts)
                    if dt <= best_dt:
                        best = dmsg
                        best_dt = dt
                    if dts < ts - slop:
                        break
                if best is None:
                    continue
                rx, ry, rtheta = gps_latest
                yield msg, best, rx, ry, rtheta


def process_bag(bag_path: str, model: YOLO, bridge: CvBridge,
                 tracks: List[CarTrack], args) -> int:
    n_detections = 0
    for rgb_msg, depth_msg, rx, ry, rtheta in sync_rgbd_gps(
            bag_path, args.rgb_topic, args.depth_topic, args.gps_topic,
            slop=args.slop):

        rgb_img = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_img = bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        results = model(rgb_img, conf=args.conf_thresh, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        valid = filter_detections_by_depth(detections, depth_img,
                                            max_depth_m=args.max_depth_m)

        img_w = rgb_img.shape[1]
        for det_row, depth_m in valid:
            x1, _, x2, _ = det_row[:4]
            u = (x1 + x2) / 2.0
            bearing_deg = ((u / img_w) - 0.5) * args.hfov_deg
            if abs(bearing_deg) > args.max_bearing_deg:
                continue

            cx, cy = project_detection_to_world(
                det_row, depth_m, rx, ry, rtheta,
                args.side, img_w, hfov_deg=args.hfov_deg)

            if math.hypot(cx - rx, cy - ry) > args.gps_range_m:
                continue

            cls_id = int(det_row[5])
            conf = float(det_row[4])
            add_detection_to_tracks(
                tracks, cx, cy, cls_id, conf, rx, ry,
                cluster_radius_m=args.cluster_radius_m)
            n_detections += 1

    return n_detections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--bags", nargs="+", required=True,
                     help="One or more .bag files, processed in order.")
    ap.add_argument("--side", default="left", choices=["left", "right"],
                     help="Physical camera mount side. Keep fixed across "
                          "all bags; the row flip is handled by heading.")
    ap.add_argument("--rgb-topic", default="/camera/color/image_raw")
    ap.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    ap.add_argument("--gps-topic", default="/gps_data")
    ap.add_argument("--max-depth-m", type=float, default=3.5)
    ap.add_argument("--gps-range-m", type=float, default=4.0)
    ap.add_argument("--conf-thresh", type=float, default=0.5)
    ap.add_argument("--max-bearing-deg", type=float, default=15.0)
    ap.add_argument("--cluster-radius-m", type=float, default=1.0)
    ap.add_argument("--min-track-score", type=float, default=10.0)
    ap.add_argument("--hfov-deg", type=float, default=69.0)
    ap.add_argument("--front-offset-m", type=float, default=3.7)
    ap.add_argument("--slop", type=float, default=0.05)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    model = YOLO(args.weights)
    bridge = CvBridge()

    tracks: List[CarTrack] = []
    trajectory: List[Tuple[float, float]] = []

    for bag_path in args.bags:
        print(f"[offline_car_mapper] Processing {bag_path} ...")
        n = process_bag(bag_path, model, bridge, tracks, args)
        print(f"[offline_car_mapper]   -> {n} detections merged, "
              f"{len(tracks)} tracks so far")

        # Collect trajectory too, for the combined plot.
        with rosbag.Bag(bag_path) as bag:
            for _, msg, _ in bag.read_messages(topics=[args.gps_topic]):
                if (not trajectory or
                        math.hypot(msg.x - trajectory[-1][0],
                                   msg.y - trajectory[-1][1]) >= 0.05):
                    trajectory.append((msg.x, msg.y))

    # ── Finalize exactly like save_final_map() in the live node ────────────
    for t in tracks:
        t.x, t.y = t.front_position(args.front_offset_m)

    kept = [t for t in tracks if t.score >= args.min_track_score]
    kept.sort(key=lambda t: t.x)
    for new_id, t in enumerate(kept):
        t.id = new_id

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    png_path = str(out / "car_map_final.png")
    plot_car_map(trajectory, tracks, png_path,
                 min_track_score=args.min_track_score)
    print(f"[offline_car_mapper] Plot saved: {png_path}")

    state = {
        "trajectory": [[round(x, 3), round(y, 3)] for x, y in trajectory],
        "cars": sorted([t.to_dict() for t in kept], key=lambda c: c["id"]),
    }
    json_path = str(out / "car_map_final.json")
    with open(json_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[offline_car_mapper] JSON saved: {json_path} "
          f"({len(kept)} cars kept, {len(tracks) - len(kept)} discarded)")


if __name__ == "__main__":
    main()
