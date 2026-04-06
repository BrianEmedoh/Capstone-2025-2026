#!/usr/bin/env python3
"""
detect.py  --  Arena detection for the A* Navigator vision system
=================================================================
Takes a single photo of the arena and produces:
    boundingboxes.csv   -- obstacle positions/sizes for the A* navigator
    robot_pose.txt      -- car position and heading for the A* navigator
    detection_preview.jpg -- annotated image for visual confirmation

Self-calibrating -- the 4 corner markers (IDs 0-3) must be visible in
every photo.  No fixed camera position or height required.

Marker assignments (DICT_4X4_50):
    ID 0  -- corner: top-left      (-1.5, +0.6)
    ID 1  -- corner: top-right     (+1.5, +0.6)
    ID 2  -- corner: bottom-left   (-1.5, -0.6)
    ID 3  -- corner: bottom-right  (+1.5, -0.6)
    ID 4  -- car
    ID 5  -- obstacle 1
    ID 6  -- obstacle 2

Usage:
    python3 detect.py --image arena_photo.jpg

Output files are written to the same directory as the script unless
--out-dir is specified.
"""

import sys
import os
import math
import csv
import numpy as np
import cv2
from cv2 import aruco

# =====================================================================
# Configuration
# =====================================================================
ARUCO_DICT_NAME = aruco.DICT_4X4_50

# Corner marker world coordinates (metres, arena centre = origin)
CORNER_WORLD = {
    0: (-1.5,  0.6),   # top-left
    1: ( 1.5,  0.6),   # top-right
    2: (-1.5, -0.6),   # bottom-left
    3: ( 1.5, -0.6),   # bottom-right
}

CAR_ID        = 4
OBSTACLE_IDS  = {5, 6}

# Physical size of each printed obstacle marker in metres.
# Measure the black border edge-to-edge on your printed marker.
# This is used to estimate the obstacle bounding box size.
MARKER_SIZE_M = 0.10   # 10 cm -- adjust to match your printed size

# The obstacle bounding box is set to a fixed size since a flat marker
# on the floor cannot convey object height/depth.
# Set these to the approximate physical size of your obstacles.
OBSTACLE_W_M  = 0.15   # width  (side to side) in metres
OBSTACLE_L_M  = 0.24   # length (front to back) in metres

# Output filenames
CSV_FILE      = "boundingboxes.csv"
POSE_FILE     = "robot_pose.txt"
PREVIEW_FILE  = "detection_preview.jpg"

# =====================================================================
# Helpers
# =====================================================================
def marker_centre_px(corners_entry):
    """Return float pixel (x, y) centre of one detected marker."""
    return corners_entry[0].mean(axis=0)


def marker_rotation_deg(corners_entry):
    """
    Estimate in-plane rotation of a marker from its corner positions.
    Returns angle in degrees, 0 = marker top edge pointing right (+X world).
    """
    c = corners_entry[0]
    # Vector from corner 0 (top-left) to corner 1 (top-right) in pixel space
    dx = c[1][0] - c[0][0]
    dy = c[1][1] - c[0][1]
    return math.degrees(math.atan2(dy, dx))


def pixel_to_world(H, px, py):
    """Apply homography H: pixel (px,py) -> world (wx, wy) in metres."""
    pt  = np.array([[[float(px), float(py)]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, H)
    return float(res[0][0][0]), float(res[0][0][1])


def world_to_pixel(H_inv, wx, wy):
    """Apply inverse homography: world (wx,wy) -> pixel (px, py)."""
    pt  = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, H_inv)
    return int(res[0][0][0]), int(res[0][0][1])


def compute_homography(corners_list, ids):
    """
    Find the 4 corner markers and compute pixel->world homography.
    Returns (H, detected_dict) or raises RuntimeError if any corner is missing.
    detected_dict maps marker_id -> corners_entry for all detected markers.
    """
    detected = {int(np.array(ids[i]).flat[0]): corners_list[i]
                for i in range(len(ids))}
    missing  = [cid for cid in CORNER_WORLD if cid not in detected]
    if missing:
        raise RuntimeError(
            f"Corner marker(s) {missing} not detected. "
            f"Make sure all 4 corner markers (IDs 0-3) are fully visible "
            f"and not obscured or cut off."
        )

    src_pts, dst_pts = [], []
    for cid, world_xy in CORNER_WORLD.items():
        cx, cy = marker_centre_px(detected[cid])
        src_pts.append([cx, cy])
        dst_pts.append(list(world_xy))

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography computation failed.")
    return H, detected


def estimate_world_size(H, corners_entry, fallback_w, fallback_l):
    """
    Return the configured obstacle dimensions (OBSTACLE_W_M, OBSTACLE_L_M).
    The ArUco marker pixel size is NOT used for the obstacle dimensions --
    the marker is just a locator tag and can be much smaller than the actual
    obstacle sitting on top of it.  The user sets the real physical size via
    OBSTACLE_W_M and OBSTACLE_L_M at the top of the script.
    """
    return fallback_w, fallback_l


# =====================================================================
# Drawing helpers
# =====================================================================
def draw_corner_marker(image, marker_id, corners_entry, colour=(0, 220, 220)):
    pts = corners_entry[0].astype(int)
    cv2.polylines(image, [pts], True, colour, 2)
    cx, cy = marker_centre_px(corners_entry).astype(int)
    cv2.circle(image, (cx, cy), 5, colour, -1)
    cv2.putText(image, f"C{marker_id}", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)


def draw_obstacle(image, marker_id, cx_px, cy_px, w_px, l_px, angle_deg, world_xy):
    colour = (0, 140, 255)
    cv2.circle(image, (cx_px, cy_px), 10, colour, 2)
    cv2.circle(image, (cx_px, cy_px), 3,  colour, -1)

    # Draw rotated bounding box
    rect  = ((cx_px, cy_px), (int(w_px), int(l_px)), angle_deg)
    box   = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(image, [box], 0, colour, 2)

    label = (f"OBS {marker_id}  "
             f"({world_xy[0]:+.2f},{world_xy[1]:+.2f})m  "
             f"{angle_deg:.0f}deg")
    cv2.putText(image, label, (cx_px + 12, cy_px),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)


def draw_car(image, cx_px, cy_px, heading_deg, world_xy):
    colour = (0, 220, 60)
    cv2.circle(image, (cx_px, cy_px), 14, colour, 3)
    cv2.circle(image, (cx_px, cy_px), 4,  colour, -1)

    # Heading arrow
    arrow_len = 40
    ex = int(cx_px + arrow_len * math.cos(math.radians(heading_deg)))
    ey = int(cy_px + arrow_len * math.sin(math.radians(heading_deg)))
    cv2.arrowedLine(image, (cx_px, cy_px), (ex, ey), colour, 3, tipLength=0.35)

    label = (f"CAR  ({world_xy[0]:+.2f},{world_xy[1]:+.2f})m  "
             f"hdg={heading_deg:.0f}deg")
    cv2.putText(image, label, (cx_px + 16, cy_px - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def draw_arena_grid(image, H_inv, h_img, w_img):
    """Draw the arena boundary and 0.5m grid on the preview image."""
    arena_world = [(-1.5, 0.6), (1.5, 0.6), (1.5, -0.6), (-1.5, -0.6)]
    pts = np.array([world_to_pixel(H_inv, *p) for p in arena_world])
    cv2.polylines(image, [pts], True, (0, 220, 220), 2)

    for x in np.arange(-1.5, 1.51, 0.5):
        p1 = world_to_pixel(H_inv, x,  0.6)
        p2 = world_to_pixel(H_inv, x, -0.6)
        cv2.line(image, p1, p2, (160, 160, 160), 1)
    for y in np.arange(-0.6, 0.61, 0.3):
        p1 = world_to_pixel(H_inv, -1.5, y)
        p2 = world_to_pixel(H_inv,  1.5, y)
        cv2.line(image, p1, p2, (160, 160, 160), 1)

    origin = world_to_pixel(H_inv, 0, 0)
    cv2.drawMarker(image, origin, (0, 220, 220), cv2.MARKER_CROSS, 30, 2)


# =====================================================================
# Main
# =====================================================================
def main():
    # -- Interactive image selection 
    # Works when run directly in an IDE (Thonny, VS Code, etc.)
    # as well as from the terminal.

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Show image files in the script's folder so user knows what is available
    image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    available  = sorted([
        f for f in os.listdir(script_dir)
        if os.path.splitext(f)[1] in image_exts
    ])

    print("=" * 55)
    print("  A* Navigator -- Arena Detection")
    print("=" * 55)

    if available:
        print("\nImage files found in script folder:")
        for name in available:
            print(f"    {name}")
    else:
        print("\nNo image files found in the script folder.")
        print(f"   Place your photo in:  {script_dir}")

    print()
    image_input = input("Enter image filename (e.g. arena_photo.jpg): ").strip()

    # Accept just the filename OR a full path
    if os.path.isabs(image_input):
        image_path = image_input
    else:
        image_path = os.path.join(script_dir, image_input)

    # Strip accidental quotes the user may have pasted
    image_path = image_path.strip("'\"")

    if not os.path.isfile(image_path):
        print(f"\nImage not found: {image_path!r}")
        print("    Check the filename and try again.")
        input("\nPress Enter to exit.")
        sys.exit(1)

    out_dir = script_dir
    os.makedirs(out_dir, exist_ok=True)

    # -- Load image 
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path!r}")
        print("    Make sure the file is a JPG or PNG and is not corrupted.")
        input("\nPress Enter to exit.")
        sys.exit(1)

    h_img, w_img = image.shape[:2]
    print(f"\nLoaded: {os.path.basename(image_path)}  ({w_img}x{h_img} px)")

    # -- Detect all ArUco markers 
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_NAME)

    def try_detect(img, params):
        detector = aruco.ArucoDetector(aruco_dict, params)
        c, i, _ = detector.detectMarkers(img)
        return c, i

    def merge_detections(results, offset_x=0, offset_y=0):
        """
        Merge multiple detection passes into one dict keyed by marker ID.
        offset_x/y shift pixel coordinates back when detecting on a crop.
        """
        seen = {}
        for corners_list, ids in results:
            if ids is None:
                continue
            for i, id_arr in enumerate(ids):
                mid = int(np.array(id_arr).flat[0])
                if mid not in seen:
                    # Shift corners back to full-image space if from a crop
                    c = corners_list[i].copy()
                    c[0][:, 0] += offset_x
                    c[0][:, 1] += offset_y
                    seen[mid] = c
        return seen

    # Build detector parameter sets
    p1 = aruco.DetectorParameters()

    p2 = aruco.DetectorParameters()
    p2.adaptiveThreshWinSizeMin  = 3
    p2.adaptiveThreshWinSizeMax  = 53
    p2.adaptiveThreshWinSizeStep = 4

    p3 = aruco.DetectorParameters()
    p3.minMarkerPerimeterRate      = 0.01
    p3.maxMarkerPerimeterRate      = 10.0
    p3.polygonalApproxAccuracyRate = 0.08
    p3.errorCorrectionRate         = 1.0

    # Contrast-enhanced image (helps washed-out or shadowed markers)
    gray         = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced     = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Aggressive CLAHE -- higher clip limit for very dark or uneven corners
    clahe_strong  = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
    enhanced_str  = clahe_strong.apply(gray)
    enhanced_str_bgr = cv2.cvtColor(enhanced_str, cv2.COLOR_GRAY2BGR)

    # Gamma-brightened image -- lifts dark corners without blowing out centre
    def gamma_correct(img, gamma=0.5):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, table)

    brightened = gamma_correct(image, gamma=0.45)

    # Full-image passes -- original, enhanced, strongly enhanced, brightened
    all_found = {}
    for img_variant in [image, enhanced_bgr, enhanced_str_bgr, brightened]:
        for params in [p1, p2, p3]:
            found = merge_detections([try_detect(img_variant, params)])
            all_found.update({k: v for k, v in found.items()
                              if k not in all_found})

    # Quadrant passes -- crop each quarter of the image and detect separately.
    # Larger overlap (25%) ensures corner markers right at the edge are
    # fully captured in at least one quadrant crop.
    overlap = 0.25   # increased from 0.15 -- captures edge markers more reliably
    qh = int(h_img * (0.5 + overlap))
    qw = int(w_img * (0.5 + overlap))

    quadrants = [
        (0,              0,              qw, qh),        # top-left
        (max(0,w_img-qw),0,              w_img, qh),     # top-right
        (0,              max(0,h_img-qh),qw, h_img),     # bottom-left
        (max(0,w_img-qw),max(0,h_img-qh),w_img, h_img), # bottom-right
    ]

    for (x0, y0, x1, y1) in quadrants:
        crop = image[y0:y1, x0:x1]
        # Upscale so markers appear larger -- increased target to 1600px
        scale = max(1.0, 1600 / max(crop.shape[:2]))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LINEAR)
            sx, sy = scale, scale
        else:
            sx, sy = 1.0, 1.0

        crop_gray      = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_clahe     = cv2.cvtColor(
            cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(crop_gray),
            cv2.COLOR_GRAY2BGR)
        crop_clahe_str = cv2.cvtColor(
            cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4,4)).apply(crop_gray),
            cv2.COLOR_GRAY2BGR)
        crop_bright    = gamma_correct(crop, gamma=0.45)

        for img_variant in [crop, crop_clahe, crop_clahe_str, crop_bright]:
            for params in [p1, p2, p3]:
                raw = merge_detections([try_detect(img_variant, params)])
                for mid, c in raw.items():
                    if mid not in all_found:
                        c_full = c.copy()
                        c_full[0][:, 0] = c_full[0][:, 0] / sx + x0
                        c_full[0][:, 1] = c_full[0][:, 1] / sy + y0
                        all_found[mid] = c_full

    if not all_found:
        print("No ArUco markers detected in any pass.")
        print("     Check: photo in focus? Even lighting? Markers fully visible?")
        sys.exit(1)

    # Convert merged dict back to list format expected by compute_homography
    corners_list = [all_found[k] for k in sorted(all_found)]
    ids          = [np.array([[k]]) for k in sorted(all_found)]

    detected_ids = sorted(all_found.keys())
    print(f"   Detected IDs: {detected_ids}")

    # -- Compute self-calibrating homography from corner markers 
    try:
        H, detected = compute_homography(corners_list, ids)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    H_inv = np.linalg.inv(H)
    print("Homography computed from corner markers")

    # -- Preview canvas 
    preview = image.copy()
    draw_arena_grid(preview, H_inv, h_img, w_img)

    # Draw corner markers on preview
    for cid in CORNER_WORLD:
        draw_corner_marker(preview, cid, detected[cid])

    # -- Process obstacle markers 
    obstacle_rows = []

    for obs_id in sorted(OBSTACLE_IDS):
        if obs_id not in detected:
            print(f"Obstacle marker ID {obs_id} not detected -- skipping")
            continue

        corners_entry = detected[obs_id]
        cx_px, cy_px  = marker_centre_px(corners_entry)
        wx, wy        = pixel_to_world(H, cx_px, cy_px)

        # Estimate physical size from pixel marker size
        w_m, l_m = estimate_world_size(H, corners_entry, OBSTACLE_W_M, OBSTACLE_L_M)

        # Rotation: pixel-space angle of the marker top edge
        rot_px = marker_rotation_deg(corners_entry)

        # Convert pixel rotation to world rotation
        # The homography can flip axes, so compute by mapping two points
        dx_px, dy_px = math.cos(math.radians(rot_px)), math.sin(math.radians(rot_px))
        wx2, wy2 = pixel_to_world(H, cx_px + dx_px * 20, cy_px + dy_px * 20)
        rot_world_deg = math.degrees(math.atan2(wy2 - wy, wx2 - wx))

        print(f"Obstacle {obs_id}: "
              f"world=({wx:+.4f},{wy:+.4f})m  "
              f"size={w_m*100:.1f}x{l_m*100:.1f}cm  "
              f"rot={rot_world_deg:.1f}deg")

        # CSV row: cx,cy,cz,width,length,height,rot_x,rot_y,rot_z_deg
        obstacle_rows.append([
            round(wx, 6),
            round(wy, 6),
            0.0,
            round(w_m, 6),
            round(l_m, 6),
            0.0,
            0.0,
            0.0,
            round(rot_world_deg, 6),
        ])

        # Draw on preview
        # Estimate pixel size for drawing box
        scale_px_per_m = 1.0 / max(abs(H[0][0]), 1e-6)
        w_px = w_m / max(abs(H[0][0]), abs(H[1][1])) * (w_img / 3.0)
        l_px = l_m / max(abs(H[0][0]), abs(H[1][1])) * (w_img / 3.0)
        draw_obstacle(preview, obs_id,
                      int(cx_px), int(cy_px),
                      w_px, l_px, rot_px,
                      (wx, wy))

    # -- Write boundingboxes.csv 
    csv_path = os.path.join(out_dir, CSV_FILE)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in obstacle_rows:
            writer.writerow(row)

    print(f"Wrote {len(obstacle_rows)} obstacle(s) to {csv_path!r}")

    # -- Process car marker 
    car_detected = False
    if CAR_ID in detected:
        corners_entry = detected[CAR_ID]
        cx_px, cy_px  = marker_centre_px(corners_entry)
        wx, wy        = pixel_to_world(H, cx_px, cy_px)

        # Car heading in world coordinates
        rot_px = marker_rotation_deg(corners_entry)
        dx_px  = math.cos(math.radians(rot_px))
        dy_px  = math.sin(math.radians(rot_px))
        wx2, wy2 = pixel_to_world(H, cx_px + dx_px * 20, cy_px + dy_px * 20)
        heading_deg = math.degrees(math.atan2(wy2 - wy, wx2 - wx))

        # Clamp position to arena
        wx = max(-1.5, min(1.5, wx))
        wy = max(-0.6, min(0.6, wy))

        print(f"Car:      "
              f"world=({wx:+.4f},{wy:+.4f})m  "
              f"heading={heading_deg:.1f}deg")

        pose_path = os.path.join(out_dir, POSE_FILE)
        with open(pose_path, "w") as f:
            f.write(f"x={wx:.6f}\n")
            f.write(f"y={wy:.6f}\n")
            f.write(f"heading_deg={heading_deg:.4f}\n")
        print(f"Car pose written to {pose_path!r}")

        draw_car(preview, int(cx_px), int(cy_px), rot_px, (wx, wy))
        car_detected = True
    else:
        print(f"Car marker ID {CAR_ID} not detected -- "
              f"robot_pose.txt not written")

    # -- Save preview 
    # Status banner at bottom
    obs_found = len(obstacle_rows)
    status = (f"Obstacles: {obs_found}/{len(OBSTACLE_IDS)}  |  "
              f"Car: {'YES' if car_detected else 'NO'}  |  "
              f"{os.path.basename(image_path)}")
    cv2.rectangle(preview, (0, h_img - 44), (w_img, h_img), (0, 0, 0), -1)
    cv2.putText(preview, status, (10, h_img - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 255, 0) if (obs_found == len(OBSTACLE_IDS) and car_detected)
                else (0, 165, 255), 2)

    preview_path = os.path.join(out_dir, PREVIEW_FILE)
    cv2.imwrite(preview_path, preview)
    print(f"Preview saved to {preview_path!r}")

    # -- Summary 
    print()
    print("Summary " + "-" * 44)
    print(f"  boundingboxes.csv : {obs_found} obstacle(s)")
    if car_detected:
        print(f"  robot_pose.txt    : car at ({wx:+.3f}, {wy:+.3f}), "
              f"heading {heading_deg:.1f} deg")
    print(f"  detection_preview.jpg : open to visually confirm")
    print()
    print("Next steps:")
    print("  1. Open detection_preview.jpg and confirm boxes/car look correct")
    print("  2. In the A* navigator press L and load boundingboxes.csv")
    print("  3. Press F to fit boxes to map if needed")
    print("  4. Press T and enter the car position from robot_pose.txt")
    print("  5. Queue goals and press SPACE")
    print()
    input("Press Enter to exit.")


if __name__ == "__main__":
    main()
