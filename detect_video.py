import cv2
import numpy as np
import csv
import os
import sys

# --- Configuration ---
VIDEO_PATH = 'videos/test video.mp4'
HOMOGRAPHY_PATH = 'homography_matrix.npy'
OUTPUT_CSV = 'pothole_data.csv'
OUTPUT_FRAMES_DIR = 'pothole_frames'

BEV_WIDTH = 1700
BEV_HEIGHT = 1000

# Calibration: 10 pixels = 1 cm²
PIXELS_PER_CM2 = 10
MIN_AREA_CM2 = 1000


def detect_potholes(bev, valid_mask, flat_l):
    """Detect potholes in a BEV frame. Returns list of (contour, area_cm2, center)."""
    lab = cv2.cvtColor(bev, cv2.COLOR_BGR2LAB)
    _, a_chan, b_chan = cv2.split(lab)

    a_smooth = cv2.GaussianBlur(a_chan.astype(float), (5, 5), 0)
    b_smooth = cv2.GaussianBlur(b_chan.astype(float), (5, 5), 0)

    # Clay/brown color detection (LAB a > 137, b > 142)
    mask_clay = ((a_smooth > 137) & (b_smooth > 142)).astype(np.uint8) * 255
    mask_clay = cv2.bitwise_and(mask_clay, valid_mask)

    # Clean clay blobs then dilate to find nearby dark holes
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    clay_clean = cv2.morphologyEx(mask_clay, cv2.MORPH_OPEN, kernel_open)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    clay_neighborhood = cv2.dilate(clay_clean, kernel_dilate)

    # Dark depression detection near clay regions
    mask_dark = ((flat_l < 75) & (clay_neighborhood > 0)).astype(np.uint8) * 255
    mask_dark = cv2.bitwise_and(mask_dark, valid_mask)

    # Combine clay + dark
    mask_combined = cv2.bitwise_or(mask_clay, mask_dark)

    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_open)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)

    # Find contours
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        area_cm2 = area_px / PIXELS_PER_CM2

        if area_cm2 < MIN_AREA_CM2:
            continue

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

        detections.append((cnt, area_cm2, (cx, cy)))

    return detections


def draw_overlay(bev, detections, frame_id):
    """Draw green segmentation overlay and labels on BEV frame."""
    overlay = bev.copy()
    vis = bev.copy()

    for i, (cnt, area_cm2, (cx, cy)) in enumerate(detections, 1):
        # Green filled overlay
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), -1)
        # Green contour outline
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 3)

    # Blend overlay
    blended = cv2.addWeighted(bev, 0.6, overlay, 0.4, 0)

    # Draw contour outlines and labels on blended
    for i, (cnt, area_cm2, (cx, cy)) in enumerate(detections, 1):
        cv2.drawContours(blended, [cnt], -1, (0, 255, 0), 3)
        label = f"P{i}: {area_cm2:.0f} cm2"
        cv2.putText(blended, label, (cx - 80, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Frame ID label
    cv2.putText(blended, f"Frame {frame_id}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return blended


def main():
    # Load homography
    if not os.path.exists(HOMOGRAPHY_PATH):
        print(f"Error: Homography matrix not found at {HOMOGRAPHY_PATH}")
        return
    H = np.load(HOMOGRAPHY_PATH)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {VIDEO_PATH}")
    print(f"  Frames: {total_frames}, FPS: {fps:.1f}")
    print(f"  Duration: {total_frames / fps:.1f}s")

    # Create output directory
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

    # CSV rows
    csv_rows = []
    frames_with_detections = 0
    frame_id = 0

    print(f"\nProcessing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Warp to BEV
        bev = cv2.warpPerspective(frame, H, (BEV_WIDTH, BEV_HEIGHT))

        # Valid region mask
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        valid_mask = (gray > 15).astype(np.uint8) * 255
        valid_mask = cv2.erode(valid_mask, np.ones((7, 7), np.uint8))

        # Illumination normalization for dark-hole detection
        lab = cv2.cvtColor(bev, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(float)
        bg_est = cv2.GaussianBlur(l_chan, (101, 101), 0) + 1.0
        mean_brightness = np.mean(l_chan[valid_mask > 0]) if np.any(valid_mask > 0) else 128
        flat_l = np.clip((l_chan / bg_est) * mean_brightness, 0, 255).astype(np.uint8)

        # Detect potholes
        detections = detect_potholes(bev, valid_mask, flat_l)

        if detections:
            frames_with_detections += 1

            # Record to CSV
            for pid, (cnt, area_cm2, center) in enumerate(detections, 1):
                csv_rows.append({
                    'Frame_ID': frame_id,
                    'Pothole_ID': pid,
                    'Area_cm2': round(area_cm2, 1),
                    'Center_Coordinate': f"({center[0]}, {center[1]})"
                })

            # Save masked frame
            masked_frame = draw_overlay(bev, detections, frame_id)
            out_path = os.path.join(OUTPUT_FRAMES_DIR, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(out_path, masked_frame)

        # Progress
        if frame_id % 100 == 0 or frame_id == total_frames:
            pct = (frame_id / total_frames) * 100
            print(f"  Frame {frame_id}/{total_frames} ({pct:.0f}%) | "
                  f"Detections so far: {len(csv_rows)} in {frames_with_detections} frames")

    cap.release()

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Frame_ID', 'Pothole_ID', 'Area_cm2', 'Center_Coordinate'])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{'='*50}")
    print(f"COMPLETE")
    print(f"{'='*50}")
    print(f"Total frames processed: {frame_id}")
    print(f"Frames with potholes:   {frames_with_detections}")
    print(f"Total detections (rows): {len(csv_rows)}")
    print(f"CSV saved to:           {OUTPUT_CSV}")
    print(f"Masked frames saved to: {OUTPUT_FRAMES_DIR}/")


if __name__ == "__main__":
    main()
