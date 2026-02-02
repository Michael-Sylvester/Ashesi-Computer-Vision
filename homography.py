import cv2
import numpy as np
import sys

# 1. LOAD CALIBRATION
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# 2. CONFIGURATION
video_path = 'road_video.MOV'
start_sec = 214
end_sec = 296

# Your captured 1x1m patch points (Pixel Coordinates)
src_pts = np.float32([[205, 560], [395, 548], [355, 573], [140, 586]])

# --- METRIC LANDSCAPE CANVAS (10px = 1cm) ---
# Total: 5.5m Wide x 3m High
out_w, out_h = 5500, 3000 

# Positioning: Patch starts 50cm from left, 50cm from top
x_offset = 500 
y_offset = 500

dst_pts = np.float32([
    [x_offset, y_offset],               # TL
    [x_offset + 1000, y_offset],        # TR
    [x_offset + 1000, y_offset + 1000], # BR
    [x_offset, y_offset + 1000]         # BL
])

M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 3. INITIALIZE VIDEO
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int((end_sec - start_sec) * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('road_rectified_FINAL.mp4', fourcc, fps, (out_w, out_h))

print(f"Processing {total_frames} frames into a 5.5m x 3m Metric Map...")

while cap.get(cv2.CAP_PROP_POS_FRAMES) < int(296 * fps):
    ret, frame = cap.read()
    if not ret: break
    
    # Intrinsic + Extrinsic Transform
    undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
    warped = cv2.warpPerspective(undistorted, M, (out_w, out_h))
    
    # 4. DRAW METRIC GRID (Visual Evidence)
    # Draw faint lines every 10cm, Bold lines every 1m
    for x in range(0, out_w, 100):
        thickness = 3 if x % 1000 == 0 else 1
        cv2.line(warped, (x, 0), (x, out_h), (200, 200, 200), thickness)
    for y in range(0, out_h, 100):
        thickness = 3 if y % 1000 == 0 else 1
        cv2.line(warped, (0, y), (out_w, y), (200, 200, 200), thickness)

    out.write(warped)
    
    # Resize preview for your monitor
    preview = cv2.resize(warped, (int(out_w * 0.15), int(out_h * 0.15)))
    cv2.imshow('Sprint 1: Rectified Metric Map', preview)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print("\nDone! Video saved as road_rectified_FINAL.mp4")