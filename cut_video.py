import cv2
import numpy as np

# 1. LOAD CALIBRATION
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# 2. SETTINGS
video_path = 'road_video.MOV'
start_sec = 214
end_sec = 296
# Output size (10 pixels = 1 cm). 
# Example: 200cm wide x 400cm long section of road
out_w, out_h = 1800, 1200 

# 3. CALCULATE THE MATRIX (Using the points you captured earlier)
# REPLACE THESE NUMBERS with the [real_x, real_y] points your terminal printed
src_pts = np.float32([[450, 600], [1450, 600], [1800, 1000], [100, 1000]]) 
dst_pts = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 4. INITIALIZE VIDEO
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('road_rectified.mp4', fourcc, fps, (out_w, out_h))

print("Processing video... this may take a minute.")

# 5. THE LOOP
current_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)

while current_frame < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    
    # A. Undistort (Intrinsic)
    undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
    
    # B. Warp (Extrinsic / Homography)
    warped = cv2.warpPerspective(undistorted, M, (out_w, out_h))
    
    # Write the frame
    out.write(warped)
    
    current_frame += 1
    if current_frame % 100 == 0:
        print(f"Processed frame {current_frame}...")

cap.release()
out.release()
print("Done! Open 'road_rectified.mp4' to see your top-down road map.")