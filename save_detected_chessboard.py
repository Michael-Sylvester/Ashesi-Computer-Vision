import cv2
import numpy as np
import os


video_path = 'videos/checkerboard video.mp4'
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(fps * 5)
chessboard_size = (9, 6)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points: (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane
gray_shape = None

from collections import deque

# ... (previous code)

saved_frames = deque(maxlen=5)

print("Processing frames...")
for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        # Refine corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw corners
        cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
        # Add frame to buffer (automatically handles maxlen)
        saved_frames.append(frame.copy())

# Save the collected frames
for idx, frame in enumerate(saved_frames):
    filename = f'detected_chessboard_{idx}.png'
    cv2.imwrite(filename, frame)
    print(f"Saved {filename}")

cap.release()

# Calibrate camera
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    print("Calibration successful")

    # Calculate re-projection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"Total mean error: {mean_error/len(objpoints)}")

else:
    print("Calibration failed: No corners found")
