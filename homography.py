import cv2
import numpy as np

# Load your calibration data
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# Load a frame from your road video (replace with your video filename)
cap = cv2.VideoCapture('road_video.MOV') 
# Select a specific frame (e.g., frame 100)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read frame. Check video path or frame number.")
    exit()

# Undistort it first!
h, w = frame.shape[:2]
new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_mtx)

# Resize so it fits your screen
display_frame = cv2.resize(undistorted_frame, (int(w*0.4), int(h*0.4)))
points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale back to original size
        real_x, real_y = int(x / 0.4), int(y / 0.4)
        points.append([real_x, real_y])
        cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Click 4 Corners of your Reference (Clockwise)', display_frame)
        if len(points) == 4:
            print(f"Captured Points: {points}")

cv2.imshow('Click 4 Corners of your Reference (Clockwise)', display_frame)
cv2.setMouseCallback('Click 4 Corners of your Reference (Clockwise)', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()