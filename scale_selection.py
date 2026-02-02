import cv2
import numpy as np

# Load Calibration
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

cap = cv2.VideoCapture('road_video.MOV')
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(222 * fps)) # Jump to road part
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]
# Undistort without cropping
undistorted = cv2.undistort(frame, mtx, dist, None, mtx)

# Resize for display
display_scale = 0.6
display_img = cv2.resize(undistorted, (int(w*display_scale), int(h*display_scale)))
points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Map clicks back to full resolution
        real_x, real_y = int(x / display_scale), int(y / display_scale)
        points.append([real_x, real_y])
        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('SELECT 4 POINTS: TL, TR, BR, BL', display_img)
        if len(points) == 4:
            print("\nCOPY THESE POINTS INTO THE NEXT SCRIPT:")
            print(f"src_pts = np.float32({points})")

cv2.imshow('SELECT 4 POINTS: TL, TR, BR, BL', display_img)
cv2.setMouseCallback('SELECT 4 POINTS: TL, TR, BR, BL', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()