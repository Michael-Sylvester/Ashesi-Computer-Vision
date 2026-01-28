import cv2
import numpy as np

# 1. Load your saved calibration data
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# 2. Load one of your extracted frames
img = cv2.imread('calibration_frames/frame_000.jpg')
if img is None:
    print("Error: Could not find frame_000.jpg in calibration_frames folder.")
    exit()

h, w = img.shape[:2]

# 3. Calculate the optimal new camera matrix 
# (This helps handle those black edges and keeps the image centered)
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 4. Undistort the image
dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

# 5. RESIZE for your monitor (Very important for iPhone high-res video)
scale_percent = 30  # Adjust this (e.g., 30 or 50) if the window is still too big/small
width = int(w * scale_percent / 100)
height = int(h * scale_percent / 100)
dim = (width, height)

img_resized = cv2.resize(img, dim)
dst_resized = cv2.resize(dst, dim)

# 6. Stack and Show
comparison = np.hstack((img_resized, dst_resized))
cv2.imshow('Left: Original | Right: Undistorted', comparison)

print("Check the window. Look at the edges of the checkerboard.")
print("The 'bent' lines on the left should be 'straight' on the right.")
print("Press any key in the image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()