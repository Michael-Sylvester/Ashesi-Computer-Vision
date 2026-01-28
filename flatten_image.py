import numpy as np
import cv2
import glob

# 1. SETUP: Change this to (width-1, height-1) of your checkerboard squares
CHECKERBOARD = (9, 6) 
# Define the termination criteria for sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Load the frames you just extracted
images = glob.glob('calibration_frames/*.jpg')

print(f"Processing {len(images)} images...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        
        # Refining corner position for better accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners to make sure it's working
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Calibration Check', img)
        cv2.waitKey(100) # Pause briefly to see the detection

cv2.destroyAllWindows()

# 2. CALIBRATION
print("Calculating calibration... this might take a second.")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 3. SAVE DATA
# We save 'mtx' (Camera Matrix) and 'dist' (Distortion Coefficients)
np.savez('calibration_data.npz', mtx=mtx, dist=dist)

print("\nCalibration Complete!")
print("Camera Matrix (Intrinsic Parameters):\n", mtx)
print("\nDistortion Coefficients:\n", dist)
print("\nData saved to 'calibration_data.npz'. You can now use this to undistort any video from this phone.")

# Calculate re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"Total error: {mean_error/len(objpoints)}")
