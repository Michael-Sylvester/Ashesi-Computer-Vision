import cv2
import numpy as np

# 1. Load Calibration
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# 2. Setup Video and Timestamp
video_path = 'road_video.MOV' # Make sure this matches your file name!
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Jump to 214 seconds
start_frame = int(221 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Error reading video.")
    exit()

# 3. Undistort
h, w = frame.shape[:2]
new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_mtx)

# 4. Point Picking Logic
display_frame = cv2.resize(undistorted_frame, (int(w*0.4), int(h*0.4)))
points = []

def get_birds_eye(img, src_points):
    """
    This function takes your 4 clicked points and stretches them 
    into a top-down rectangle.
    """
    # Define the output size in pixels
    # Let's assume you want a 500x800 pixel top-down view
    out_w, out_h = 500, 500
    
    # Target points: A perfect rectangle
    dst_points = np.float32([
        [0, 0],         # Top Left
        [out_w, 0],     # Top Right
        [out_w, out_h], # Bottom Right
        [0, out_h]      # Bottom Left
    ])
    
    # Calculate Matrix and Warp
    M = cv2.getPerspectiveTransform(np.float32(src_points), dst_points)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))
    return warped

def click_event(event, x, y, flags, params):
    global display_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        real_x, real_y = int(x / 0.4), int(y / 0.4)
        points.append([real_x, real_y])
        cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('1. Click 4 Corners (TL, TR, BR, BL)', display_frame)
        
        if len(points) == 4:
            print(f"Points captured: {points}")
            result = get_birds_eye(undistorted_frame, points)
            cv2.imshow('2. Bird\'s Eye View (Sprint 1 Result)', result)
            print("Look at the result. Are the road lines parallel? (Press any key to exit)")

cv2.imshow('1. Click 4 Corners (TL, TR, BR, BL)', display_frame)
cv2.setMouseCallback('1. Click 4 Corners (TL, TR, BR, BL)', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()