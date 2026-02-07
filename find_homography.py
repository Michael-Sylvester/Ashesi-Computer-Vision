import cv2
import numpy as np
import os

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def find_homography_chessboard(image_path='extrinisic_calibration/road_chessboard_0.png'):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    print(f"Processing {image_path} for chessboard...")
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not read image.")
        return

    chessboard_size = (9, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        print(f"Chessboard found in {image_path}!")
        # Refine corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw corners for visual verification of the source frame
        src_debug_frame = frame.copy()
        cv2.drawChessboardCorners(src_debug_frame, chessboard_size, corners2, ret)
        
        # Draw the order of points
        for idx, point in enumerate(corners2):
            x, y = point.ravel()
            cv2.putText(src_debug_frame, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        cv2.imwrite('homography_debug_order.png', src_debug_frame)
        cv2.imwrite('homography_source_frame.png', src_debug_frame)
        
        # Define destination points (top-down view)
        square_size = 50
        offset = 100 
        
        dst_points = []
        for i in range(chessboard_size[1]): # 6 rows
            for j in range(chessboard_size[0]): # 9 columns
                x = offset + j * square_size
                y = offset + i * square_size
                dst_points.append([x, y])
        
        dst_points = np.array(dst_points, dtype=np.float32)
        src_points = corners2.reshape(-1, 2)
        
        # Compute Homography
        H, status = cv2.findHomography(src_points, dst_points)
        
        if H is not None:
            print("Homography matrix computed (chessboard).")
            np.save('homography_matrix.npy', H)
            
            # Verify by warping the frame
            h, w = frame.shape[:2]
            warped_image = cv2.warpPerspective(frame, H, (w, h))
            cv2.imwrite('warped_verification.png', warped_image)
            print("Saved homography_matrix.npy and warped_verification.png")
    else:
        print("Could not find chessboard corners in the image.")

def find_homography_yellow_box(image_path='extrinisic_calibration/road_chessboard_59.png'):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    print(f"Processing {image_path} for yellow box...")
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not read image.")
        return

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for yellow color
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    cv2.imwrite('yellow_mask.png', mask) # Save mask for debugging

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No yellow contours found.")
        return

    # Filter contours by area to ignore small noise (leaves, etc.)
    min_area = 1000 # Adjust this threshold based on image resolution
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        print("No valid yellow contours found (too small).")
        return

    # Debug: Draw ALL valid contours to see what we are picking from
    all_contours_img = frame.copy()
    cv2.drawContours(all_contours_img, valid_contours, -1, (0, 255, 0), 2)
    cv2.imwrite('all_contours_debug.png', all_contours_img)
    print("Saved all_contours_debug.png")

    # Find the largest contour (assuming it's the box)
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If the approximated contour has 4 points, we assume it's our box
    if len(approx) == 4:
        print("Yellow box detected!")
        
        # Reshape points
        pts = approx.reshape(4, 2)
        ordered_pts = order_points(pts)
        
        # Draw the detected box and points for verification
        debug_frame = frame.copy()
        cv2.drawContours(debug_frame, [approx], -1, (0, 255, 0), 3)
        
        for i, pt in enumerate(ordered_pts):
            cv2.putText(debug_frame, str(i), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imwrite('detected_yellow_box.png', debug_frame)
        print("Saved detected_yellow_box.png for verification.")

        # Define destination points for a 1mx1m square
        square_size = 500
        dst_points = np.array([
            [0, 0],
            [square_size - 1, 0],
            [square_size - 1, square_size - 1],
            [0, square_size - 1]
        ], dtype="float32")

        # Compute Homography
        H, status = cv2.findHomography(ordered_pts, dst_points)
        
        if H is not None:
            print("Homography matrix computed (yellow box).")
            np.save('homography_matrix_box.npy', H)
            
            # Verify by warping the frame
            h, w = frame.shape[:2]
            warped_image = cv2.warpPerspective(frame, H, (square_size, square_size))
            cv2.imwrite('warped_box_verification.png', warped_image)
            print("Saved homography_matrix_box.npy and warped_box_verification.png")
    else:
        print(f"Could not detect a 4-sided polygon. Found {len(approx)} sides.")
        # Save debug image with contours
        debug_frame = frame.copy()
        cv2.drawContours(debug_frame, [largest_contour], -1, (0, 0, 255), 2)
        cv2.imwrite('failed_detection_debug.png', debug_frame)

if __name__ == "__main__":
    # Uncomment the method you want to use
    # find_homography_yellow_box()
    find_homography_chessboard()