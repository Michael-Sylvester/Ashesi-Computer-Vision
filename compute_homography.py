import cv2
import numpy as np
import os

def compute_homography():
    image_path =  'extrinisic_calibration/road_chessboard_1.png'
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    print(f"Processing {image_path}...")
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
        #  scale: 10 pixels = 1 cm
        PIXELS_PER_CM = 10
        SQUARE_SIZE_CM = 3.4 #
        
        square_size_px = SQUARE_SIZE_CM * PIXELS_PER_CM # 35 pixels
        
        # Offset to position the checkerboard in the BEV output
        # Using smaller offsets for a more practical output size
        offset_x = 700
        offset_y = 300                                                                                                                                          
        
        dst_points = []
        for i in range(chessboard_size[1]): # 6 rows
            for j in range(chessboard_size[0]): # 9 columns
                x = offset_x + j * square_size_px
                y = offset_y + i * square_size_px
                dst_points.append([x, y])
        
        dst_points = np.array(dst_points, dtype=np.float32)
        src_points = corners2.reshape(-1, 2)
        
        # Compute Homography
        H, status = cv2.findHomography(src_points, dst_points)
        
        if H is not None:
            print("Homography matrix computed.")
            np.save('homography_matrix.npy', H)
            
            # Verify by warping the frame
            # Use a larger canvas to see more of the road in BEV
            # Adjust these values based on how much area you want to capture
            canvas_width = 1700   # width in pixels (at 10px/cm = 1 meter)
            # canvas_width = int(offset_x + chessboard_size[0] * square_size_px + 500) #1006
            canvas_height = 1000  # height in pixels (at 10px/cm = 1 meter)
            # canvas_height = int(offset_y + chessboard_size[1] * square_size_px + 500) #904
            print(f'canvas height:{canvas_height}')
            print(f'canvas width:{canvas_width}')
            warped_image = cv2.warpPerspective(frame, H, (canvas_width, canvas_height))
            cv2.imwrite('warped_verification.png', warped_image)
            print("Saved homography_matrix.npy and warped_verification.png")
    else:
        print("Could not find chessboard corners in the image.")

if __name__ == "__main__":
    compute_homography()
