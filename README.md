## Execution Order
To correctly process the data and reproduce the results, please run the scripts in the following order:

### 1. `extract_frames.py`
**Purpose**: Extracts individual frames from video files.

**Inputs**: 
- Video file specified in `video_path` variable (e.g., `checkerboard.MOV`)

**Outputs**: 
- Folder `calibration_frames/` containing extracted frames as JPG files (e.g., `frame_000.jpg`, `frame_001.jpg`, etc.)
- One frame is extracted every 30 frames (approximately 1 second intervals)

**Next Step**: Use the extracted frames as input to `flatten_image.py`

---

### 2. `flatten_image.py`
**Purpose**: Performs camera calibration using a checkerboard pattern to compute the camera matrix and distortion coefficients.

**Inputs**: 
- Folder `calibration_frames/` containing calibration images with a visible checkerboard pattern
- Checkerboard configuration: `CHECKERBOARD = (9, 6)` (width-1, height-1 of squares)

**Outputs**:
- `calibration_data.npz` - Contains the camera matrix (`mtx`) and distortion coefficients (`dist`)
- Console output showing the calibration parameters and re-projection error

**Next Step**: Use the generated `calibration_data.npz` as input to `scale_selection.py`

---

### 3. `scale_selection.py`
**Purpose**: Let the user pick a 1×1 m reference patch in a representative frame and produce the source points for perspective transform.

**Inputs**:
- `calibration_data.npz` - Camera matrix and distortion coefficients from `flatten_image.py`
- Video file specified in `road_video.MOV` (script seeks to a sample frame)

**Outputs**:
- Console output that prints a Python assignment like `src_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])` which contains the four corner pixel coordinates (TL, TR, BR, BL).

**Next Step**: Copy the printed `src_pts` into the `src_pts` variable in `homography.py` before running it.

---

### 4. `homography.py`
**Purpose**: Applies camera calibration and perspective transformation to rectify video frames into a metric coordinate system.

**Inputs**:
- `calibration_data.npz` - Camera matrix and distortion coefficients from `flatten_image.py`
- `src_pts` - Pixel coordinates produced by `scale_selection.py` (paste into `homography.py`)
- Video file specified in `video_path` variable (e.g., `road_video.MOV`)
- Perspective transformation parameters (output canvas size, positioning)

**Outputs**:
- `road_rectified_FINAL.mp4` - Rectified video with metric grid overlay showing the top-down view
- Grid dimensions: 5.5m × 3m with 10px = 1cm scale
