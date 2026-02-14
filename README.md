# ICS 555 - Computer Vision: Pothole Detection and Measurement

A computer vision pipeline for detecting and measuring potholes from road video footage using homography-based bird's eye view (BEV) transformation and LAB color space segmentation.

## Project Overview

This project processes video of road surfaces through three stages:

1. **Calibration** - Camera calibration using chessboard patterns to compute a homography matrix
2. **BEV Generation & Cleaning** - Transform video frames into a top-down bird's eye view and clean artifacts
3. **Pothole Detection & Measurement** - Detect potholes using color-based segmentation and measure their area in real-world units

## Pipeline

### Sprint 1 - Bird's Eye View Generation

#### 1. Extract chessboard frames for intrinsic calibration

Detect and save chessboard corners from a calibration video.

```bash
python save_detected_chessboard.py
```

#### 2. Find chessboard on road for extrinsic calibration

Locate the chessboard pattern placed on the road surface.

```bash
python find_homography.py
```

#### 3. Compute homography matrix

Compute the perspective transformation matrix from the detected chessboard correspondences. The output is saved to `homography_matrix.npy`.

```bash
python compute_homography.py
```

#### 4. Generate BEV video and frames

Apply the homography to the test video to produce bird's eye view frames. BEV canvas size: **1700 x 1000** pixels.

```bash
python generate_bev.py
```

### Sprint 2 - BEV Image Cleaning

Clean the BEV frames by removing warping artifacts and noise from the transformed images.

```bash
python clean_bev.py
```

### Sprint 3 - Pothole Detection and Measurement

#### Detection Methodology

The detection algorithm uses a two-cue approach in LAB color space:

1. **Clay/brown color detection** - Potholes expose clay-colored sub-surface material. Detected via LAB thresholds: `a > 137` (reddish) and `b > 142` (yellowish), which separates pothole regions from the surrounding road surface (typically `a ~ 133-135`, `b ~ 135-138`).

2. **Dark depression detection** - The depressed area within potholes appears darker than the surrounding surface. After dilating detected clay regions, dark pixels (normalized `L < 75`) within that neighborhood are identified as part of the pothole.

3. **Morphological cleanup** - Opening (9x9 ellipse) removes noise, closing (31x31 ellipse) merges the clay and dark sub-regions into cohesive contours. Contours below **400 cm²** are filtered out.

#### Calibration

- **10 pixels = 1 cm²** in the BEV space
- Area is computed from contour pixel area: `area_cm2 = area_pixels / 10`

#### Run on the full video

Processes all frames in the video, detects potholes in each BEV frame, and generates:

- `pothole_data.csv` - CSV with columns: `Frame_ID`, `Pothole_ID`, `Area_cm2`, `Center_Coordinate`
- `pothole_frames/` - Saved BEV frames with green segmentation mask overlay for frames containing detections

```bash
python detect_video.py
```

## Project Structure

```
.
├── save_detected_chessboard.py   # Extract chessboard frames from calibration video
├── find_homography.py            # Detect chessboard on road
├── compute_homography.py         # Compute homography matrix
├── generate_bev.py               # Generate BEV video/frames
├── clean_bev.py                  # Clean BEV artifacts
├── detect_video.py               # Full video pothole detection pipeline  
├── homography_matrix.npy         # Computed homography matrix
├── pothole_data.csv              # Detection results (output)
├── pothole_frames/               # Masked detection frames (output)
├── bev_frames/                   # BEV-transformed frames
├── cleaned_bev_frames/           # Cleaned BEV frames
└── videos/
    ├── test video.mp4            # Test video for pothole detection
    ├── homography1.mp4           # Homography calibration video 1
    ├── homography2.mp4           # Homography calibration video 2
    └── IMG_7043.MOV              # Source road footage
```

## Requirements

- Python 3.10+
- OpenCV (`cv2`)
- NumPy

## CSV Output Format

| Column | Description |
|--------|-------------|
| `Frame_ID` | Sequential frame number in the video |
| `Pothole_ID` | Pothole index within the frame (1-indexed) |
| `Area_cm2` | Estimated area in cm² |
| `Center_Coordinate` | Centroid position `(x, y)` in BEV pixel coordinates |

## Team Members

- Cajetan Songwae
- Nana Sam Yeboah
- Innocent Farai Chikwanda
- Michael Kwabena Sylvester
- Kelvin Mhodi
