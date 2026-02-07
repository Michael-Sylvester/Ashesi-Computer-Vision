import cv2
import numpy as np
import os
import sys

def main():
    # Load homography
    if not os.path.exists('homography_matrix.npy'):
        print("Error: homography_matrix.npy not found")
        return
    
    try:
        H = np.load('homography_matrix.npy')
    except Exception as e:
        print(f"Error loading homography matrix: {e}")
        return

    video_path = 'videos/test video.mp4'
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Output setup
    bev_dir = 'bev_frames'
    os.makedirs(bev_dir, exist_ok=True)
    out_path = 'bev_side_by_side.mp4'
    
    # Combined width will be 2 * original width
    out_width = width * 2
    out_height = height
    
    # Video Writer
    # Try mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (out_width, out_height))
    
    if not out.isOpened():
        print("Error: Could not create video writer. Trying MJPG...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_path = 'bev_side_by_side.avi'
        out = cv2.VideoWriter(out_path, fourcc, fps, (out_width, out_height))
        if not out.isOpened():
            print("Error: Could not create video writer with MJPG either.")
            return

    print(f"Processing {total_frames} frames. Saving output to {out_path} and frames to {bev_dir}/...")
    
    # Large canvas size for BEV (4000x6000) to match the offset and scale
    bev_dsize = (1700, 1000)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Warp perspective to generate BEV
        # Use bilinear interpolation for smoother results
        bev = cv2.warpPerspective(frame, H, bev_dsize, flags=cv2.INTER_LINEAR)
        
        # Save individual BEV frame (FULL RESOLUTION)
        frame_filename = os.path.join(bev_dir, f'frame_{frame_idx:05d}.png')
        cv2.imwrite(frame_filename, bev)
        
        # Resize BEV to match video height for side-by-side
        bev_small = cv2.resize(bev, (width, height))
        
        # Create side-by-side view
        combined = cv2.hconcat([frame, bev_small])
        
        # Write to video
        out.write(combined)

        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames", end='\r')

    cap.release()
    out.release()
    print(f"\nDone! Processed {frame_idx} frames.")
    print(f"Saved side-by-side video to {out_path}")
    print(f"Saved BEV frames to {bev_dir}/")

if __name__ == '__main__':
    main()
