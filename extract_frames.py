import cv2
import os

# 1. Setup paths
video_path = 'checkerboard.MOV'
output_folder = 'calibration_frames'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
saved_count = 0
# Extract one frame every 30 frames (approx. 1 second)
interval = 30 

print("Starting extraction...")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break # End of video
    
    if frame_count % interval == 0:
        # Save frame as a high-quality JPG
        file_name = f"{output_folder}/frame_{saved_count:03d}.jpg"
        cv2.imwrite(file_name, frame)
        saved_count += 1
        print(f"Saved: {file_name}")
        
    frame_count += 1

cap.release()
print(f"Done! Extracted {saved_count} frames to '{output_folder}'.")