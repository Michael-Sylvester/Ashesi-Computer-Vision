import cv2
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor

def clean_frame(frame_path, output_dir):
    filename = os.path.basename(frame_path)
    output_path = os.path.join(output_dir, filename)
    
    img = cv2.imread(frame_path)
    if img is None:
        print(f"Error reading {frame_path}")
        return

    # 1. Denoising using Bilateral Filter
    # Keeps edges sharp while removing noise
    # d=9, sigmaColor=75, sigmaSpace=75 are standard good starting values
    denoised = cv2.bilateralFilter(img, 9, 75, 75)
    
    # 2. Illumination Normalization (Shadow Removal) and Flattening
    # Work in LAB color space to separate Lightness from Color
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Estimate background illumination using strong morphological closing
    # This removes dark spots (cracks, tarmac texture) from the background estimate
    # leaving only the lighting variation (shadows)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    bg_est = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, kernel)
    
    # Further blur to smooth out the background estimate
    bg_est = cv2.GaussianBlur(bg_est, (51, 51), 0)
    
    # Normalize: Result = (Original / Background) * Scale
    # We ideally want to keep the average brightness of the road, 
    # but remove the variations.
    # Avoid division by zero
    bg_est = bg_est.astype(float) + 1.0
    l_float = l_channel.astype(float)
    
    # "Flat field" correction
    # We scale by the mean of the background to restore typical brightness
    mean_brightness = np.mean(bg_est)
    flat_l = (l_float / bg_est) * mean_brightness
    
    # Clip and convert back
    flat_l = np.clip(flat_l, 0, 255).astype(np.uint8)
    
    # 3. Local Contrast Enhancement (Stabilizing texture appearance)
    # Using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(flat_l)
    
    # Merge back
    result_lab = cv2.merge((enhanced_l, a, b))
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    
    cv2.imwrite(output_path, result_bgr)
    # print(f"Saved {output_path}", end='\r')

def main():
    input_dir = 'bev_frames'
    output_dir = 'cleaned_bev_frames'
    
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    frames = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    print(f"Found {len(frames)} frames to process.")
    
    # Process in parallel for speed
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, frame_path in enumerate(frames):
            executor.submit(clean_frame, frame_path, output_dir)
            if i % 10 == 0:
                 print(f"Scheduled {i}/{len(frames)}", end='\r')
                 
    print(f"\nDone! Processed frames saved to {output_dir}/")

if __name__ == '__main__':
    main()
