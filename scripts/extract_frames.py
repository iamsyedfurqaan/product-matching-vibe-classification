import cv2
import os

# Updated paths according to Flickd submission structure
video_dir = 'videos'      # All input videos are here
output_dir = 'frames'     # All extracted frames will go here

# Create output_dir if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all video files in 'videos/'
for video_file in os.listdir(video_dir):
    if not video_file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        continue  # Skip non-video files

    video_path = os.path.join(video_dir, video_file)
    video_name = os.path.splitext(video_file)[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(video_output_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames from {video_file} to {video_output_dir}")

# This script assumes the current working directory is the root of the submission folder,
# and videos are in 'videos/' and frames will be saved in 'frames/'.
