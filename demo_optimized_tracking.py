import cv2
import os
from tracker.optimized_tracker import OptimizedPlayerTracker
from tracker.simple_tracker import SimpleTracker

# Change these paths as needed
INPUT_VIDEO = 'input_files/15sec_input_720p.mp4'
OUTPUT_VIDEO = 'output_videos/optimized_output.avi'
SIMPLE_OUTPUT_VIDEO = 'output_videos/simple_output.avi'
MODEL_PATH = 'best.pt'

# Load video
cap = cv2.VideoCapture(INPUT_VIDEO)
frames = []
print('Reading video frames...')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f'Total frames read: {len(frames)}')

if len(frames) == 0:
    print('No frames found in video! Exiting.')
    exit(1)

# Initialize tracker
frame_height, frame_width = frames[0].shape[:2]
tracker = OptimizedPlayerTracker(MODEL_PATH, frame_width, frame_height)
simple_tracker = SimpleTracker(MODEL_PATH, frame_width, frame_height)

# Run tracking
print('Running optimized tracking...')
annotated_frames = tracker.process_frames(frames)
print('Optimized tracking complete.')

# Save output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 25, (frame_width, frame_height))
for frame in annotated_frames:
    out.write(frame)
out.release()
print(f'Optimized output saved to {OUTPUT_VIDEO}')

# Run simple tracking
print('Running simple tracking...')
simple_annotated_frames = simple_tracker.process_frames(frames)
print('Simple tracking complete.')

# Save simple tracker output video
simple_out = cv2.VideoWriter(SIMPLE_OUTPUT_VIDEO, fourcc, 25, (frame_width, frame_height))
for frame in simple_annotated_frames:
    simple_out.write(frame)
simple_out.release()
print(f'Simple tracker output saved to {SIMPLE_OUTPUT_VIDEO}') 