# Football Player Tracking System (Optimized)

This project provides a clean, optimized football player tracking system that:
- Assigns unique IDs to each player
- Maintains player identity even if a player leaves and re-enters the frame (simple re-identification)
- Is easy to use and efficient

## Features
- **YOLO-based detection** (using your `best.pt` model)
- **ByteTrack** for robust multi-object tracking
- **Simple re-identification** using color histogram and position
- **Consistent player IDs** across frame exits and re-entries
- **Easy demo script** for running on your own videos

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your YOLO model file (`best.pt`) in the project root.
3. Place your input video in the `input_files/` directory.

## Usage

### Run the Demo

The main script is `demo_optimized_tracking.py`. By default, it will:
- Read `input_files/15sec_input_720p.mp4`
- Run tracking
- Save the output to `output_videos/optimized_output.avi`

Run:
```bash
python demo_optimized_tracking.py
```

You can change the input/output paths at the top of the script as needed.

### How it Works
- The script loads all frames from the input video.
- It initializes the `OptimizedPlayerTracker` from `tracker/optimized_tracker.py`.
- It processes all frames, assigning and maintaining player IDs.
- It draws bounding boxes and IDs on each frame.
- It saves the annotated video to the output path.

## File Structure

```
football-detection/
├── best.pt                      # YOLO model file
├── demo_optimized_tracking.py   # Main demo script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── tracker/
│   ├── __init__.py
│   └── optimized_tracker.py     # Optimized tracker implementation
├── input_files/                 # Input videos
└── output_videos/               # Output videos
```


## Notes
- Only the files listed above are needed. All other trackers, scripts, and utilities have been removed for clarity.
- The tracker is designed for balance: not overly fast or complex, but robust and easy to use.

## License
MIT 