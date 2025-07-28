import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

class SimpleTracker:
    def __init__(self, model_path, frame_width=1920, frame_height=1080):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.prev_centers = {}  # for speed calculation
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Map YOLO class IDs to labels — adjust this according to your model
        self.class_labels = {
            0: 'Goalkeeper',
            1: 'Player',
            2: 'Ball'
        }

    def _calculate_speed(self, obj_key, center, fps=30):
        prev = self.prev_centers.get(obj_key)
        self.prev_centers[obj_key] = center
        if prev is None:
            return 0
        dx = center[0] - prev[0]
        dy = center[1] - prev[1]
        return np.sqrt(dx**2 + dy**2) * fps / 100  # adjust this scale if needed

    def process_frames(self, frames):
        annotated_frames = []

        for frame in frames:
            results = self.model.predict(frame, conf=0.2)[0]
            detections = sv.Detections.from_ultralytics(results)
            tracked = self.tracker.update_with_detections(detections)

            for i, bbox in enumerate(tracked.xyxy):
                x1, y1, x2, y2 = map(int, bbox)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Get class ID and map to label
                class_id = int(tracked.class_id[i]) if hasattr(tracked, 'class_id') else -1
                label_text = self.class_labels.get(class_id, 'Object')

                # Generate unique key per object (not using tracker_id)
                obj_key = (class_id, i)
                speed = self._calculate_speed(obj_key, (cx, cy))
                label_text += f' | {speed:.1f}px/s'

                # Choose color per class
                if label_text.startswith("Ball"):
                    color = (0, 255, 255)   # Yellow
                elif label_text.startswith("Goalkeeper"):
                    color = (255, 0, 0)     # Blue
                else:
                    color = (0, 255, 0)     # Green for players

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            annotated_frames.append(frame.copy())

        return annotated_frames
