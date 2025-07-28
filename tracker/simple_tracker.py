import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

class SimpleTracker:
    def __init__(self, model_path, frame_width=1920, frame_height=1080):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.prev_centers = {}  # For tracking object movement
        self.frame_width = frame_width
        self.frame_height = frame_height

    def _calculate_speed(self, obj_key, center, fps=30):
        prev = self.prev_centers.get(obj_key)
        self.prev_centers[obj_key] = center
        if prev is None:
            return 0
        dx = center[0] - prev[0]
        dy = center[1] - prev[1]
        return np.sqrt(dx**2 + dy**2) * fps / 100  

    def process_frames(self, frames):
        annotated_frames = []
        for frame in frames:
            results = self.model.predict(frame, conf=0.2)[0]
            detections = sv.Detections.from_ultralytics(results)
            tracked = self.tracker.update_with_detections(detections)

            for i in range(len(tracked.xyxy)):
                x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1

                if class_id == 0:
                    label = 'Ball'
                    color = (0, 255, 255)
                else:
                    label = 'Player'
                    color = (0, 255, 0)

                speed = self._calculate_speed((class_id, i), (cx, cy))
                label += f' | {speed:.1f}px/s'

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            annotated_frames.append(frame.copy())
        return annotated_frames
