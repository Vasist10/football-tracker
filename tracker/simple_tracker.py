import cv2
from ultralytics import YOLO
import supervision as sv

class SimpleTracker:
    def __init__(self, model_path, frame_width=1920, frame_height=1080):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.frame_width = frame_width
        self.frame_height = frame_height

    def process_frames(self, frames):
        annotated_frames = []
        for frame in frames:
            detection_results = self.model.predict(frame, conf=0.2)[0]
            detections = sv.Detections.from_ultralytics(detection_results)
            tracked = self.tracker.update_with_detections(detections)
            for i, bbox in enumerate(tracked.xyxy):
                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0) if hasattr(tracked, 'class_id') and tracked.class_id[i] != 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'ID:{tracked.id[i]}'
                if hasattr(tracked, 'class_id'):
                    label += f' C:{tracked.class_id[i]}'
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            annotated_frames.append(frame.copy())
        return annotated_frames 