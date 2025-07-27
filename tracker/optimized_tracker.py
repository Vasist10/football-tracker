import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class OptimizedPlayerTracker:
    def __init__(self, model_path, frame_width=1920, frame_height=1080, reid_threshold=0.7):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.reid_threshold = reid_threshold
        self.next_id = 0
        self.active_players = {}  # id -> player info
        self.lost_players = {}    # id -> player info
        self.max_lost_frames = 60

    def extract_features(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(32)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return np.zeros(32)
        region = cv2.resize(region, (16, 32))
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 2], [0, 180, 0, 256, 0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        return hist

    def reidentify(self, features, center):
        best_id = None
        best_score = 0
        for pid, pdata in self.lost_players.items():
            app_sim = cosine_similarity([features], [pdata['features']])[0][0]
            dist = np.linalg.norm(center - pdata['center'])
            pos_sim = max(0, 1 - dist / 100)
            score = 0.7 * app_sim + 0.3 * pos_sim
            if score > best_score and score > self.reid_threshold:
                best_score = score
                best_id = pid
        return best_id

    def update_lost_players(self, frame_idx):
        to_remove = []
        for pid, pdata in self.lost_players.items():
            if frame_idx - pdata['last_frame'] > self.max_lost_frames:
                to_remove.append(pid)
        for pid in to_remove:
            del self.lost_players[pid]

    def process_frames(self, frames):
        annotated_frames = []
        for frame_idx, frame in enumerate(frames):
            detection_results = self.model.predict(frame, conf=0.2)[0]
            detections = sv.Detections.from_ultralytics(detection_results)
            tracked = self.tracker.update_with_detections(detections)
            frame_ids = set()
            for i, bbox in enumerate(tracked.xyxy):
                # Only track players (skip ball if class_id==0)
                if hasattr(tracked, 'class_id') and tracked.class_id[i] == 0:
                    continue
                center = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
                features = self.extract_features(frame, bbox)
                matched_id = None
                for pid, pdata in self.active_players.items():
                    dist = np.linalg.norm(center - pdata['center'])
                    if dist < 30:
                        matched_id = pid
                        break
                if matched_id is None:
                    matched_id = self.reidentify(features, center)
                if matched_id is None:
                    matched_id = self.next_id
                    self.next_id += 1
                self.active_players[matched_id] = {
                    'features': features,
                    'center': center,
                    'bbox': bbox,
                    'last_frame': frame_idx
                }
                frame_ids.add(matched_id)
                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'ID:{matched_id}'
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            to_remove = []
            for pid, pdata in self.active_players.items():
                if pid not in frame_ids:
                    self.lost_players[pid] = pdata
                    self.lost_players[pid]['last_frame'] = frame_idx
                    to_remove.append(pid)
            for pid in to_remove:
                del self.active_players[pid]
            self.update_lost_players(frame_idx)
            annotated_frames.append(frame.copy())
        return annotated_frames 