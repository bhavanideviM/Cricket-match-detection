
import cv2
import numpy as np
from collections import defaultdict
from IPython.display import Video
from deep_sort_realtime.deepsort_tracker import DeepSort

import torch
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

from google.colab import files
uploaded = files.upload()   # upload cricket video
# VIDEO_PATH = "cricket.mp4"
print("Uploaded files:", uploaded.keys())

from IPython.display import Video

Video("cricket2.mp4", embed=True)

VIDEO_PATH = "cricket2.mp4"

from ultralytics import YOLO

model = YOLO("yolov8l.pt")

PERSON_CLASS_ID = 0



tracker = DeepSort(
    max_age=60,
    n_init=2,
    max_iou_distance=0.2,
    embedder="mobilenet",
    half=True
)


cap = cv2.VideoCapture(VIDEO_PATH)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "output_tracking.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

track_history = defaultdict(list)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame, conf=0.15, classes=[PERSON_CLASS_ID], imgsz=1280)[0]

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        detections.append(([x1, y1, x2-x1, y2-y1], conf, "player"))

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        track_history[track_id].append((cx, cy))

        # Bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

        # Draw trajectory
        pts = track_history[track_id]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (255,0,0), 2)

    out.write(frame)
    

cap.release()
out.release()

cv2.putText(frame,
            f"Detections: {len(results.boxes)}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2)


Video("output_tracking.mp4", embed=True)

files.download("output_tracking.mp4")

