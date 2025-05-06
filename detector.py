# detector.py
import cv2
from ultralytics import YOLO
from collections import defaultdict
import math

model = YOLO("visDrone.pt")

def estimate_speed(path, frame_rate):
    if len(path) < 2:
        return 0.0
    f1, p1 = path[0]
    f2, p2 = path[-1]
    time_elapsed = (f2 - f1) / frame_rate
    if time_elapsed == 0:
        return 0.0
    pixel_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return round(pixel_dist / time_elapsed, 2)

def process_video(video_path, stats):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    line_y = 200
    object_paths = defaultdict(list)
    counted_ids = set()
    global_detected_ids = set()
    class_names = set()
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=0.5, iou=0.5)
        annotated_frame = results[0].plot()
        cv2.line(annotated_frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cls = results[0].boxes.cls.cpu().numpy()
            for i, obj_id in enumerate(ids):
                x1, y1, x2, y2 = boxes[i]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                class_id = int(cls[i])
                class_names.add(model.names[class_id])

                global_detected_ids.add(obj_id)
                object_paths[int(obj_id)].append((frame_num, (cx, cy)))

                path = object_paths[int(obj_id)]
                if len(path) >= 2:
                    _, prev = path[-2]
                    _, curr = path[-1]
                    if prev[1] < line_y <= curr[1] and int(obj_id) not in counted_ids:
                        counted_ids.add(int(obj_id))

        # Update live stats
        stats['global_count'] = len(global_detected_ids)
        stats['region_count'] = len(counted_ids)
        stats['classes'] = list(class_names)

        # Yield frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_num += 1

    cap.release()
