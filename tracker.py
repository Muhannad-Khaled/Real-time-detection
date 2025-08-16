import cv2
from ultralytics import YOLO
import numpy as np
import sys

# --- Adjustable settings ---
WEIGHTS = "yolov8s.pt"        # Use yolov8s.pt for higher accuracy, or yolov8n.pt if your device is weak
CONF_THRES = 0.5              # Confidence threshold
IMG_SIZE = 640                # Input image size
TRACKER_CFG = "bytetrack.yaml"  # or "botsort.yaml"
FILTER_CLASSES = [2]          # 0 = person only. Use None for all classes
# -----------------------------

# Detect 'q' press in Windows console even if the OpenCV window is not focused
def want_quit():
    try:
        import msvcrt
        if msvcrt.kbhit():
            key = msvcrt.getch().decode(errors="ignore").lower()
            if key == 'q':
                return True
    except Exception:
        pass
    return False

model = YOLO(WEIGHTS)

def draw_tracks(frame, result):
    """Draw bounding boxes + IDs + confidence"""
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().int().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
    ids  = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(xyxy), -1)

    for (x1, y1, x2, y2), c, tid in zip(xyxy, conf, ids):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {tid}  {c:.2f}" if tid >= 0 else f"{c:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

# Example function if you want to run tracking on a video file
# def run_tracker_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     gen = model.track(
#         source=video_path,
#         stream=True,
#         imgsz=IMG_SIZE,
#         conf=CONF_THRES,
#         iou=0.45,
#         tracker=TRACKER_CFG,
#         persist=True,
#         classes=FILTER_CLASSES,
#         verbose=False,
#         show=False
#     )
#
#     for result in gen:
#         frame = result.orig_img
#         frame = draw_tracks(frame, result)
#         cv2.imshow("Tracking", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
#     cap.release()

def run_tracker_live(cam_index=0):
    # Open camera at higher resolution for better results
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Use Ultralytics streaming generator (stream=True)
    gen = model.track(
        source=cam_index,
        stream=True,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=0.45,
        tracker=TRACKER_CFG,
        persist=True,            # Keep IDs across frames
        classes=FILTER_CLASSES,  # Filter to person only
        verbose=False,
        show=False               # We handle drawing/display ourselves
    )

    for result in gen:
        frame = result.orig_img  # Original BGR frame
        frame = draw_tracks(frame, result)

        cv2.imshow("Tracking", frame)
        # Safe exit with 'q' from either OpenCV window or Windows console
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or want_quit():
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    run_tracker_live()
