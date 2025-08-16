# YOLOv8 Object Tracking with OpenCV  

This project demonstrates **real-time object detection and tracking** using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and **OpenCV**.  
The system supports live tracking from a webcam or video file, and displays bounding boxes, object IDs, and confidence scores.  

---

## 🚀 Features
- 🎯 **Real-time Object Detection** with YOLOv8  
- 🧾 **Object Tracking** using ByteTrack / BoT-SORT  
- 🆔 **Persistent IDs** across frames  
- 🔍 **Configurable Confidence Threshold & Input Size**  
- 🖼️ **Bounding Boxes + IDs + Confidence Scores** overlay  
- 👤 **Filter by classes** (default: person only)  
- ⌨️ **Safe exit** by pressing **Q** (works in both OpenCV window & Windows console)  

---

## 🛠️ Requirements
Make sure you have the following installed:  

- Python **3.8+**
- [Ultralytics YOLOv8](https://docs.ultralytics.com)  
- OpenCV  
- NumPy  

Install dependencies:
```bash
pip install ultralytics opencv-python numpy

Project Structure:
├── tracker.py          # Main script (tracking live from webcam)
├── requirements.txt    # Dependencies (optional)
├── README.md           # Project documentation


You can adjust the following parameters in the script:

WEIGHTS = "yolov8s.pt"       # Model weights (yolov8n.pt for faster inference)
CONF_THRES = 0.5             # Confidence threshold
IMG_SIZE = 640               # Input image size
TRACKER_CFG = "bytetrack.yaml"  # or "botsort.yaml"
FILTER_CLASSES = [0]         # 0 = person only (None for all classes)


Usage
1. Run live tracking from webcam
python tracker.py

2. Run tracking on a video file
python tracker.py

