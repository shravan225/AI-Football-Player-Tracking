# AI-Football-Player-Tracking

AI-powered football player tracking system using YOLOv8, ByteTrack, OpenCV, and Dynamic ROI filtering.  
The project detects and tracks football players from tactical camera footage, generates heatmaps, player statistics, and annotated match analysis videos.

---

# Football Multi-Object Detection & Persistent ID Tracking — v2 (Production)

Assignment submission — YOLOv8l + ByteTrack + Dynamic Optical-Flow ROI

Video Source: Napoli vs AS Roma — Tactical Cam

---

## Features

- YOLOv8-based player detection
- ByteTrack multi-object tracking
- Persistent player IDs
- Dynamic Optical Flow ROI filtering
- Ball detection
- Tactical camera support
- Heatmap generation
- Player statistics export
- Kalman-based trajectory smoothing
- Multi-player real-time annotation

---

## Results

### Real-Time Player Tracking

The system successfully detects and tracks football players using YOLOv8 and ByteTrack.  
Each player receives a persistent tracking ID while ROI filtering removes audience and non-field detections.

![Player Tracking](Screenshot%202026-05-14%20170851.png)

---

### Dynamic ROI Filtering & Tactical Analysis

The green polygon represents the dynamically updated football field ROI generated using optical flow.  
Only players inside the field are tracked, improving tracking quality and reducing false positives.

![ROI Filtering](Screenshot%202026-05-14%20170816.png)

---

## Problems Fixed in v2

| Problem | Root Cause | Solution |
|---|---|---|
| ID explosion (100+ IDs) | Naive IoU tracking | ByteTrack + track_buffer=60 |
| Missing distant players | Low-resolution inference | YOLOv8l + imgsz=1280 |
| ID switching during overlap | Pure IoU association | ByteTrack two-stage matching |
| ROI drift during camera pan | Static polygon | Optical-flow polygon warping |
| Noisy tracking trails | Kalman-only predictions | TrailManager detection filtering |
| Scattered configuration | Hardcoded values | Unified DEFAULT_CONFIG |

---

## File Structure

```bash
AI-Football-Player-Tracking/
├── track_football.py
├── field_roi.py
├── define_roi.py
├── requirements.txt
├── README.md
├── output_heatmap.png
├── output_stats.json
```

---

## Installation

```bash
python -m venv venv
```

### Windows
```bash
venv\Scripts\activate
```

### Linux / Mac
```bash
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

YOLO weights download automatically during first execution.

---

## Quick Start

### Step 1 — Draw ROI Polygon

```bash
python define_roi.py --input video.mp4 --output field_roi.json
```

---

### Step 2 — Run Tracker

```bash
python track_football.py --input video.mp4 --output output.mp4 --roi field_roi.json --conf 0.25
```

---

## Configuration Presets

| Goal | Command |
|---|---|
| Best quality | `--model yolov8l.pt --imgsz 1280` |
| Tiny distant players | `--tiled` |
| CPU fast mode | `--model yolov8n.pt --imgsz 640 --skip 2` |
| Balanced CPU | `--model yolov8m.pt --imgsz 960` |
| Maximum GPU quality | `--model yolov8x.pt --imgsz 1280 --tiled` |

---

## ROI Modes

| Mode | Description |
|---|---|
| optical_flow | Dynamically warps ROI using LK optical flow |
| segmentation | Grass-mask based ROI estimation |
| static | Fixed polygon ROI |

Example:

```bash
python track_football.py --input video.mp4 --roi field_roi.json --roi-mode optical_flow
```

---

## Architecture

```text
Input Video
    │
    ▼
Dynamic ROI Update
    │
    ▼
YOLOv8 Detection
    │
    ▼
ROI Filtering
    │
    ▼
ByteTrack Association
    │
    ▼
Trail Manager
    │
    ▼
Annotated Output + Heatmap + Stats
```

---

## Output Files

| File | Description |
|---|---|
| output_tracked.mp4 | Annotated tracking video |
| output_heatmap.png | Player movement heatmap |
| output_stats.json | Player statistics |

---

## Dependencies

| Package | Purpose |
|---|---|
| ultralytics | YOLOv8 |
| supervision | ByteTrack |
| opencv-python | Video processing |
| numpy | Numerical operations |

---

## Tech Stack

- Python
- YOLOv8
- ByteTrack
- OpenCV
- NumPy
- Supervision

---

## Future Improvements

- StrongSORT / BoT-SORT re-identification
- Jersey color clustering
- Bird-eye homography projection
- Football-specific fine-tuning
- HOTA / IDF1 evaluation metrics

---

## Author

Samala Shravan Kumar

GitHub:
https://github.com/shravan225/AI-Football-Player-Tracking
