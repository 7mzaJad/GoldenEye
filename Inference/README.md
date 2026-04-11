# GoldenEye — Real-Time Search and Rescue Human Detection Interface

This project is a real-time Search and Rescue (SAR) human detection system designed to help locate lost people in desert environments using ONNX-based object detection models, with emphasis on detecting small and distant human targets. It provides a professional desktop application interface for live camera feeds, live screen capture, and uploaded image prediction.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PySide6](https://img.shields.io/badge/PySide6-6.6+-green)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.17+-orange)

---

## Features

- **Live Camera Detection** — Real-time inference on webcam feed
- **Live Screen Capture** — Real-time detection directly off connected displays
- **Image Upload Integration** — Choose between hardware camera stream or uploading a single `.jpg`/`.png`
- **Dynamic Source UI** — Instantly scan and toggle between connected webcams, iPhones (Continuity), and local displays
- **Dual Model Support** — Switch between available tracking models at runtime
- **Adjustable Thresholds** — Confidence and IoU sliders
- **Performance Metrics** — Inference latency, FPS, and human presence summation
- **Professional Dark Theme** — Gold-accented UI suitable for demo presentations

---

## Quick Start

### 1. Install Dependencies

```bash
cd Inference
pip install -r requirements.txt
```

> **Note:** On macOS with Apple Silicon, `onnxruntime` will automatically
> use the CoreML execution provider for hardware acceleration.

### 2. Verify Model Files

Ensure your ONNX model files are in place:

```
models/
├── Yolov8n/
│   └── best.onnx    ← YOLOv8 Nano model
└── Yolov26/
    └── best.onnx    ← YOLOv26 model
```

The paths are configured in `Inference/config.yaml` relative to the
`Inference/` directory.

### 3. Run the Application

```bash
cd Inference
python main.py
```

---

## Configuration

All settings are in **`config.yaml`**:

```yaml
models:
  yolov8n:
    path: "../models/Yolov8n/best.onnx"   # Relative to Inference/
    type: "yolov8"                         # Postprocessing pipeline
  yolov26:
    path: "../models/Yolov26/best.onnx"
    type: "yolov26"                        # NMS-free pipeline

class_names:
  - "human"                                # Add more classes here

defaults:
  confidence_threshold: 0.25
  iou_threshold: 0.45
  input_size: 640
  camera_index: 0                          # Change if using external camera
  default_model: "yolov8n"
```

### Adding New Models

1. Place the `.onnx` file in the `models/` directory
2. Add an entry under `models:` in `config.yaml`
3. Set `type` to `"yolov8"` (needs NMS) or `"yolov26"` (NMS-free)

### Adding New Classes

Update the `class_names` list in `config.yaml` to match your model's
class order:

```yaml
class_names:
  - "human"
  - "vehicle"
  - "debris"
```

---

## Project Structure

```
Inference/
├── main.py                 # Application entry point
├── config.yaml             # Model paths, classes, thresholds
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── core/
│   ├── engine.py           # ONNX Runtime session manager
│   ├── preprocessing.py    # Letterbox resize, normalize, NCHW
│   └── postprocessing.py   # YOLOv8 + YOLOv26 decoders, NMS
├── handlers/
│   ├── camera.py           # Camera capture thread (QThread)
│   └── image.py            # Image upload + prediction
├── ui/
│   ├── main_window.py      # Main window orchestration
│   ├── camera_panel.py     # Live camera preview
│   ├── upload_panel.py     # Image upload section
│   ├── controls_panel.py   # Model selector, sliders, status
│   ├── results_panel.py    # Detection table + stats
│   └── styles.py           # Dark theme stylesheet
└── utils/
    ├── constants.py         # Colors, fonts, app metadata
    ├── config_loader.py     # YAML loader + validation
    └── drawing.py           # Bounding box rendering
```

---

## Model Output Formats

The postprocessor auto-selects decoding logic based on `type` in config:

### YOLOv8 (`type: "yolov8"`)
- **Output shape:** `[1, 4+num_classes, 8400]`
- **Format:** `[cx, cy, w, h, class_scores...]` per anchor
- **Requires:** Confidence filtering + Non-Maximum Suppression

### YOLOv26 (`type: "yolov26"`)
- **Output shape:** `[1, 300, 6]`
- **Format:** `[x1, y1, x2, y2, confidence, class_id]` per detection
- **NMS-free:** Post-processed by the model itself

### Custom Models

If your ONNX model has a different output format, edit
`core/postprocessing.py`. The decoding logic is isolated in:
- `_postprocess_yolov8()` — for standard YOLO outputs
- `_postprocess_yolov26()` — for NMS-free outputs

---

## Adjusting Thresholds

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Confidence | 5–95% | 25% | Higher = fewer but more certain detections |
| IoU (NMS) | 10–95% | 45% | Higher = more overlapping boxes allowed |

Both can be adjusted in real-time via the UI sliders.

---

## Performance Notes

- **Threading:** Camera capture and inference run in separate threads
- **Frame skipping:** If inference can't keep up, frames are dropped
- **Warm-up:** Two dummy inferences run at startup to pre-allocate buffers
- **Graph optimization:** `ORT_ENABLE_ALL` applied to ONNX session
- **Camera buffer:** Set to 1 frame to reduce latency

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot access camera" | Check camera permissions in System Settings → Privacy |
| "Model file not found" | Verify paths in `config.yaml` relative to `Inference/` |
| Slow inference | Ensure `onnxruntime` (not `onnxruntime-gpu`) is installed on CPU-only machines |
| Window too small | Resize the window; minimum is 1100×700 |
| Black camera feed | Try changing `camera_index` in config to `1` or `2` |

---

## Requirements

- Python 3.10+
- macOS / Windows / Linux
- Webcam (for live detection)
- ~500 MB RAM for model + application
