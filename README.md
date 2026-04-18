# GoldenEye

GoldenEye is a human-detection project with three parallel developer workflows in one repository:

1. Web app stack: FastAPI backend + React frontend for live status and inference.
2. Desktop app stack: PySide6 + ONNX Runtime interface in the Inference folder.
3. Training and evaluation stack: Ultralytics YOLO pipeline under src.

This README is the entry point for contributors. Additional folder-specific docs are available in:

- frontend/README.md
- src/README.md
- Inference/README.md
- Dataset/README.md
- models/README.md
- notebooks/README.md

## Repository Layout

```text
GoldenEye/
├── README.md
├── requirements.txt            # Root Python dependencies (web + training)
├── main.py                     # Root training + inference pipeline runner
├── src/                        # Root ML modules + FastAPI server
├── frontend/                   # React + Vite web UI
├── Inference/                  # PySide6 desktop application
├── Dataset/                    # Dataset metadata and local split directories
├── models/                     # YOLO model artifacts (.pt / .onnx)
└── notebooks/                  # Exploration notebooks
```

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- Webcam/screen access (optional, for camera/screen inference)

## Quick Start

### 1) Create and activate virtual environment

Windows Git Bash:

```bash
cd /c/Users/Omar/Desktop/GoldenEye
python -m venv .venv
source ./.venv/Scripts/activate
```

PowerShell:

```powershell
cd C:\Users\Omar\Desktop\GoldenEye
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install root Python dependencies

```bash
pip install -r requirements.txt
```

### 3) Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## Run the Web Stack (recommended for most contributors)

Start backend:

```bash
cd /c/Users/Omar/Desktop/GoldenEye
./.venv/Scripts/python -m uvicorn api_server:app --app-dir src --host 127.0.0.1 --port 8000
```

Start frontend (new terminal):

```bash
cd /c/Users/Omar/Desktop/GoldenEye/frontend
npm run dev
```

Open:

- Frontend: http://localhost:5173
- Backend docs: http://127.0.0.1:8000/docs

## Run the Desktop App

The desktop app has separate dependencies:

```bash
cd /c/Users/Omar/Desktop/GoldenEye/Inference
pip install -r requirements.txt
python main.py
```

See Inference/README.md for model configuration and desktop-specific controls.

## Run the Training Pipeline

```bash
cd /c/Users/Omar/Desktop/GoldenEye
./.venv/Scripts/python main.py
```

This will:

1. Validate Dataset/data.yaml and split folders.
2. Train a YOLOv8 model using src/train.py.
3. Run test-set inference if Dataset/test/images exists.

## Core Web API Endpoints

- GET /api/health
- GET /api/model/status
- POST /api/model/select
- GET /api/sources
- GET /api/inference/latest
- POST /api/inference/image
- POST /api/inference/camera
- POST /api/inference/screen

Detailed endpoint behavior is documented in src/README.md.

## Data and Models

- Dataset metadata: Dataset/data.yaml
- Default root model path: models/Yolov8n/best.pt
- Desktop ONNX model paths: configured in Inference/config.yaml

Folder-level setup instructions are in Dataset/README.md and models/README.md.

## Common Issues

1. Frontend shows proxy errors (ECONNREFUSED 127.0.0.1:8000): backend is not running.
2. Uvicorn cannot bind to port 8000: another process is already using the port.
3. Camera index warnings on Windows: expected when probing unavailable camera indices.
4. Screen capture unavailable: ensure mss is installed in the active Python environment.

## Developer Notes

- Keep training, API, and desktop dependencies in sync when changing inference logic.
- Update folder READMEs when introducing new endpoints, scripts, or setup steps.
- Do not commit local dataset files under Dataset/train, Dataset/valid, Dataset/test unless explicitly needed.
