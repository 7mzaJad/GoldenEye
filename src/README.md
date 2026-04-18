# GoldenEye src

This folder contains the root YOLO pipeline modules and the FastAPI backend used by the web UI.

## Modules

- `config.py`: shared paths, defaults, and model selection constants.
- `data_loader.py`: dataset YAML validation and split/sample inspection.
- `train.py`: Ultralytics training entrypoint.
- `evaluate.py`: validation metric helper.
- `inference.py`: inference wrapper with detailed detection parsing.
- `api_server.py`: FastAPI live API for web client integration.

## Run the API Server

From repository root:

```bash
./.venv/Scripts/python -m uvicorn api_server:app --app-dir src --host 127.0.0.1 --port 8000
```

Open docs at:

- http://127.0.0.1:8000/docs

## API Endpoints

### Health and status

- `GET /api/health`
- `GET /api/model/status`
- `POST /api/model/select`
- `GET /api/sources?max_camera_index=1`

### Inference

- `GET /api/inference/latest`
- `POST /api/inference/image`
- `POST /api/inference/camera`
- `POST /api/inference/screen`

## Request Notes

- `conf_threshold` and `iou_threshold` must be in `[0, 1]`.
- `model_path` can be provided to inference endpoints to switch runtime model.
- image inference expects multipart form file field: `file`.
- camera inference expects form field: `device_index`.
- screen inference expects form field: `screen_index`.

## Response Notes

Inference responses include:

- `inference`: structured detection payload (counts, confidences, timings, items).
- `preview_jpeg_base64`: annotated frame preview for direct frontend display.

`GET /api/inference/latest` returns cached latest values of both fields.

## Dependencies

Root dependencies are defined in `../requirements.txt`.

Important runtime dependencies for API paths:

- `fastapi`
- `uvicorn`
- `python-multipart`
- `opencv-python`
- `mss` (required for screen capture endpoint)

## Quick Validation

From repository root:

```bash
./.venv/Scripts/python -m py_compile src/api_server.py src/inference.py
```
