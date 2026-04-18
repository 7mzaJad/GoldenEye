# Models Folder

This folder holds model artifacts used by both the root pipeline and desktop/web inference paths.

## Typical Layout

```text
models/
├── Yolov8n/
│   ├── best.pt
│   └── best.onnx
└── Yolov26/
    ├── best.pt
    └── best.onnx
```

## Which Components Use Which Files

- Root web API (`src/api_server.py`) defaults to:
  - `models/Yolov8n/best.pt`
- Root training/inference modules (`src/*.py`) also use `.pt` checkpoints.
- Desktop app (`Inference/`) uses `.onnx` paths configured in `Inference/config.yaml`.

## Adding or Replacing Models

1. Add files under an existing model folder or create a new one.
2. For root API/runtime selection:
   - use `POST /api/model/select` with absolute or resolvable model path.
3. For desktop app:
   - update `Inference/config.yaml` under `models:` entries.

## Supported Formats in API Discovery

`/api/model/status` artifact discovery currently lists:

- `.pt`
- `.onnx`

for all files under the `models/` directory.
