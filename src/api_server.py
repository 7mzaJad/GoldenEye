"""FastAPI service for live model status and image inference results."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import mss
except ImportError:
    mss = None

from config import (
    BEST_MODEL_PATH,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    MODELS_DIR,
)
from inference import YOLOInference


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_model_identifier(model_path: str) -> str:
    candidate = Path(model_path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return str(model_path)


def _discover_model_artifacts() -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    if not MODELS_DIR.exists():
        return artifacts

    for candidate in sorted(MODELS_DIR.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in {".pt", ".onnx"}:
            continue

        size_mb = round(candidate.stat().st_size / (1024 * 1024), 2)
        artifacts.append(
            {
                "name": candidate.stem,
                "path": str(candidate.resolve()),
                "format": candidate.suffix.lower().lstrip("."),
                "size_mb": size_mb,
                "exists": True,
            }
        )

    return artifacts


def _encode_preview_jpeg(frame_bgr: np.ndarray) -> str | None:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _scan_cameras(max_index: int = 5) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        ok, _ = cap.read()
        cap.release()
        if not ok:
            continue

        devices.append({"index": idx, "name": f"Camera {idx}"})
    return devices


def _scan_screens() -> dict[str, Any]:
    if mss is None:
        return {
            "supported": False,
            "devices": [],
        }

    devices: list[dict[str, Any]] = []
    try:
        with mss.mss() as sct:
            monitors = sct.monitors
            for index, _ in enumerate(monitors):
                devices.append(
                    {
                        "index": index,
                        "name": "All Displays" if index == 0 else f"Display {index}",
                    }
                )
    except Exception:
        return {
            "supported": False,
            "devices": [],
        }

    return {
        "supported": True,
        "devices": devices,
    }


def _capture_camera_frame(device_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(
            status_code=400,
            detail=f"Unable to open camera index {device_index}.",
        )

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read a frame from camera index {device_index}.",
        )

    return frame


def _capture_screen_frame(screen_index: int) -> np.ndarray:
    if mss is None:
        raise HTTPException(
            status_code=501,
            detail="Screen capture dependency is not installed.",
        )

    try:
        with mss.mss() as sct:
            monitors = sct.monitors
            if screen_index < 0 or screen_index >= len(monitors):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid screen index {screen_index}.",
                )
            shot = sct.grab(monitors[screen_index])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to capture screen: {exc}",
        ) from exc

    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
    return frame


def _resolve_thresholds(
    conf_threshold: float | None,
    iou_threshold: float | None,
) -> tuple[float, float]:
    conf = (
        float(conf_threshold)
        if conf_threshold is not None
        else float(DEFAULT_CONF_THRESHOLD)
    )
    iou = (
        float(iou_threshold)
        if iou_threshold is not None
        else float(DEFAULT_IOU_THRESHOLD)
    )

    if not (0.0 <= conf <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="conf_threshold must be between 0 and 1.",
        )
    if not (0.0 <= iou <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="iou_threshold must be between 0 and 1.",
        )

    return conf, iou


class ModelSelectionRequest(BaseModel):
    model_path: str


class LiveInferenceService:
    """Thread-safe inference facade for API handlers."""

    def __init__(self, model_path: str = BEST_MODEL_PATH):
        self.model_path = _normalize_model_identifier(model_path)
        self._predictor: YOLOInference | None = None
        self._last_result: dict[str, Any] | None = None
        self._last_preview_jpeg: str | None = None
        self._lock = Lock()

    @property
    def model_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def last_result(self) -> dict[str, Any] | None:
        return self._last_result

    @property
    def last_preview_jpeg(self) -> str | None:
        return self._last_preview_jpeg

    def _ensure_predictor(self) -> YOLOInference:
        if self._predictor is None:
            self._predictor = YOLOInference(model_path=self.model_path)
        return self._predictor

    def set_model_path(self, model_path: str) -> dict[str, Any]:
        with self._lock:
            normalized = _normalize_model_identifier(model_path)
            changed = normalized != self.model_path
            self.model_path = normalized
            if changed:
                self._predictor = None

            return {
                "model_path": self.model_path,
                "changed": changed,
                "model_loaded": self._predictor is not None,
            }

    def infer_frame(
        self,
        image_bgr: np.ndarray,
        filename: str,
        conf_threshold: float,
        iou_threshold: float,
        model_path: str | None = None,
        source: str = "image",
    ) -> dict[str, Any]:
        with self._lock:
            if model_path:
                normalized = _normalize_model_identifier(model_path)
                if normalized != self.model_path:
                    self.model_path = normalized
                    self._predictor = None

            predictor = self._ensure_predictor()
            inference, annotated = predictor.predict_array_with_preview(
                image_bgr,
                conf=conf_threshold,
                iou=iou_threshold,
            )

            payload = {
                "filename": filename,
                "source": source,
                "received_at": _utc_now_iso(),
                "model_path": self.model_path,
                "thresholds": {
                    "confidence": conf_threshold,
                    "iou": iou_threshold,
                },
                **inference,
            }

            preview_jpeg = _encode_preview_jpeg(annotated)
            self._last_result = payload
            self._last_preview_jpeg = preview_jpeg

            return {
                "inference": payload,
                "preview_jpeg_base64": preview_jpeg,
            }


app = FastAPI(title="GoldenEye Live API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = LiveInferenceService()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/model/status")
def model_status() -> dict[str, Any]:
    configured_path = Path(service.model_path).expanduser()
    path_exists = configured_path.exists()
    return {
        "service": "online",
        "checked_at": _utc_now_iso(),
        "model_loaded": service.model_loaded,
        "configured_model_path": service.model_path,
        "configured_model_exists": path_exists,
        "defaults": {
            "confidence": DEFAULT_CONF_THRESHOLD,
            "iou": DEFAULT_IOU_THRESHOLD,
        },
        "available_models": _discover_model_artifacts(),
        "last_inference_at": (
            service.last_result.get("received_at") if service.last_result else None
        ),
    }


@app.post("/api/model/select")
def select_model(payload: ModelSelectionRequest) -> dict[str, Any]:
    if not payload.model_path.strip():
        raise HTTPException(status_code=400, detail="model_path cannot be empty.")

    result = service.set_model_path(payload.model_path)
    return {
        "ok": True,
        **result,
    }


@app.get("/api/sources")
def list_sources(max_camera_index: int = 1) -> dict[str, Any]:
    if max_camera_index < 0 or max_camera_index > 20:
        raise HTTPException(
            status_code=400,
            detail="max_camera_index must be between 0 and 20.",
        )

    screens = _scan_screens()
    return {
        "camera_devices": _scan_cameras(max_index=max_camera_index),
        "screen_devices": screens["devices"],
        "screen_supported": screens["supported"],
    }


@app.get("/api/inference/latest")
def latest_inference() -> dict[str, Any]:
    return {
        "inference": service.last_result,
        "preview_jpeg_base64": service.last_preview_jpeg,
    }


@app.post("/api/inference/image")
async def infer_image(
    file: UploadFile = File(...),
    conf_threshold: float | None = Form(None),
    iou_threshold: float | None = Form(None),
    model_path: str | None = Form(None),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded filename.")

    conf, iou = _resolve_thresholds(conf_threshold, iou_threshold)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    np_buffer = np.frombuffer(content, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(
            status_code=400, detail="Unsupported or corrupt image payload."
        )

    try:
        inference_payload = service.infer_frame(
            image_bgr=image_bgr,
            filename=file.filename,
            conf_threshold=conf,
            iou_threshold=iou,
            model_path=model_path,
            source="image",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "ok": True,
        **inference_payload,
    }


@app.post("/api/inference/camera")
def infer_camera(
    device_index: int = Form(0),
    conf_threshold: float | None = Form(None),
    iou_threshold: float | None = Form(None),
    model_path: str | None = Form(None),
) -> dict[str, Any]:
    conf, iou = _resolve_thresholds(conf_threshold, iou_threshold)
    frame = _capture_camera_frame(device_index)

    try:
        inference_payload = service.infer_frame(
            image_bgr=frame,
            filename=f"camera-{device_index}.jpg",
            conf_threshold=conf,
            iou_threshold=iou,
            model_path=model_path,
            source="camera",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "ok": True,
        **inference_payload,
    }


@app.post("/api/inference/screen")
def infer_screen(
    screen_index: int = Form(0),
    conf_threshold: float | None = Form(None),
    iou_threshold: float | None = Form(None),
    model_path: str | None = Form(None),
) -> dict[str, Any]:
    conf, iou = _resolve_thresholds(conf_threshold, iou_threshold)
    frame = _capture_screen_frame(screen_index)

    try:
        inference_payload = service.infer_frame(
            image_bgr=frame,
            filename=f"screen-{screen_index}.jpg",
            conf_threshold=conf,
            iou_threshold=iou,
            model_path=model_path,
            source="screen",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "ok": True,
        **inference_payload,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
