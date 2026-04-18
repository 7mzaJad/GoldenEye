export const telemetry = [
  { label: "mAP50", value: 0.9469, tone: "signal" },
  { label: "mAP50-95", value: 0.6585, tone: "accent" },
  { label: "Precision", value: 0.959, tone: "signal" },
  { label: "Recall", value: 0.9, tone: "warning" },
];

export const modelDeck = [
  {
    key: "yolov8n",
    title: "YOLOv8n",
    role: "Primary root pipeline model",
    artifacts: ["best.pt 6.0MB", "best.onnx 12MB"],
    shape: "640 x 640",
    status: "Ready",
  },
  {
    key: "yolov26",
    title: "YOLOv26",
    role: "NMS-free desktop inference model",
    artifacts: ["best.pt 5.2MB", "best.onnx 9.4MB"],
    shape: "300 detections x 6",
    status: "Ready",
  },
];

export const timeline = [
  {
    stage: "Dataset Validation",
    detail: "Current run blocked by missing train/images and valid/images",
    state: "blocked",
  },
  {
    stage: "Root Training",
    detail: "Ultralytics YOLO pipeline is wired and import-safe",
    state: "armed",
  },
  {
    stage: "Desktop Inference",
    detail: "ONNX engine with YOLOv8 and YOLOv26 decoding paths",
    state: "live",
  },
  {
    stage: "Historical Evaluation",
    detail: "Notebook run logged 1144 images and 1463 instances",
    state: "archive",
  },
];
