const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/+$/, "");

async function parseJsonResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail || response.statusText || "Request failed";
    throw new Error(detail);
  }
  return payload;
}

export async function fetchModelStatus() {
  const response = await fetch(`${API_BASE}/model/status`, {
    method: "GET",
    cache: "no-store",
  });
  return parseJsonResponse(response);
}

export async function fetchLatestInference() {
  const response = await fetch(`${API_BASE}/inference/latest`, {
    method: "GET",
    cache: "no-store",
  });
  return parseJsonResponse(response);
}

export async function fetchSources() {
  const response = await fetch(`${API_BASE}/sources?max_camera_index=1`, {
    method: "GET",
    cache: "no-store",
  });
  return parseJsonResponse(response);
}

export async function selectModel(modelPath) {
  const response = await fetch(`${API_BASE}/model/select`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model_path: modelPath }),
  });
  return parseJsonResponse(response);
}

function appendInferenceOptions(formData, confThreshold, iouThreshold, modelPath) {
  if (typeof confThreshold === "number") {
    formData.append("conf_threshold", String(confThreshold));
  }
  if (typeof iouThreshold === "number") {
    formData.append("iou_threshold", String(iouThreshold));
  }
  if (typeof modelPath === "string" && modelPath.trim()) {
    formData.append("model_path", modelPath.trim());
  }
}

export async function inferImage(file, confThreshold, iouThreshold, modelPath) {
  const formData = new FormData();
  formData.append("file", file);
  appendInferenceOptions(formData, confThreshold, iouThreshold, modelPath);

  const response = await fetch(`${API_BASE}/inference/image`, {
    method: "POST",
    body: formData,
  });
  return parseJsonResponse(response);
}

export async function inferCamera(
  deviceIndex,
  confThreshold,
  iouThreshold,
  modelPath
) {
  const formData = new FormData();
  formData.append("device_index", String(deviceIndex));
  appendInferenceOptions(formData, confThreshold, iouThreshold, modelPath);

  const response = await fetch(`${API_BASE}/inference/camera`, {
    method: "POST",
    body: formData,
  });
  return parseJsonResponse(response);
}

export async function inferScreen(
  screenIndex,
  confThreshold,
  iouThreshold,
  modelPath
) {
  const formData = new FormData();
  formData.append("screen_index", String(screenIndex));
  appendInferenceOptions(formData, confThreshold, iouThreshold, modelPath);

  const response = await fetch(`${API_BASE}/inference/screen`, {
    method: "POST",
    body: formData,
  });
  return parseJsonResponse(response);
}
