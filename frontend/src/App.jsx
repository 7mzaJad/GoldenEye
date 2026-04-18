import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchLatestInference,
  fetchModelStatus,
  fetchSources,
  inferCamera,
  inferImage,
  inferScreen,
  selectModel,
} from "./api";

const LIVE_INTERVAL_MS = 1400;
const FPS_WINDOW_MS = 5000;

function formatPercent(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function pathTail(value) {
  const normalized = String(value || "").trim();
  if (!normalized) {
    return "";
  }
  const chunks = normalized.split(/[/\\]/);
  return chunks[chunks.length - 1] || normalized;
}

function App() {
  const [statusPayload, setStatusPayload] = useState(null);
  const [sourcesPayload, setSourcesPayload] = useState({
    camera_devices: [],
    screen_devices: [],
    screen_supported: false,
  });
  const [latestInference, setLatestInference] = useState(null);
  const [previewBase64, setPreviewBase64] = useState("");

  const [selectedModelPath, setSelectedModelPath] = useState("");
  const [sourceType, setSourceType] = useState("image");
  const [selectedCameraIndex, setSelectedCameraIndex] = useState(0);
  const [selectedScreenIndex, setSelectedScreenIndex] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);

  const [confThreshold, setConfThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);

  const [isStatusLoading, setIsStatusLoading] = useState(true);
  const [isInferring, setIsInferring] = useState(false);
  const [isLiveRunning, setIsLiveRunning] = useState(false);
  const [inferenceState, setInferenceState] = useState("idle");
  const [liveFps, setLiveFps] = useState(null);
  const [apiError, setApiError] = useState("");

  const initializedRef = useRef(false);
  const runLockRef = useRef(false);
  const frameTimesRef = useRef([]);

  const availableModels = useMemo(() => {
    const discovered = statusPayload?.available_models;
    if (Array.isArray(discovered) && discovered.length > 0) {
      return discovered;
    }

    if (selectedModelPath) {
      return [
        {
          path: selectedModelPath,
          name: pathTail(selectedModelPath) || "Configured model",
          format: "custom",
          size_mb: "--",
        },
      ];
    }

    return [];
  }, [statusPayload, selectedModelPath]);

  const deviceOptions = useMemo(() => {
    if (sourceType === "camera") {
      return sourcesPayload.camera_devices ?? [];
    }
    if (sourceType === "screen") {
      return sourcesPayload.screen_devices ?? [];
    }
    return [];
  }, [sourceType, sourcesPayload]);

  const refreshData = async (includeSources = false) => {
    const requests = [fetchModelStatus(), fetchLatestInference()];
    if (includeSources) {
      requests.push(fetchSources());
    }

    const [status, latest, sources] = await Promise.all(requests);
    setStatusPayload(status);
    setLatestInference(latest?.inference ?? null);
    setPreviewBase64(latest?.preview_jpeg_base64 ?? "");

    if (sources) {
      setSourcesPayload(sources);
      setSelectedCameraIndex((current) => {
        const available = sources.camera_devices ?? [];
        if (available.some((item) => item.index === current)) {
          return current;
        }
        return available[0]?.index ?? 0;
      });
      setSelectedScreenIndex((current) => {
        const available = sources.screen_devices ?? [];
        if (available.some((item) => item.index === current)) {
          return current;
        }
        return available[0]?.index ?? 0;
      });
    }

    setSelectedModelPath((current) => current || status?.configured_model_path || "");

    if (!initializedRef.current) {
      setConfThreshold(Number(status?.defaults?.confidence ?? 0.25));
      setIouThreshold(Number(status?.defaults?.iou ?? 0.45));
      initializedRef.current = true;
    }
  };

  useEffect(() => {
    let mounted = true;

    const loadInitial = async () => {
      setIsStatusLoading(true);
      try {
        await refreshData(true);
        if (mounted) {
          setApiError("");
        }
      } catch (error) {
        if (mounted) {
          setApiError(error.message || "Unable to reach backend API");
        }
      } finally {
        if (mounted) {
          setIsStatusLoading(false);
        }
      }
    };

    loadInitial();
    const pollId = setInterval(() => {
      refreshData(false).catch(() => {
        // Silent polling failure keeps UI responsive.
      });
    }, 15000);

    return () => {
      mounted = false;
      clearInterval(pollId);
    };
  }, []);

  const runInferenceOnce = useCallback(
    async (silentError = false) => {
      if (runLockRef.current) {
        return;
      }

      if (sourceType === "image" && !selectedFile) {
        if (!silentError) {
          setApiError("Choose an image file before running inference.");
        }
        return;
      }

      if ((sourceType === "camera" || sourceType === "screen") && deviceOptions.length === 0) {
        if (!silentError) {
          setApiError("No capture device found for the selected source.");
        }
        return;
      }

      if (sourceType === "screen" && !sourcesPayload.screen_supported) {
        if (!silentError) {
          setApiError("Screen capture is not available in this backend environment.");
        }
        return;
      }

      runLockRef.current = true;
      setIsInferring(true);
      setInferenceState("running");
      if (!silentError) {
        setApiError("");
      }

      try {
        let response;
        if (sourceType === "image") {
          response = await inferImage(
            selectedFile,
            confThreshold,
            iouThreshold,
            selectedModelPath
          );
        } else if (sourceType === "camera") {
          response = await inferCamera(
            selectedCameraIndex,
            confThreshold,
            iouThreshold,
            selectedModelPath
          );
        } else {
          response = await inferScreen(
            selectedScreenIndex,
            confThreshold,
            iouThreshold,
            selectedModelPath
          );
        }

        const inference = response?.inference ?? null;
        setLatestInference(inference);
        setPreviewBase64(response?.preview_jpeg_base64 || "");

        setStatusPayload((current) => {
          if (!current) {
            return current;
          }
          return {
            ...current,
            model_loaded: true,
            configured_model_path:
              inference?.model_path || selectedModelPath || current.configured_model_path,
            last_inference_at: inference?.received_at || current.last_inference_at,
          };
        });

        if (sourceType === "camera" || sourceType === "screen") {
          const now = performance.now();
          frameTimesRef.current = frameTimesRef.current
            .filter((time) => now - time <= FPS_WINDOW_MS)
            .concat(now);

          if (frameTimesRef.current.length >= 2) {
            const elapsedSeconds =
              (frameTimesRef.current[frameTimesRef.current.length - 1] -
                frameTimesRef.current[0]) /
              1000;
            if (elapsedSeconds > 0) {
              setLiveFps((frameTimesRef.current.length - 1) / elapsedSeconds);
            }
          }
        } else {
          frameTimesRef.current = [];
          setLiveFps(null);
        }

        setInferenceState(isLiveRunning ? "running" : "idle");
      } catch (error) {
        setInferenceState("error");
        if (!silentError) {
          setApiError(error.message || "Inference request failed");
        }
      } finally {
        runLockRef.current = false;
        setIsInferring(false);
      }
    },
    [
      sourceType,
      selectedFile,
      selectedCameraIndex,
      selectedScreenIndex,
      confThreshold,
      iouThreshold,
      selectedModelPath,
      isLiveRunning,
      deviceOptions,
      sourcesPayload,
    ]
  );

  useEffect(() => {
    if (!isLiveRunning) {
      return;
    }
    if (sourceType === "image") {
      return;
    }

    let cancelled = false;
    const tick = async () => {
      if (cancelled) {
        return;
      }
      await runInferenceOnce(true);
    };

    tick();
    const intervalId = setInterval(tick, LIVE_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [
    isLiveRunning,
    sourceType,
    selectedCameraIndex,
    selectedScreenIndex,
    selectedModelPath,
    confThreshold,
    iouThreshold,
    runInferenceOnce,
  ]);

  const handleRefresh = async (includeSources = true) => {
    setIsStatusLoading(true);
    setApiError("");
    try {
      await refreshData(includeSources);
    } catch (error) {
      setApiError(error.message || "Refresh request failed");
    } finally {
      setIsStatusLoading(false);
    }
  };

  const handleSourceChange = (nextSource) => {
    setSourceType(nextSource);
    setApiError("");
    frameTimesRef.current = [];
    setLiveFps(null);
    if (isLiveRunning) {
      setIsLiveRunning(false);
      setInferenceState("idle");
    }
  };

  const handleStartLive = () => {
    if (sourceType === "image") {
      setApiError("Live capture is available for camera and screen only.");
      return;
    }
    setApiError("");
    setInferenceState("running");
    setIsLiveRunning(true);
  };

  const handleStopLive = () => {
    setIsLiveRunning(false);
    setInferenceState("idle");
    frameTimesRef.current = [];
    setLiveFps(null);
  };

  const handleModelChange = async (event) => {
    const nextModelPath = event.target.value;
    setSelectedModelPath(nextModelPath);
    setApiError("");

    if (!nextModelPath) {
      return;
    }

    try {
      await selectModel(nextModelPath);
      setStatusPayload((current) => {
        if (!current) {
          return current;
        }
        return {
          ...current,
          configured_model_path: nextModelPath,
          model_loaded: false,
        };
      });
    } catch (error) {
      setApiError(error.message || "Unable to switch model");
    }
  };

  const classCounts = latestInference?.class_counts
    ? Object.entries(latestInference.class_counts).sort((a, b) => b[1] - a[1])
    : [];

  const confidenceValues = Array.isArray(latestInference?.items)
    ? latestInference.items
        .map((item) => Number(item?.confidence ?? 0))
        .filter((value) => Number.isFinite(value))
    : [];

  const maxConfidence =
    confidenceValues.length > 0 ? Math.max(...confidenceValues) : 0;
  const avgConfidence =
    confidenceValues.length > 0
      ? confidenceValues.reduce((sum, value) => sum + value, 0) /
        confidenceValues.length
      : 0;

  const previewUrl = previewBase64
    ? `data:image/jpeg;base64,${previewBase64}`
    : "";

  const statusCheckedAt = statusPayload?.checked_at
    ? new Date(statusPayload.checked_at).toLocaleTimeString()
    : "Not checked yet";

  const lastInferenceTime = latestInference?.received_at
    ? new Date(latestInference.received_at).toLocaleTimeString()
    : "No runs yet";

  const selectedDeviceIndex =
    sourceType === "camera" ? selectedCameraIndex : selectedScreenIndex;

  return (
    <div className="page">
      <header className="header">
        <div>
          <p className="eyebrow">GoldenEye</p>
          <h1>Detection Console</h1>
          <p className="subhead">
            Simple controls with the same core flow: choose model, source, thresholds,
            then run image or live detection.
          </p>
        </div>
        <button
          className="btn btn-ghost"
          onClick={() => handleRefresh(true)}
          disabled={isStatusLoading}
        >
          {isStatusLoading ? "Refreshing..." : "Refresh"}
        </button>
      </header>

      <main className="layout">
        <section className="card controls-card">
          <h2>Controls</h2>

          <div className="field">
            <label htmlFor="model-select">Model</label>
            <select
              id="model-select"
              value={selectedModelPath}
              onChange={handleModelChange}
            >
              {availableModels.length === 0 ? (
                <option value="">No models discovered</option>
              ) : (
                availableModels.map((model) => (
                  <option key={model.path} value={model.path}>
                    {model.name || pathTail(model.path)} ({String(model.format).toUpperCase()})
                  </option>
                ))
              )}
            </select>
          </div>

          <div className="field">
            <label>Input source</label>
            <div className="segmented" role="tablist" aria-label="Input source">
              <button
                type="button"
                className={sourceType === "image" ? "active" : ""}
                onClick={() => handleSourceChange("image")}
              >
                Image
              </button>
              <button
                type="button"
                className={sourceType === "camera" ? "active" : ""}
                onClick={() => handleSourceChange("camera")}
              >
                Camera
              </button>
              <button
                type="button"
                className={sourceType === "screen" ? "active" : ""}
                onClick={() => handleSourceChange("screen")}
              >
                Screen
              </button>
            </div>
          </div>

          {sourceType === "image" ? (
            <div className="field">
              <label htmlFor="image-file">Image file</label>
              <label className="file-field" htmlFor="image-file">
                <input
                  id="image-file"
                  type="file"
                  accept="image/*"
                  onChange={(event) =>
                    setSelectedFile(event.target.files?.[0] ?? null)
                  }
                />
                <span>{selectedFile ? selectedFile.name : "Choose image"}</span>
              </label>
            </div>
          ) : (
            <div className="field">
              <label htmlFor="device-select">
                {sourceType === "camera" ? "Camera device" : "Screen device"}
              </label>
              <div className="inline-row">
                <select
                  id="device-select"
                  value={String(selectedDeviceIndex)}
                  onChange={(event) => {
                    const index = Number(event.target.value);
                    if (sourceType === "camera") {
                      setSelectedCameraIndex(index);
                    } else {
                      setSelectedScreenIndex(index);
                    }
                  }}
                >
                  {deviceOptions.length === 0 ? (
                    <option value="">No devices found</option>
                  ) : (
                    deviceOptions.map((device) => (
                      <option key={device.index} value={device.index}>
                        {device.name}
                      </option>
                    ))
                  )}
                </select>
                <button
                  className="btn btn-ghost"
                  type="button"
                  onClick={() => handleRefresh(true)}
                >
                  Rescan
                </button>
              </div>
              {sourceType === "screen" && !sourcesPayload.screen_supported ? (
                <p className="hint warning-text">
                  Screen capture is unavailable because backend screen dependency is missing.
                </p>
              ) : null}
            </div>
          )}

          <div className="field">
            <div className="slider-row">
              <label htmlFor="conf-threshold">Confidence</label>
              <strong>{formatPercent(confThreshold)}</strong>
            </div>
            <input
              id="conf-threshold"
              type="range"
              min="0.05"
              max="0.95"
              step="0.01"
              value={confThreshold}
              onChange={(event) => setConfThreshold(Number(event.target.value))}
            />
          </div>

          <div className="field">
            <div className="slider-row">
              <label htmlFor="iou-threshold">IoU (NMS)</label>
              <strong>{formatPercent(iouThreshold)}</strong>
            </div>
            <input
              id="iou-threshold"
              type="range"
              min="0.1"
              max="0.95"
              step="0.01"
              value={iouThreshold}
              onChange={(event) => setIouThreshold(Number(event.target.value))}
            />
          </div>

          <div className="action-row">
            <button
              className="btn btn-primary"
              type="button"
              onClick={() => runInferenceOnce(false)}
              disabled={isInferring || (sourceType === "image" && !selectedFile)}
            >
              {isInferring
                ? "Running..."
                : sourceType === "image"
                  ? "Run Image"
                  : "Run Snapshot"}
            </button>
            <button
              className="btn"
              type="button"
              onClick={handleStartLive}
              disabled={isLiveRunning || sourceType === "image"}
            >
              Start Live
            </button>
            <button
              className="btn"
              type="button"
              onClick={handleStopLive}
              disabled={!isLiveRunning}
            >
              Stop Live
            </button>
          </div>

          {apiError ? <p className="error-text">{apiError}</p> : null}
        </section>

        <section className="content-column">
          <section className="card status-card">
            <h2>Status</h2>
            <div className="status-grid">
              <article>
                <span>API</span>
                <strong className={statusPayload?.service === "online" ? "ok" : "warn"}>
                  {statusPayload?.service === "online" ? "Online" : "Offline"}
                </strong>
              </article>
              <article>
                <span>Model</span>
                <strong className={statusPayload?.model_loaded ? "ok" : "warn"}>
                  {statusPayload?.model_loaded ? "Loaded" : "Not loaded"}
                </strong>
              </article>
              <article>
                <span>Capture</span>
                <strong className={isLiveRunning ? "ok" : "muted"}>
                  {isLiveRunning ? "Running" : "Stopped"}
                </strong>
              </article>
              <article>
                <span>Inference</span>
                <strong
                  className={
                    inferenceState === "running"
                      ? "ok"
                      : inferenceState === "error"
                        ? "warn"
                        : "muted"
                  }
                >
                  {inferenceState}
                </strong>
              </article>
              <article>
                <span>Backend check</span>
                <strong className="muted">{statusCheckedAt}</strong>
              </article>
              <article>
                <span>Last inference</span>
                <strong className="muted">{lastInferenceTime}</strong>
              </article>
            </div>
          </section>

          <section className="card preview-card">
            <h2>Live view</h2>
            {previewUrl ? (
              <img src={previewUrl} alt="Latest annotated inference frame" />
            ) : (
              <div className="preview-placeholder">
                <p>No frame yet.</p>
                <p>Run image, camera, or screen inference to populate this panel.</p>
              </div>
            )}
          </section>

          <section className="card results-card">
            <h2>Detection results</h2>
            <div className="stats-grid">
              <article>
                <span>Humans detected</span>
                <strong>{latestInference?.detections ?? 0}</strong>
              </article>
              <article>
                <span>Max confidence</span>
                <strong>{maxConfidence > 0 ? formatPercent(maxConfidence) : "-"}</strong>
              </article>
              <article>
                <span>Avg confidence</span>
                <strong>{avgConfidence > 0 ? formatPercent(avgConfidence) : "-"}</strong>
              </article>
              <article>
                <span>Inference (ms)</span>
                <strong>
                  {Number.isFinite(latestInference?.latency_ms)
                    ? Number(latestInference.latency_ms).toFixed(1)
                    : "-"}
                </strong>
              </article>
              <article>
                <span>Live FPS</span>
                <strong>{liveFps ? liveFps.toFixed(1) : "-"}</strong>
              </article>
            </div>

            {classCounts.length > 0 ? (
              <ul className="class-list">
                {classCounts.map(([className, count]) => (
                  <li key={className}>
                    <span>{className}</span>
                    <strong>{count}</strong>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="hint">No class counts yet.</p>
            )}
          </section>
        </section>
      </main>
    </div>
  );
}

export default App;
