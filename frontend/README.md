# GoldenEye Frontend

This folder contains the React + Vite web interface for live GoldenEye inference controls and results.

## Stack

- React 18
- Vite 5

## Prerequisites

- Node.js 18+
- npm
- Running backend API (default: http://127.0.0.1:8000)

## Setup

```bash
cd frontend
npm install
```

## Development

```bash
npm run dev
```

Default local URL:

- http://localhost:5173

By default, Vite proxies `/api` requests to `http://127.0.0.1:8000` via `vite.config.js`.

## Build and Preview

```bash
npm run build
npm run preview
```

## Optional Environment Variable

You can override API base path by setting:

- `VITE_API_BASE_URL`

Example:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000/api npm run dev
```

## Main Features in This UI

- Model selection from discovered backend artifacts.
- Source mode selection: image, camera, or screen.
- Camera/screen device selection with rescan.
- Confidence and IoU slider controls.
- Single-run inference and live start/stop capture mode.
- Annotated preview panel from `preview_jpeg_base64`.
- Detection stats and class-count summaries.

## API Contract Used

The frontend relies on these backend endpoints:

- `GET /api/model/status`
- `POST /api/model/select`
- `GET /api/sources`
- `GET /api/inference/latest`
- `POST /api/inference/image`
- `POST /api/inference/camera`
- `POST /api/inference/screen`

See `../src/README.md` for endpoint details.
