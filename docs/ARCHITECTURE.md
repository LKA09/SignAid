# Architecture

## Forward emergency communication

```text
Korean text / preset
        ↓
Offline intent parser (regex + aliases + fuzzy matching)
        ↓
Emergency intent + Korean Sign Language glosses
        ↓
Exact / alias / fuzzy motion retrieval
        ↓
FPS normalization + interpolated blending
        ↓
3D body + palm normal + head/face feature blending
        ↓
Browser VRM renderer ── fallback ── Three.js skeleton / optional MP4 export
        ↓
React emergency UI
```

FastAPI owns orchestration, validation, and file serving. Core modules contain no web framework assumptions. The UI first requests `/api/text-to-sign`, displays the explainable intent/gloss result, then requests `/api/render` and drives the local VRM model directly from the returned compact motion. MP4/GIF export is opt-in.

## Reverse recognition preparation

```text
Webcam
   ↓
MediaPipe Holistic (optional future capture adapter)
   ↓
Normalized pose + hand keypoint sequence
   ↓
GRU baseline / Transformer baseline / KSL-main 47-joint ST-GCN adapter
   ↓
Emergency class
   ↓
Korean text display
```

`/api/recognize` accepts a `(T,J,2|3)` JSON sequence. Without a trained checkpoint it returns `503`; deterministic mock output requires the explicit development-only `allow_mock` flag. This API does not change when a trained model is connected.

## Dependency boundaries

- SignAid core: Python 3.11, NumPy, FastAPI, matplotlib.
- Recognition: optional modern PyTorch environment.
- SignAvatars: separate legacy CUDA environment/container, native SMPL-X data only.
- Frontend: React + Vite; it communicates only through the HTTP API.
