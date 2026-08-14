# SignAid

SignAid는 한국어 응급 문장을 한국수어 글로스와 3D 아바타 동작으로 바꾸는 로컬 웹앱입니다. 한 문장 안의 여러 표현을 순서대로 보존하고, AI Hub 3D 신체·양손·얼굴 랜드마크가 연결된 21개 동작과 2D 키포인트 기반 동작 1개를 우선 사용하며, 연결되지 않은 표현은 화면에서 시연용 동작임을 분명히 표시합니다.

## 바로 실행

Python 3.11 이상과 Node.js 20.19 이상이 필요합니다.

```powershell
cd SignAid
python -m pip install -e ".[dev]"
signaid
```

첫 실행에서는 웹 의존성을 설치하고 화면을 빌드한 뒤 `http://127.0.0.1:8000`을 엽니다. 이후에는 변경된 경우에만 다시 빌드합니다. 브라우저를 자동으로 열지 않으려면 `signaid --no-open`, 같은 네트워크의 다른 기기에서도 접속하려면 `signaid --host 0.0.0.0 --no-open`을 사용하세요.

> 이 서비스는 응급 소통을 돕는 보조 수단입니다. 실제 긴급상황에서는 119에 연락하고, 검증되지 않은 동작을 유일한 의사소통 수단으로 사용하지 마세요.

## What works now

- Offline Korean intent parsing with compound-expression, regex, alias, keyword, and fuzzy matching
- 31 emergency concepts, 21 AI Hub 3D motions, one AI Hub 2D motion, and explicit demo fallbacks
- Exact/alias/fuzzy motion retrieval with confidence and missing-motion reporting
- FPS normalization, clip concatenation, and interpolated transitions
- Real MP4 rendering when ffmpeg is installed, GIF fallback otherwise
- One-command FastAPI + React deployment with same-origin assets and API
- Korean speech input, request timeout/retry, playback-speed control, and avatar fallback
- ZIP-streaming JSON/XML inspection and emergency filtering
- AI Hub's `.zip`-named 7-Zip archives and XLSX `Information` blocks
- 61,760 filtered real emergency sentences/gloss sequences across 16 categories
- GRU, Transformer, and direct KSL-main ST-GCN adapters
- Optional isolated SignAvatars subprocess with safe skeleton fallback

Mock status is explicit in API responses and the UI. Parser, retrieval, blending, rendering, API integration, ZIP inspection, and filtering are real implementations.

## Architecture

```text
Text → Intent → Gloss → Motion Retrieval → Blending → Skeleton / SignAvatars → Web
Webcam → Keypoints → GRU / Transformer / KSL ST-GCN → Emergency Intent → Korean
```

See [architecture](docs/ARCHITECTURE.md), [data pipeline](docs/DATA_PIPELINE.md), [reuse decisions](docs/THIRD_PARTY_ANALYSIS.md), and [license inventory](THIRD_PARTY_LICENSES.md).

## Folder structure

```text
app/api       FastAPI application
app/web       React/Vite frontend
signaid       parser, data, motion, avatar, recognition, model modules
scripts       dataset inspection/filtering, indexing, training entry points
configs       application, keywords, and training configuration
data          ignored raw/derived artifacts with tracked placeholders
tests         end-to-end and unit tests
docker        isolated SignAvatars environment
docs          architecture and third-party analysis
```

The existing sibling `../third_party` directory is never moved or cloned.

## 개발 모드

```bash
cd SignAid
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
uvicorn app.api.main:app --reload
```

In another terminal:

```bash
cd SignAid/app/web
npm ci
npm run dev
```

Open `http://localhost:5173`.

## Tests and demos

```powershell
pytest -q
python -m signaid.text.parser "엘리베이터를 이용하지 말고 계단으로 대피하세요"
python scripts\build_motion_index.py
python scripts\inspect_aihub_dataset.py --zip "C:\path\to\수어스크립트_TL.zip"
python scripts\import_aihub_dictionary_motions.py
python scripts\build_submission.py
```

Example API call:

```bash
curl -X POST http://localhost:8000/api/text-to-sign \
  -H "Content-Type: application/json" \
  -d '{"text":"가슴이 너무 아파요"}'
```

Useful endpoints: `GET /health`, `GET /api/status`, `GET /api/emergency-signs`, `GET /api/dataset/status`, `POST /api/dataset/search`, `POST /api/text-to-sign`, `POST /api/render`, and `POST /api/recognize`. Interactive API documentation is at `http://localhost:8000/docs`. `/api/render` returns browser-ready motion immediately; pass `"include_media": true` only when an MP4/GIF export is needed.

운영 환경에서는 [`.env.example`](.env.example)을 참고해 `SIGNAID_DATA_DIR`로 데이터 위치를, `SIGNAID_CORS_ORIGINS`로 쉼표 구분 허용 출처를 지정할 수 있습니다. 기본 구성은 프런트와 API를 같은 주소에서 제공하므로 별도 CORS 설정이 필요 없습니다.

## AI Hub integration after download

1. Run the archive inspector. It detects standard ZIP and AI Hub files that use a `.zip` name but contain 7-Zip/XLSX data.
2. Confirm field paths and encoding from its printed samples.
3. Run the emergency filter for all training/validation metadata ZIPs.
4. Join the resulting IDs to keypoint members and standardize selected samples to NPZ—do not extract everything.
5. Audit class and signer distributions, then create signer-disjoint train/validation splits.
6. Install `.[ml]`, validate the ST-GCN adapter on one batch, train baselines, and place versioned checkpoints in `checkpoints/`.
7. Configure `RecognitionPipeline` with the selected checkpoint/labels and remove the API mock flag only after validation.

## Third-party and current limitations

KSL-main graph/model code is imported directly. The disaster benchmark gloss helper is reused opportunistically. SignAvatars is called only across a subprocess boundary because it requires CUDA, legacy packages, native SMPL-X parameters, and model assets. Its bundled SMPL-X license is non-commercial; the viewer is GPLv3; the top-level repository and disaster repository lack clear standalone licenses. Read `THIRD_PARTY_LICENSES.md` before redistribution.

The current filtered index contains real AI Hub Korean sentences and time-ordered glosses. Twenty-one concepts use selectively extracted AI Hub 3D body, hand, and face landmarks; one additional concept uses AI Hub 2D keypoints. Their source spans are annotation-aligned but have not been reviewed by a KSL expert. Remaining concepts use clearly marked procedural demo clips. `/api/recognize` returns `503` without a trained checkpoint; deterministic mock output is available only when a developer explicitly sends `"allow_mock": true`. Full KSL linguistic review and production accessibility field testing are still required before clinical or public-safety certification.

## Compact submission

`python scripts\build_submission.py` creates `release/SignAid-submission.zip`. It includes the compact derived motions, local VRM avatar, web build, and application source while excluding the 143 GiB source archives, temporary extracts, rendered videos, checkpoints, and `node_modules`. See [submission notes](SUBMISSION.md).
