from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from signaid.avatar.signavatars_adapter import SignAvatarsAdapter
from signaid.config import settings
from signaid.datasets.index import AIHubEmergencyIndex
from signaid.emergency.taxonomy import CONCEPTS
from signaid.motion.blender import blend_motions
from signaid.motion.constraints import add_avatar_hand_clearance
from signaid.motion.retriever import MotionRetriever
from signaid.recognition.pipeline import RecognitionPipeline
from signaid.text.parser import OfflineKoreanEmergencyParser


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class RenderRequest(BaseModel):
    text: str | None = Field(default=None, max_length=1000)
    gloss: list[str] | None = Field(default=None, max_length=20)
    include_media: bool = False

    @model_validator(mode="after")
    def has_input(self):
        if not self.text and not self.gloss:
            raise ValueError("text or gloss is required")
        if self.gloss and any(not word.strip() or len(word) > 50 for word in self.gloss):
            raise ValueError("each gloss must contain between 1 and 50 characters")
        return self


class RecognizeRequest(BaseModel):
    keypoints: list = Field(min_length=1, max_length=600)
    allow_mock: bool = False


class DatasetSearchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=3, ge=1, le=10)


app = FastAPI(title="SignAid API", version="0.2.0", description="Emergency Korean Sign Language service")
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "SIGNAID_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
settings.render_dir.mkdir(parents=True, exist_ok=True)
app.mount("/renders", StaticFiles(directory=settings.render_dir), name="renders")
avatar_asset_dir = Path(__file__).resolve().parents[3] / "third_party" / "vrm-samples" / "vroid"
if avatar_asset_dir.exists():
    app.mount("/avatar-assets", StaticFiles(directory=avatar_asset_dir), name="avatar-assets")

parser = OfflineKoreanEmergencyParser()
retriever = MotionRetriever()
recognizer = RecognitionPipeline()
dataset_index = AIHubEmergencyIndex()

INTENT_LABELS = {concept_id: concept.ko for concept_id, concept in CONCEPTS.items()}
INTENT_LABELS.update({
    "EVACUATION_INSTRUCTION": "엘리베이터 대신 계단으로 대피",
    "COMPOUND": "복합 응급 안내",
    "CUSTOM": "선택한 수어",
    "UNKNOWN": "확인 필요",
})


def _blend_feature_clips(
    clips: list[np.ndarray],
    source_fps: list[float],
    target_fps: float,
    transition_frames: int,
) -> np.ndarray:
    """FPS-normalize and blend arbitrary per-frame avatar features."""
    normalized: list[np.ndarray] = []
    for clip, fps in zip(clips, source_fps):
        target_frames = max(1, round(len(clip) * target_fps / fps))
        if target_frames == len(clip):
            normalized.append(clip.astype(np.float32, copy=True))
            continue
        old_t = np.linspace(0.0, 1.0, len(clip))
        new_t = np.linspace(0.0, 1.0, target_frames)
        flat = clip.reshape(len(clip), -1)
        sampled = np.stack([np.interp(new_t, old_t, flat[:, index]) for index in range(flat.shape[1])], axis=1)
        normalized.append(sampled.reshape((target_frames, *clip.shape[1:])).astype(np.float32))
    output = normalized[0]
    for clip in normalized[1:]:
        count = min(transition_frames, len(output), len(clip))
        if count:
            alpha = np.linspace(0, 1, count + 2, dtype=np.float32)[1:-1]
            alpha = alpha.reshape((count,) + (1,) * (output.ndim - 1))
            bridge = output[-1:] * (1 - alpha) + clip[:1] * alpha
            output = np.concatenate((output, bridge, clip), axis=0)
        else:
            output = np.concatenate((output, clip), axis=0)
    return output.astype(np.float32)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > 8 * 1024 * 1024:
                return JSONResponse(status_code=413, content={"detail": "요청 데이터가 너무 큽니다."})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "잘못된 Content-Length 헤더입니다."})
    return await call_next(request)


def _text_result(text: str) -> tuple[dict, object]:
    parsed = parser.parse(text)
    retrieval = retriever.retrieve(parsed.gloss)
    response = parsed.to_dict()
    response.update({
        "intent_label": INTENT_LABELS.get(parsed.intent, parsed.intent),
        "motions": [match.to_dict() for match in retrieval.matches],
        "missing": retrieval.missing,
        "motion_confidence": round(retrieval.confidence, 3),
        "reference_samples": dataset_index.search(text, limit=2),
        "supported": parsed.intent != "UNKNOWN" and not retrieval.missing,
    })
    return response, retrieval


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
def service_status() -> dict:
    real_motions = sum(
        1 for concept_id in CONCEPTS
        if (retriever.real_motion_dir / f"{concept_id}.npz").exists()
    )
    return {
        "status": "ok",
        "text_to_sign": True,
        "recognition": recognizer.model is not None,
        "real_motions": real_motions,
        "three_dimensional_motions": sum(
            1 for concept_id in CONCEPTS
            if (path := retriever.real_motion_dir / f"{concept_id}.npz").exists()
            and _motion_source(path) == "aihub_landmarks_3d"
        ),
        "total_concepts": len(CONCEPTS),
        "dataset": dataset_index.status(),
    }


def _motion_source(path: Path) -> str:
    try:
        with np.load(path) as data:
            return str(np.asarray(data.get("motion_source", "unknown")).item())
    except (OSError, ValueError, KeyError):
        return "unavailable"


@app.get("/api/emergency-signs")
def emergency_signs() -> list[dict]:
    result = []
    for concept in CONCEPTS.values():
        item = concept.to_dict()
        path = retriever.path_for_concept(concept.id)
        item.pop("motion_path", None)
        item["motion_available"] = path.exists()
        try:
            with np.load(path) as data:
                item["motion_source"] = str(np.asarray(data.get("motion_source", "unknown")).item())
                item["mock_motion"] = bool(np.asarray(data.get("mock", True)).item())
                item["linguistically_validated"] = bool(np.asarray(data.get("linguistically_validated", False)).item())
                item["annotation_aligned"] = bool(np.asarray(data.get("annotation_aligned", False)).item())
                item["expert_validated"] = bool(np.asarray(data.get("expert_validated", False)).item())
                item["landmark_dimensions"] = int(np.asarray(data.get("landmark_dimensions", 2)).item())
                item["tracking_quality"] = round(float(np.asarray(data.get("tracking_quality", 0)).item()), 4)
                item["sample_id"] = str(np.asarray(data.get("sample_id", "")).item())
        except (OSError, ValueError, KeyError):
            item.update({
                "motion_source": "unavailable", "mock_motion": True,
                "linguistically_validated": False, "annotation_aligned": False,
                "expert_validated": False, "landmark_dimensions": 0,
                "tracking_quality": 0.0, "sample_id": "",
            })
        result.append(item)
    return result


@app.get("/api/dataset/status")
def dataset_status() -> dict:
    return dataset_index.status()


@app.post("/api/dataset/search")
def dataset_search(request: DatasetSearchRequest) -> dict:
    return {"query": request.text, "results": dataset_index.search(request.text, request.limit)}


@app.post("/api/text-to-sign")
def text_to_sign(request: TextRequest) -> dict:
    return _text_result(request.text)[0]


@app.post("/api/render")
def render(request: RenderRequest) -> dict:
    unknown_intent = False
    matched_concepts: list[str] = []
    if request.gloss:
        gloss = [word.strip() for word in request.gloss]
        intent = "CUSTOM"
    else:
        parsed = parser.parse(request.text or "")
        gloss, intent = parsed.gloss, parsed.intent
        matched_concepts = parsed.matched_concepts
        unknown_intent = intent == "UNKNOWN"
    retrieval = retriever.retrieve(gloss)
    requested_missing = list(retrieval.missing)
    fallback_motion = False
    if unknown_intent or not retrieval.paths:
        fallback_motion = True
        requested_missing = requested_missing or gloss or [request.text or "알 수 없는 문장"]
        retrieval = retriever.retrieve(["도움"])
        if not retrieval.paths:
            raise HTTPException(status_code=503, detail="Fallback motion database is unavailable")
    clips, fps_values = [], []
    mock_values, source_values, validation_values, expert_values = [], [], [], []
    alignment_values, dimension_values, tracking_values, sample_ids = [], [], [], []
    palm_clips, expression_clips, head_clips = [], [], []
    for path in retrieval.paths:
        with np.load(path) as data:
            clip = np.asarray(data["motion"], dtype=np.float32)
            source_value = str(np.asarray(data.get("motion_source", "unknown")).item())
            if "aihub_keypoints_2d" in source_value:
                clip = add_avatar_hand_clearance(clip)
            clips.append(clip)
            clip_fps = float(data.get("fps", settings.fps))
            fps_values.append(clip_fps)
            palm_clips.append(np.asarray(data["palm_normals"], dtype=np.float32) if "palm_normals" in data else np.zeros((len(clip), 2, 3), dtype=np.float32))
            expression_clips.append(np.asarray(data["facial_expressions"], dtype=np.float32) if "facial_expressions" in data else np.zeros((len(clip), 4), dtype=np.float32))
            head_clips.append(np.asarray(data["head_rotations"], dtype=np.float32) if "head_rotations" in data else np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (len(clip), 1)))
            mock_values.append(bool(np.asarray(data.get("mock", True)).item()))
            source_values.append(source_value)
            validation_values.append(bool(np.asarray(data.get("linguistically_validated", False)).item()))
            expert_values.append(bool(np.asarray(data.get("expert_validated", False)).item()))
            alignment_values.append(bool(np.asarray(data.get("annotation_aligned", False)).item()))
            dimension_values.append(int(np.asarray(data.get("landmark_dimensions", 2)).item()))
            tracking_values.append(float(np.asarray(data.get("tracking_quality", 0)).item()))
            sample_id = str(np.asarray(data.get("sample_id", "")).item())
            if sample_id:
                sample_ids.append(sample_id)
    motion = blend_motions(clips, settings.transition_frames, fps_values, settings.fps)
    palm_normals = _blend_feature_clips(palm_clips, fps_values, settings.fps, settings.transition_frames)
    facial_expressions = _blend_feature_clips(expression_clips, fps_values, settings.fps, settings.transition_frames)
    head_rotations = _blend_feature_clips(head_clips, fps_values, settings.fps, settings.transition_frames)
    palm_lengths = np.linalg.norm(palm_normals, axis=-1, keepdims=True)
    normalized_palms = np.zeros_like(palm_normals)
    np.divide(palm_normals, np.maximum(palm_lengths, 1e-6), out=normalized_palms, where=palm_lengths > 1e-6)
    palm_normals = normalized_palms
    head_rotations /= np.maximum(np.linalg.norm(head_rotations, axis=-1, keepdims=True), 1e-6)
    is_mock = any(mock_values)
    motion_source = "+".join(dict.fromkeys(source_values))
    is_validated = bool(validation_values) and all(validation_values)
    is_expert_validated = bool(expert_values) and all(expert_values)
    is_annotation_aligned = bool(alignment_values) and all(alignment_values)
    cache_key = hashlib.sha256(json.dumps({
        "intent": intent,
        "gloss": gloss,
        "sources": [
            f"{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
            for path in retrieval.paths
        ],
        "fps": settings.fps,
        "transition_frames": settings.transition_frames,
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    safe_stem = f"{intent.lower()}_{cache_key}"
    combined_path = settings.motion_dir / f"{safe_stem}.npz"
    if not combined_path.exists():
        with tempfile.NamedTemporaryFile(dir=settings.motion_dir, suffix=".npz", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            np.savez_compressed(
                temp_path,
                motion=motion,
                fps=np.int32(settings.fps),
                mock=np.bool_(is_mock),
                motion_source=np.array(motion_source),
                linguistically_validated=np.bool_(is_validated),
                annotation_aligned=np.bool_(is_annotation_aligned),
                expert_validated=np.bool_(is_expert_validated),
                palm_normals=palm_normals,
                facial_expressions=facial_expressions,
                head_rotations=head_rotations,
            )
            os.replace(temp_path, combined_path)
        finally:
            temp_path.unlink(missing_ok=True)

    rendered = None
    renderer_name = "vrm_3d"
    fallback_reason = None
    if request.include_media:
        requested = settings.render_dir / f"{safe_stem}.mp4"
        cached = requested if requested.exists() else requested.with_suffix(".gif")
        if cached.exists():
            rendered = cached
            renderer_name = "cached_media"
        else:
            media_renderer = SignAvatarsAdapter()
            rendered = media_renderer.render(combined_path, requested)
            fallback_reason = media_renderer.last_error
            renderer_name = "skeleton_video" if fallback_reason else "signavatars"
    return {
        "intent": intent,
        "intent_label": INTENT_LABELS.get(intent, intent),
        "matched_concepts": matched_concepts,
        "gloss": gloss,
        "missing": requested_missing,
        "renderer": renderer_name,
        "fallback_reason": fallback_reason,
        "url": f"/renders/{rendered.name}" if rendered else None,
        "frames": len(motion),
        "fps": settings.fps,
        "mock_motion": is_mock,
        "motion_source": motion_source,
        "linguistically_validated": is_validated,
        "annotation_aligned": is_annotation_aligned,
        "expert_validated": is_expert_validated,
        "landmark_dimensions": min(dimension_values) if dimension_values else 0,
        "tracking_quality": round(min(tracking_values), 4) if tracking_values else 0.0,
        "sample_ids": sample_ids,
        "fallback_motion": fallback_motion,
        "motion": np.round(motion, 4).tolist(),
        "palm_normals": np.round(palm_normals, 4).tolist(),
        "facial_expressions": np.round(facial_expressions, 4).tolist(),
        "head_rotations": np.round(head_rotations, 5).tolist(),
    }


@app.post("/api/recognize")
def recognize(request: RecognizeRequest) -> dict:
    if recognizer.model is None and not request.allow_mock:
        raise HTTPException(status_code=503, detail="수어 인식 모델이 아직 구성되지 않았습니다.")
    try:
        points = np.asarray(request.keypoints, dtype=np.float32)
        if points.ndim not in (2, 3):
            raise ValueError("keypoints must have shape (T, J, 2|3)")
        joint_count = points.shape[-2]
        if joint_count < 1 or joint_count > 100:
            raise ValueError("keypoints must contain between 1 and 100 joints")
        if not np.isfinite(points).all():
            raise ValueError("keypoints must contain only finite numbers")
        return recognizer.predict(points)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


web_dist = settings.project_root / "app" / "web" / "dist"
if web_dist.joinpath("index.html").exists():
    app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="web-assets")

    @app.get("/", include_in_schema=False)
    def web_index() -> FileResponse:
        return FileResponse(web_dist / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def web_fallback(path: str):
        candidate = (web_dist / path).resolve()
        if candidate.is_relative_to(web_dist.resolve()) and candidate.is_file():
            return FileResponse(candidate)
        if path.startswith(("api/", "renders/", "avatar-assets/")) or Path(path).suffix:
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(web_dist / "index.html")
