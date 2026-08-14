from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import py7zr

from signaid.config import settings
from signaid.datasets.aihub import list_archive_members
from signaid.datasets.aihub_motion import (
    FACIAL_EXPRESSION_NAMES,
    AvatarMotion3D,
    TimedGloss,
    landmark_json_to_avatar_motion,
    timed_glosses_from_landmark_json,
)
from signaid.emergency.taxonomy import CONCEPTS


def _default_archive(pattern: str) -> Path:
    matches = list(settings.data_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Archive matching {pattern!r} was not found")
    return matches[0]


def _load_records() -> list[dict]:
    path = settings.data_dir / "index" / "emergency_samples.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _landmark_members(path: Path) -> dict[str, str]:
    return {
        Path(item["name"]).stem: item["name"]
        for item in list_archive_members(path)
        if not item["is_directory"] and item["size"] and item["name"].endswith(".json")
    }


def _preferred_samples(output: Path) -> dict[str, str]:
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return {
            str(item["concept_id"]): str(item["sample_id"])
            for item in json.loads(manifest_path.read_text(encoding="utf-8"))
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def _candidate_records(
    records: list[dict], members: dict[str, str], preferred: dict[str, str], limit: int = 5,
) -> dict[str, list[dict]]:
    selected: dict[str, list[dict]] = {}
    for concept in CONCEPTS.values():
        candidates = [
            record for record in records
            if record.get("sample_id") in members
            and all(gloss in record.get("gloss", []) for gloss in concept.gloss)
        ]
        candidates.sort(key=lambda record: (
            record.get("sample_id") != preferred.get(concept.id),
            len(record.get("gloss", [])),
            record.get("sample_id", ""),
        ))
        if candidates:
            selected[concept.id] = candidates[:limit]
    return selected


def _event_for(events: list[TimedGloss], gloss: str) -> TimedGloss | None:
    matches = [event for event in events if event.gloss == gloss]
    return max(matches, key=lambda event: event.end - event.start) if matches else None


def _blend_features(clips: list[np.ndarray], transition_frames: int = 2) -> np.ndarray:
    output = clips[0]
    for clip in clips[1:]:
        count = min(transition_frames, len(output), len(clip))
        if count:
            alpha_shape = (count,) + (1,) * (output.ndim - 1)
            alpha = np.linspace(0, 1, count + 2, dtype=np.float32)[1:-1].reshape(alpha_shape)
            bridge = output[-1:] * (1 - alpha) + clip[:1] * alpha
            output = np.concatenate((output, bridge, clip), axis=0)
        else:
            output = np.concatenate((output, clip), axis=0)
    return output.astype(np.float32)


def _slice_features(source: AvatarMotion3D, events: list[TimedGloss]) -> dict[str, np.ndarray]:
    slices: dict[str, list[np.ndarray]] = {
        "motion": [],
        "palm_normals": [],
        "facial_expressions": [],
        "head_rotations": [],
    }
    for event in events:
        start = max(0, int(np.floor(event.start * source.fps)) - 2)
        end = min(len(source.motion), int(np.ceil(event.end * source.fps)) + 2)
        if end - start < 2:
            continue
        for key in slices:
            slices[key].append(getattr(source, key)[start:end])
    if not slices["motion"]:
        return {}
    result = {key: _blend_features(value) for key, value in slices.items()}
    result["palm_normals"] /= np.maximum(
        np.linalg.norm(result["palm_normals"], axis=-1, keepdims=True), 1e-6
    )
    result["head_rotations"] /= np.maximum(
        np.linalg.norm(result["head_rotations"], axis=-1, keepdims=True), 1e-6
    )
    return result


def main() -> None:
    cli = argparse.ArgumentParser(
        description="Selectively import AI Hub 3D body/hand/face landmarks without unpacking the full archive"
    )
    cli.add_argument("--landmark-archive", type=Path)
    cli.add_argument("--output", type=Path, default=settings.data_dir / "processed" / "aihub_motions")
    args = cli.parse_args()
    landmark_archive = args.landmark_archive or _default_archive("*형태소*비수지*json*.zip")
    members = _landmark_members(landmark_archive)
    candidates = _candidate_records(_load_records(), members, _preferred_samples(args.output))
    targets = sorted({
        members[record["sample_id"]]
        for concept_candidates in candidates.values()
        for record in concept_candidates
    })

    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="signaid_aihub_3d_") as temp:
        root = Path(temp)
        print(f"Selectively extracting {len(targets)} JSON records from {landmark_archive.name}…")
        with py7zr.SevenZipFile(landmark_archive, mode="r") as archive:
            archive.extract(path=root, targets=targets)

        for concept_id, concept_candidates in candidates.items():
            record = None
            avatar_motion = None
            features: dict[str, np.ndarray] = {}
            failure = "no usable candidate"
            for candidate in concept_candidates:
                member = members[candidate["sample_id"]]
                try:
                    payload = json.loads((root / Path(member)).read_text(encoding="utf-8-sig"))
                    timed = timed_glosses_from_landmark_json(payload)
                    events = [_event_for(timed, gloss) for gloss in CONCEPTS[concept_id].gloss]
                    if not all(events):
                        failure = "aligned gloss span was not found"
                        continue
                    candidate_motion = landmark_json_to_avatar_motion(payload)
                    candidate_features = _slice_features(candidate_motion, events)  # type: ignore[arg-type]
                    if not candidate_features:
                        failure = "aligned segment was empty"
                        continue
                    record, avatar_motion, features = candidate, candidate_motion, candidate_features
                    break
                except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
                    failure = str(exc)
            if record is None or avatar_motion is None:
                print(f"Skipped {concept_id}: {failure}")
                continue
            if not features:
                print(f"Skipped {concept_id}: aligned segment was empty")
                continue

            output = args.output / f"{concept_id}.npz"
            np.savez_compressed(
                output,
                **features,
                fps=np.int32(avatar_motion.fps),
                mock=np.bool_(False),
                motion_source=np.array("aihub_landmarks_3d"),
                annotation_aligned=np.bool_(True),
                expert_validated=np.bool_(False),
                linguistically_validated=np.bool_(False),
                landmark_dimensions=np.int32(3),
                tracking_quality=np.float32(avatar_motion.tracking_quality),
                facial_expression_names=np.array(FACIAL_EXPRESSION_NAMES),
                sample_id=np.array(record["sample_id"]),
                source_gloss=np.array(list(CONCEPTS[concept_id].gloss)),
            )
            manifest.append({
                "concept_id": concept_id,
                "sample_id": record["sample_id"],
                "gloss": list(CONCEPTS[concept_id].gloss),
                "frames": len(features["motion"]),
                "fps": avatar_motion.fps,
                "source": "AIHUB_3D_LANDMARKS",
                "tracking_quality": round(avatar_motion.tracking_quality, 4),
                "annotation_aligned": True,
                "expert_validated": False,
            })
            print(
                f"Imported {concept_id}: {len(features['motion'])} frames from "
                f"{record['sample_id']} (tracking {avatar_motion.tracking_quality:.1%})"
            )

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Imported {len(manifest)} compact 3D dictionary motions into {args.output}")


if __name__ == "__main__":
    main()
