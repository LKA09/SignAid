# Third-party licenses

## Three.js

- Version: 0.185.1
- License: MIT
- Integration: the minified ES module is vendored in `app/web/src/vendor/` for the interactive WebGL avatar.

## VRoid sample avatar (`AvatarSample_C.vrm`)

- Repository: `third_party/vrm-samples` (`madjin/vrm-samples`), commit `e16eb187100149a315ad92c3c9968f1d5baa6c7d`.
- Upstream conditions: the model is not CC0, but pixiv permits anyone to use it for profit or non-profit work, create images/videos, modify it, and redistribute modified versions subject to the listed prohibited conduct. Attribution is not required.
- Embedded VRM metadata: use by everyone and commercial use are allowed; its legacy `licenseName` field says `Other`.
- Integration: the model remains in `third_party/` and is served read-only by FastAPI. SignAid does not duplicate the model file.
- Limitation: do not sell the original sample `.vroid`/VRM data, present it as CC0, use its data to build a character-creation service, imply pixiv endorsement, or use it for prohibited unlawful/discriminatory/extremist conduct. Upstream terms may change, so recheck them before release.

This inventory is based on the files present in `../third_party` on 2026-08-10. It is not legal advice.

## Korean Disaster Safety Information Sign Language Translation Benchmark Dataset

- Repository: `third_party/Korean-Disaster-Safety-Information-Sign-Language-Translation-Benchmark-Dataset-main`
- License: **No LICENSE/COPYING file found.** The README contains citations but no explicit software grant.
- Integration: `LanguageProcessor._create_gloss_sequence` is called at runtime, when its optional dependencies import successfully. No source was copied.
- Other use: its JSON schema and MediaPipe output conventions informed compatibility tests only.
- Limitation: absent an explicit license, do not redistribute, copy, or commercially ship its source. Obtain permission/clarification from the authors before release.

## KSL-main

- Repository: `third_party/KSL-main`
- License: no standalone license file found. The README states “License CC BY 4.0,” but does not clearly delimit whether that statement covers all code and upstream ST-GCN-derived portions.
- Integration: `Graph.py` and `Models.py` are imported directly at runtime by `signaid/models/ksl_stgcn_adapter.py`; no source was copied.
- Reused components: 47-joint `mediapipe_KSL` graph, normalized adjacency, ST-GCN blocks, temporal convolutions, edge-importance weighting, and optional two-stream model.
- Limitation: preserve attribution and cited papers. Confirm code and upstream-license coverage before distribution or commercial use.

## SignAvatars

- Repository: `third_party/SignAvatars-main`
- Top-level license: **No top-level LICENSE file found.** Individual bundled components have different terms.
- Integration: `vis.py` may be executed in an isolated legacy Python/CUDA environment for native 182-parameter SMPL-X `.pkl` inputs. SignAid does not copy its renderer.
- Dependency limitations: CUDA, PyTorch, pyrender/EGL, SMPL-X model assets, and several legacy pins. Its native input is not a joint-only NPZ.

### SignAvatars visualizer / AITViewer

- File: `third_party/SignAvatars-main/visualizer/LICENSE`
- License: GNU GPL v3.
- Consequence: distribution of a combined/derivative work can trigger GPL source and redistribution obligations. SignAid keeps it behind a subprocess boundary.

### Bundled SMPL-X

- File: `third_party/SignAvatars-main/common/utils/smplx/LICENSE`
- License: Max Planck Institute software license for **non-commercial scientific research, education, or artistic projects only**.
- Restrictions: commercial use is prohibited without a separate license; redistribution and transfer are restricted; citation is required for publications.
- Consequence: the optional avatar mode is not approved for commercial deployment. The default SignAid skeleton renderer has no SMPL-X dependency.

Model files and checkpoints may carry terms separate from source code. Review their licenses before adding them to `checkpoints/` or a container image.
