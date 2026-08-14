# Third-party reuse analysis

## AI Hub structured-data processing

Component: JSON/gloss parsing  
Existing implementation: disaster benchmark `src/language_processor.py`, especially `_create_gloss_sequence`, orders `sign_gestures_both`, `sign_gestures_strong`, and `sign_gestures_weak` by start time.  
Reuse: `signaid.datasets.aihub._third_party_gloss` imports and calls that exact method when its optional OpenCV/MoviePy dependencies are available.  
Compatibility boundary: the original class parses CLI arguments during construction, so the adapter creates an uninitialized instance only for its pure gloss helper.

Component: ZIP and XML inspection  
Existing implementation: the repository reads extracted JSON paths from one fixed AI Hub directory tree. No ZIP streaming or XML parser exists under `src/`.  
Why it cannot be reused: the SignAid download is still compressed, file naming/layout may differ, and extracting the whole dataset violates the MVP requirement.  
Replacement approach: defensive standard-library `zipfile`, `json`, and `ElementTree` streaming in `signaid/datasets/aihub.py`, with no third-party source copied.

Component: keypoint extraction/standardization  
Existing implementation: `KeypointExtractor` runs MediaPipe Holistic over extracted video frames and produces a transposed array containing hands/body.  
Reuse decision: its output convention is accepted and standardized. Direct extraction is not run in the API because it requires extracted frames, MediaPipe, OpenCV, and its CLI-configured directory layout. Future batch jobs can invoke `SignProcessor` in a data-preparation environment.  
Replacement approach: SignAid only reshapes/splits arrays and never duplicates MediaPipe inference.

## KSL recognition

Component: graph adjacency and ST-GCN  
Existing implementation: `KSL-main/Graph.py` and `Models.py` provide the 47-joint MediaPipe graph, graph convolution, temporal convolution, channel attention, and two-stream network.  
Reuse: direct runtime import through `KSLSTGCNAdapter`; tensor conversion changes `(N,T,V,C)` into `(N,C,T,V)` and selects the original 47 joints from 75 pose/hand joints.  
Changes: none inside the third-party repository. GRU and Transformer are separate baselines, not replacements for ST-GCN.

## Rendering

Component: SMPL-X mesh rendering  
Existing implementation: `SignAvatars-main/vis.py`, `common/utils/human_models.py`, bundled SMPL-X, pyrender, and AITViewer.  
Reuse: optional subprocess execution of `vis.py` for its native 182-parameter SMPL-X pickle.  
Why it is isolated: the script assumes CUDA, EGL, legacy dependencies, model assets, repository-specific metadata, and non-commercial SMPL-X terms. Importing it initializes CUDA and reads datasets immediately.  
Fallback approach: the default matplotlib skeleton renderer accepts `(T,J,3)` joint data. This is a distinct lightweight MVP requirement, not a rewrite of SMPL-X mesh generation.

Component: motion viewer  
Existing implementation: AITViewer offers full interactive SMPL/mesh scenes under GPLv3.  
Why it is not embedded: browser delivery, headless server rendering, GPL obligations, and joint-only inputs make direct embedding impractical. The adapter remains available for a separately licensed environment.

Component: browser VRM motion smoothing  
Existing implementation: SignAvatars stores pre-smoothed SMPL-X axis-angle parameters, while its AITViewer dependency offers Python/Scipy spline helpers for SMPL motion.  
Why it cannot be reused: the browser receives joint positions rather than SMPL-X rotations, and neither implementation runs in the JavaScript VRM renderer.  
Replacement approach: the frontend applies a small centered temporal filter, eased fractional-frame interpolation, a loop-transition bridge, and frame-rate-independent quaternion damping with per-joint speed limits. Lower-body and torso retargeting are intentionally restrained so signing motion remains stable.
