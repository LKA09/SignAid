# AI Hub data pipeline

1. Keep downloaded ZIP files in `data/raw/` or any external data volume. ZIP files are ignored by Git.
2. Inspect each archive without full extraction. The inspector detects both standard ZIP and the AI Hub downloader's `.zip`-named 7-Zip format:

   ```powershell
   python scripts/inspect_aihub_dataset.py --zip "data/raw/수어스크립트_TL.zip"
   ```

3. For the current script archive, the parser reads XLSX `Information` blocks, Korean sentences, and `sign_gestures_both/strong/weak`, then orders glosses by their `start(s)` time. Review the printed sample before filtering another delivery version.
4. Create the emergency-only index directly from one or more archives:

   ```powershell
   python scripts/filter_emergency_dataset.py --zip "data/raw/수어스크립트_TL.zip"
   ```

5. Join selected sample IDs with keypoint archive members after that download is complete. Standard ZIP members stream directly; solid 7-Zip archives extract only configured category workbooks to a temporary directory. Use `save_standardized_npz` for selected arrays; do not extract complete archives by default.
6. Validate output keys: `pose`, `left_hand`, `right_hand`, `face`, `mask`, `length`, `label`, `signer_id`.
7. Split by signer (never random frames) to avoid identity leakage, then train a baseline or the KSL ST-GCN adapter.

## Importing real dictionary motions

Run the selective importer after the filtered index and the morphology/non-manual JSON archive are present:

```powershell
python scripts/import_aihub_dictionary_motions.py
```

The importer uses each JSON record's aligned gloss spans, extracts at most five candidates per supported concept from the solid 7-Zip archive, rejects malformed tracking records, repairs triangulation spikes, and converts 3D body and hand landmarks to SignAid's 59-joint layout. Palm normals, relative head rotation, mouth opening, eye blinks, and eyebrow movement are stored alongside the compact motion in `data/processed/aihub_motions/*.npz`. `MotionRetriever` prefers these clips over procedural demos.

Only selected records are extracted temporarily. The current 21 3D clips occupy under 0.5MB after conversion, while the 96GB source archive remains untouched and is excluded from submissions. One legacy pregnancy clip remains 2D because the available matching JSON records do not contain valid 3D landmark arrays.

`annotation_aligned=true` means that the clip boundaries came from AI Hub annotations. `expert_validated=false` remains set until a KSL expert reviews the avatar output; annotation matching must not be presented as linguistic certification.

The disaster benchmark gloss ordering helper is reused when its environment imports successfully. Otherwise, the adapter reads common gloss fields defensively and reports detected schema so any adjustment is evidence-driven.
