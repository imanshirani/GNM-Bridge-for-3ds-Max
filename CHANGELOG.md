# Changelog

All notable changes to GNM Head Generator for 3ds Max are documented here.

---
## [0.0.3] — 2026-07-19

### Added

#### Wav2Vec2 Offline AI Lip Sync
- New **Wav2Vec2 Lip Sync (Offline AI)** button in the Animation tab
- Powered by [Facebook Wav2Vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) — no API key, no internet after first run
- Runs as an isolated subprocess so 3ds Max never imports torch directly (avoids DLL hang)
- WAV files are automatically resampled to 16kHz mono before inference
- Character-level CTC output mapped to 9 Preston Blair phoneme codes → GNM expression keyframes
- **Strength** spinner (0.1 – 3.0) to scale phoneme expression intensity
- Model (~378 MB) and PyTorch CPU (~200 MB) downloaded once, cached in `vendor/wav2vec2/`

#### Lip Sync from Audio (Rhubarb)
- New **Lip Sync from Audio** section in the Animation tab
- Powered by [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) (~8 MB, downloaded automatically)
- Browse a WAV file, set FPS, start frame, and blend frames (cross-fade between phonemes)
- **Generate Lip Sync** — runs Rhubarb in a background worker, converts 9 Preston Blair phoneme codes to GNM expression keyframes
- **Clear Lip Sync** — removes only lip-sync keyframes, leaving manual keyframes intact
- Phoneme → GNM expression mapping updated with richer multi-dimensional values (dims 200–215)
- Expression decoder used for calibration when h5py is available (150-dim lower-face slice)
- Sparse-detection: falls back to decoder output if calibration JSON has fewer than 15 non-zero dims per phoneme


## [0.0.2] — 2025-07-17

### Added

#### Animation tab
- New **Animation** tab with GNM keyframe storage and linear interpolation
- **Arm/Disarm** toggle — live mesh update on Max timeline scrub via `registerTimeCallback`
- **Add Keyframe** — snapshot current identity, expression, and rotation sliders at any frame
- **Go To Frame** — jump Max timeline to the selected keyframe
- **Delete / Clear All** keyframe controls
- **Bake to Timeline** — generates a Morpher modifier with real Max timeline keys for each GNM keyframe (one morph channel per keyframe, weights keyed to 0/100/0)
- **Save / Load JSON** — persist and restore keyframe lists to `GNM/animation/`

#### Presets gallery
- Rebuilt Presets tab as a thumbnail gallery grid (3 per row, 128 × 128 px square thumbnails)
- Viewport thumbnails captured automatically on preset save using `gw.getViewportDib()`
- Category headers with collapse/expand and right-click rename
- Double-click to load, right-click context menu for Load / Rename / Set Category / Delete

#### Gender slider
- Female ↔ Male blend slider in the Shape tab
- Pure numpy + h5py inference of the GNM identity decoder — no TensorFlow required
- `get_gender_vector()` samples zero-latent male and female averages from `identity_decoder_model.h5`

#### About dialog
- GitHub repository link (opens browser)
- Google GNM project link
- PayPal donate button
- Accessible from the ⚙ settings menu

#### Settings menu
- **Reveal Install Path** — opens the GNM plugin folder in Windows Explorer

### Fixed

- **DirtyNotificationEventMonitor error** (`call depth count < 0`) on Max 2027
  - Removed `redrawViews()` from all Max notification callbacks (`update_max_mesh_vertices`, `interpolate_and_update`, time callback)
  - All viewport redraws now deferred via `QtCore.QTimer.singleShot(0, ...)`
- **Randomize not working on first click** — race condition between debounce timer and mesh worker resolved by reordering worker setup (connect signals → assign → start)
- **UV mapping failure** causing "Attempt to access deleted scene object" — UV assignment moved outside the undo block and made non-fatal; mesh is returned even if UV fails
- **MAXScript `local` at top level** compile error in generated MAXScript strings — rewrote UV mapping using the pymxs Python API directly (`meshop.setMapVert` / `meshop.setMapFace`)
- **Max 2027 numpy missing on first launch** — module-level `import numpy` replaced with `_NUMPY_OK` flag; setup page shown when numpy is absent instead of crashing

### Changed

- `update_max_mesh_vertices` and `create_max_mesh` no longer call `redrawViews()` — callers are responsible for scheduling a deferred redraw
- `_apply_uvw` rewritten with pymxs Python API (no MAXScript string execution)
- `interpolate_and_update` no longer calls `redrawViews()` directly

---

## [0.0.1] — 2025 2025-07-16

### Added

- Shape tab with 15 identity sliders
- Expression tab with 78 sliders in 5 collapsible anatomical groups
- Pose controls (head rotation, neck, eyes)
- Population tab — batch-generate a grid of random heads
- Live mesh update with 400 ms debounce on slider change
- UV texture coordinates (channel 1) on all generated meshes
- Z-up coordinate system, scale GNM metres → 3ds Max centimetres (×100)
- "Use Selected" — load identity from an existing GNM mesh
- First-run setup wizard: downloads GNM from GitHub, installs Python dependencies
- Max 2025 / 2026 / 2027 compatibility
  - Auto-selects `numpy < 2` for Python 3.11/3.12 (Max 2025/2026)
  - Auto-selects `numpy >= 2` for Python 3.13 (Max 2027)
  - `ensurepip` bootstrap for Max 2027 (pip not pre-installed)
