# GNM Head Generator for 3ds Max

[![Donate ❤️](https://img.shields.io/badge/Donate-PayPal-00457C?style=flat-square&logo=paypal&logoColor=white)](https://www.paypal.com/donate/?hosted_button_id=LAMNRY6DDWDC4)
![3dsmax](https://img.shields.io/badge/Autodesk-3ds%20Max-0696D7?style=flat-square&logo=autodesk)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-41CD52?style=flat-square&logo=qt&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

A parametric 3D human head generator plugin for **3ds Max 2025 / 2026 / 2027**, powered by [Google's GNM (Generative Neural Mesh)](https://github.com/google/GNM) model.

Generate, randomize, and animate realistic human head meshes directly in 3ds Max using identity sliders, expression controls, pose controls, and a gender slider — all in real time.

---

## Screenshots

> *(Add screenshots here)*

---

## Features

| Tab | What it does |
|-----|-------------|
| **Shape** | 15 identity sliders (face width, jaw, brow, cheekbones, nose, etc.) + Gender slider (Female ↔ Male) |
| **Expression** | 78 expression sliders in 5 anatomical groups: Lower Face / Mouth / Jaw, Left Eye, Right Eye, Tongue, Pupils |
| **Expression — Pose** | 8 pose controls: Head nod/turn/tilt, Neck, Left eye up/down/left/right, Right eye up/down/left/right |
| **Presets** | Save/load presets with categories and viewport thumbnail previews. Double-click to load, right-click for options |
| **Population** | Batch-generate a grid of random heads with configurable count, columns, spacing, and random seed |

**Additional features:**
- Live mesh update with 400 ms debounce — move a slider, the mesh updates automatically
- UV texture coordinates (channel 1) applied to all generated meshes
- Z-up coordinate system (native 3ds Max convention)
- Scale: GNM metres → 3ds Max centimetres (×100)
- "Use Selected" — pick an existing GNM mesh in the viewport and edit it with the sliders
- Preset viewport thumbnails captured automatically on save
- All UI in English

---

## Requirements

- **3ds Max 2025, 2026, or 2027**
- Internet connection (first run only, to download GNM ~15 MB)
- ~200 MB disk space

---

## Installation

### Copy the plugin

Place the `GNM` folder anywhere on your machine. Example:

```
C:\3ds max python\GNM\
```


> **Tip:** Save this snippet as a MAXScript macro or toolbar button so you can open the tool with one click on every session.

### 3. First-time setup

The **Setup** page appears on first run. Click **Install & Setup GNM** to:

1. Download the GNM repository from GitHub (~15 MB)
2. Extract it to `GNM/vendor/`
3. Install all required Python packages into 3ds Max's Python
4. Register GNM on `sys.path`

This takes **2–5 minutes** depending on your internet speed. After completion the tool switches to the ready state automatically.

---

## Compatibility

| 3ds Max | Python | NumPy installed |
|---------|--------|-----------------|
| 2025    | 3.11   | `numpy < 2` (1.x) |
| 2026    | 3.12   | `numpy < 2` (1.x) |
| 2027    | 3.13   | `numpy >= 2` (2.x) |

The installer detects your Python version and installs the correct NumPy version automatically.

---

## Usage

### Shape tab

Move the **identity sliders** to sculpt the head shape. Each slider corresponds to a principal component of GNM's identity space.

- **Reset All** — return all identity sliders to zero
- **Randomize** — fill with random identity values
- **Gender slider** — blend between a statistically average female and male face (requires h5py, installed automatically)
- **Create New Mesh** — generate a new mesh in the scene with the current settings
- **Use Selected** — select an existing GNM mesh in the viewport, then click to load its identity into the sliders

### Expression tab

Five collapsible groups of expression sliders control facial expressions. **Lower Face** is open by default; the other groups start collapsed.

Pose controls at the top let you rotate the head, neck, and eyes.

### Presets tab

1. Set a **Name** and optional **Category**
2. Click **Save Preset** — the current slider values and a viewport thumbnail are saved
3. Browse presets in the gallery (3 per row)
4. **Double-click** a preset to load it
5. **Right-click** for: Load / Rename / Set Category / Delete

Category headers can be **renamed** by right-clicking the category name.

### Population tab

Generate a grid of random heads in one click:

| Setting | Description |
|---------|-------------|
| Count | Total number of heads |
| Columns | Heads per row |
| Spacing X / Y | Distance between heads (cm) |
| Seed | Random seed for reproducibility |

---

## File Structure

```
GNM/
├── launch.py           ← Entry point — run this from Max
├── core.py             ← Main UI (QDockWidget, all tabs)
├── gnm_bridge.py       ← GNM model interface, mesh creation, UV mapping
├── setup_manager.py    ← Download, extract, install dependencies
├── constants.py        ← Version, colours, stylesheet
├── utils.py            ← QLogger helper
├── __init__.py
├── slider_names.json   ← Persisted custom slider names (auto-created)
├── config.json         ← Installation state (auto-created)
├── presets/            ← Saved presets as JSON + PNG thumbnail pairs
│   ├── MyPreset.json
│   └── MyPreset.png
└── vendor/             ← Downloaded GNM repository (auto-created)
    └── GNM-main/
```

---

## Resetting / Reinstalling

Click **⚙ → Reinstall** in the tool, or manually delete:

```
GNM/vendor/       ← downloaded GNM repo and model data
GNM/config.json   ← installation state
```

Then reopen the tool — the Setup page will appear again.

> **Note:** If you installed on 3ds Max 2026 first, reinstalling on 2027 will skip the download and only install packages — much faster.

---

## Known Issues & Troubleshooting

### NumPy version conflict on 3ds Max 2025

**Symptom:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**Cause:** Max 2025's compiled extensions (USD Tools, MAXtoA) require NumPy 1.x. If another tool installed NumPy 2.x, this conflict appears.

**Fix — run once in the Max 2025 Python Listener:**
```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "--force-reinstall", "--user", "numpy<2"])
```
Then **close and reopen 3ds Max 2025**.

**Prevention:** When installing additional pip packages in Max 2025, always pin NumPy:
```python
subprocess.run([sys.executable, "-m", "pip", "install",
                "--user", "your-package", "numpy<2"])
```

---

### Setup stuck at 0% on 3ds Max 2027 (first run)

**Cause:** Max 2027's Python distribution does not include `pip`. The installer bootstraps it automatically via `ensurepip`, but this can take **30–60 seconds** with no visible progress.

**What to do:** Wait. You will see `pip not found — bootstrapping...` in the log, followed by `pip ready.` when it completes.

**If it still fails after 60 seconds:**

1. Close the GNM tool window (do **not** close 3ds Max itself)
2. Rerun the launch script in the MAXScript Listener Python tab:
   ```python
   import GNM.launch
   GNM.launch.launch()
   ```
3. Click **Install & Setup GNM** again — pip is now bootstrapped, and the install will proceed normally


---

### DirtyNotificationEventMonitor error

```
-- Error: DirtyNotificationEventMonitor call depth count < 0
```

**Cause:** Harmless Max internal warning that occasionally appears when a Qt dock widget is closed and reopened quickly. The tool continues to work normally. If it appears repeatedly, close and rerun `GNM.launch.launch()`.

---



---

## Dependencies

All installed automatically:

| Package | Purpose |
|---------|---------|
| `numpy` | Array math (1.x for Max 2025/2026, 2.x for Max 2027) |
| `h5py` | Read gender decoder model without TensorFlow |
| `scipy` | Scientific computing (GNM dependency) |
| `trimesh` | Mesh utilities (GNM dependency) |
| `absl-py` | Google utilities (GNM dependency) |
| `etils` | Google utilities (GNM dependency) |
| `immutabledict` | Immutable dicts (GNM dependency) |
| `importlib_resources` | Resource loading (GNM dependency) |
| `opt-einsum` | Optimised einsum (GNM dependency) |
| `rtree` | Spatial indexing (GNM dependency) |
| `tqdm` | Progress bars (GNM dependency) |
| `typeguard` | Runtime type checking (GNM dependency) |
| `opencv-python` | Image processing (GNM dependency) |

---

## Credits

- **GNM model:** [Google LLC](https://github.com/google/GNM) — Apache 2.0 License
- **Plugin:** Iman Shirani

> **Note on the GNM model:** The GNM was trained on datasets using binary gender categories and four broad demographic groups (Middle Eastern, Asian, White, Black). It does not represent all gender identities or the full diversity of the global population. See the [GNM README](https://github.com/google/GNM) for details.

---

## License

This plugin is provided as-is for personal and professional use. The underlying GNM model is subject to Google's [Apache 2.0 License](https://github.com/google/GNM/blob/main/LICENSE).
