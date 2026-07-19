VERSION = "0.0.3"
TOOL_NAME = "GNM Head Generator"
AUTHOR = "Iman Shirani"

GNM_MODEL_VERSION = "V3"
GNM_MESH_SCALE = 100.0   # GNM units are ~meters, Max works in cm — scale up
GNM_ZIP_URL = "https://github.com/google/GNM/archive/refs/heads/main.zip"
GNM_VENDOR_DIR = "vendor"
CONFIG_FILE = "config.json"
PRESETS_DIR = "presets"

# Rhubarb Lip Sync binary
RHUBARB_DOWNLOAD_URL = (
    "https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/"
    "v1.14.0/Rhubarb-Lip-Sync-1.14.0-Windows.zip"
)

# Wav2Vec2 offline lip sync
WAV2VEC2_MODEL = "facebook/wav2vec2-base-960h"
WAV2VEC2_CACHE_DIR = "vendor/wav2vec2"

# Preston Blair phoneme → GNM expression dims 200-349 (150 floats per entry).
# Fallback used when expression decoder (h5py) is unavailable.
# Analysis of expression_basis (coeff=3 effect):
#   idx 0  (dim200): jaw closer (-0.91cm) — NEGATIVE opens
#   idx 1  (dim201): jaw opener (+0.37cm)
#   idx 2  (dim202): lip spread horizontal (+0.03cm)
#   idx 3  (dim203): jaw closer strongest (-1.00cm) — NEGATIVE opens
#   idx 4  (dim204): jaw closer (-0.50cm) — NEGATIVE opens
#   idx 5  (dim205): lip corners / cheek
#   idx 6  (dim206): jaw opener (+0.36cm)
#   idx 7  (dim207): upper lip raise
#   idx 8  (dim208): lower lip drop
#   idx 9  (dim209): lip pucker / protrusion
#   idx 10 (dim210): lip tightening
#   idx 11 (dim211): nasolabial / smile crease
#   idx 12 (dim212): mouth corner pull
#   idx 13 (dim213): chin raise
#   idx 14 (dim214): lip roll (inward)
#   idx 15 (dim215): asymmetric mouth
def _ph(**kw):
    v = [0.0] * 150
    for k, val in kw.items():
        v[int(k)] = val
    return v

PHONEME_EXPR = {
    # X: rest — lips lightly closed
    "X": _ph(**{"0": 0.2, "3": 0.2, "10": 0.2}),
    # A: "ah" — jaw open + slight spread
    "A": _ph(**{"0": -1.8, "1": 2.0, "3": -1.5, "4": -1.0, "6": 2.0,
               "5": 0.5, "7": 0.7, "8": 1.0, "11": 0.3, "12": 0.5}),
    # B: m/b/p — lips firmly pressed
    "B": _ph(**{"0": 1.0, "3": 2.0, "4": 0.8, "10": 1.5, "14": 1.0,
               "7": -0.3, "8": -0.3}),
    # C: "ee" — spread wide + slight open
    "C": _ph(**{"0": -0.5, "1": 1.2, "2": 2.0, "6": 1.2,
               "5": 1.2, "11": 0.8, "12": 1.2, "7": 0.3}),
    # D: "eh" — moderate open + slight spread
    "D": _ph(**{"0": -1.2, "1": 1.5, "3": -0.6, "6": 1.5,
               "2": 0.6, "5": 0.4, "7": 0.4, "8": 0.6, "12": 0.4}),
    # E: "oh" — rounded + moderate open
    "E": _ph(**{"0": -1.0, "1": 1.5, "2": -1.2, "6": 1.5,
               "9": 2.0, "10": 0.8, "5": -0.6, "12": -0.4}),
    # F: f/v — slight open + lip tuck
    "F": _ph(**{"1": 0.6, "6": 0.6, "3": 0.4, "7": 0.8,
               "8": -0.8, "14": 0.8, "10": 0.4}),
    # G: "th" — slight open + tongue
    "G": _ph(**{"0": -0.6, "1": 1.2, "6": 1.2, "7": 0.4, "8": 0.6, "2": 0.2}),
    # H: l/n/d — slight open + tongue up
    "H": _ph(**{"0": -0.4, "1": 1.0, "6": 1.0, "7": 0.2, "8": 0.4, "2": 0.2}),
}

del _ph

INSTALL_INFO = {
    "download_mb": "~15",
    "disk_mb": "~200",
    "time_min": "2-5",
}

# UI Colors
COLOR_BG_DARK = "#1e1e1e"
COLOR_BG_MID = "#2a2a2a"
COLOR_BG_LIGHT = "#333333"
COLOR_ACCENT = "#00AAFF"
COLOR_TEXT = "#dddddd"
COLOR_TEXT_DIM = "#888888"
COLOR_LOG_INFO = "#00FF00"
COLOR_LOG_WARN = "#FFA500"
COLOR_LOG_ERROR = "#FF4444"

STYLESHEET = f"""
QWidget {{
    background-color: {COLOR_BG_DARK};
    color: {COLOR_TEXT};
    font-family: Segoe UI;
    font-size: 12px;
}}
QGroupBox {{
    border: 1px solid {COLOR_BG_LIGHT};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 6px;
}}
QGroupBox::title {{
    color: {COLOR_ACCENT};
    subcontrol-origin: margin;
    left: 8px;
}}
QPushButton {{
    background-color: {COLOR_BG_LIGHT};
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 5px 12px;
    color: {COLOR_TEXT};
}}
QPushButton:hover {{
    background-color: #444444;
    border-color: {COLOR_ACCENT};
}}
QPushButton:pressed {{
    background-color: #222222;
}}
QPushButton#btn_primary {{
    background-color: #005588;
    border-color: {COLOR_ACCENT};
    font-weight: bold;
}}
QPushButton#btn_primary:hover {{
    background-color: #0066aa;
}}
QLineEdit {{
    background-color: {COLOR_BG_MID};
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 3px 6px;
    color: {COLOR_TEXT};
}}
QTextEdit {{
    background-color: #111111;
    border: 1px solid #444444;
    border-radius: 3px;
    font-family: Consolas, monospace;
    font-size: 11px;
}}
QLabel {{
    color: {COLOR_TEXT_DIM};
}}
QLabel#lbl_title {{
    color: {COLOR_ACCENT};
    font-size: 14px;
    font-weight: bold;
}}
"""
