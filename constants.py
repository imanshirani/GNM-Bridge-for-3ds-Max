VERSION = "0.0.1"
TOOL_NAME = "GNM Head Generator"
AUTHOR = "Iman Shirani"

GNM_MODEL_VERSION = "V3"
GNM_MESH_SCALE = 100.0   # GNM units are ~meters, Max works in cm — scale up
GNM_ZIP_URL = "https://github.com/google/GNM/archive/refs/heads/main.zip"
GNM_VENDOR_DIR = "vendor"
CONFIG_FILE = "config.json"
PRESETS_DIR = "presets"

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
