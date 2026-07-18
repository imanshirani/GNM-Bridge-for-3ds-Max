import json
import math
from pathlib import Path

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    np = None
    _NUMPY_OK = False

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Signal

from . import constants
from . import gnm_bridge
from . import setup_manager
from .utils import QLogger

_NUM_IDENTITY_SLIDERS = 15
_SLIDER_SCALE = 10       # slider integer / scale = float value  (-30..+30 -> -3.0..+3.0)
_DEBOUNCE_MS = 400

_SLIDER_STYLE = (
    "QSlider::groove:horizontal { background:#333; height:4px; border-radius:2px; }"
    "QSlider::handle:horizontal { background:#00AAFF; width:12px; height:12px; "
    "margin:-4px 0; border-radius:6px; }"
    "QSlider::sub-page:horizontal { background:#005588; border-radius:2px; }"
)

_PRESETS_DIR = Path(__file__).parent / constants.PRESETS_DIR
_SLIDER_NAMES_FILE = Path(__file__).parent / "slider_names.json"

_IDENTITY_NAMES_DEFAULT = [
    "Face Width", "Jaw Shape", "Brow Ridge", "Cheekbones", "Nose Bridge",
    "Eye Depth", "Forehead", "Temple Width", "Chin Shape", "Face Height",
    "Lip Area", "Nasal Shape", "Orbital Depth", "Face Roundness", "Skull Shape",
]
# GNM HEAD expression vector regions (total 383 dims):
#   0–99   left_eye_region
#   100–199 right_eye_region
#   200–349 lower_face_region (mouth, jaw, lips, cheeks)
#   350     tongue_mean
#   351–381 tongue fine control
#   382     pupils
_EXPRESSION_REGIONS = [
    # (label,                    start_dim, count, prefix)
    ("Lower Face — Mouth & Jaw", 200,       25,    "LF"),
    ("Left Eye",                   0,       20,    "LE"),
    ("Right Eye",                100,       20,    "RE"),
    ("Tongue",                   350,       12,    "TG"),
    ("Pupils / Iris",            382,        1,    "PU"),
]


def _load_slider_names() -> dict:
    if _SLIDER_NAMES_FILE.exists():
        try:
            return json.loads(_SLIDER_NAMES_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_slider_names(data: dict):
    try:
        _SLIDER_NAMES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _set_val_label(lbl: "QtWidgets.QLabel", v: float):
    base = "font-size:12px; font-family:Consolas; font-weight:bold;"
    if abs(v) < 0.05:
        lbl.setText("0.0")
        lbl.setStyleSheet(f"color:#555; {base}")
    elif v > 0:
        lbl.setText(f"+{v:.1f}")
        lbl.setStyleSheet(f"color:#00AAFF; {base}")
    else:
        lbl.setText(f"{v:.1f}")
        lbl.setStyleSheet(f"color:#FF5555; {base}")


# ─── Background workers ──────────────────────────────────────────────────────

class SetupWorker(QtCore.QThread):
    sig_progress = Signal(int, str)
    sig_log = Signal(str)
    sig_done = Signal(bool, str)

    def run(self):
        import traceback
        try:
            success, result = setup_manager.run_full_setup(
                progress_cb=lambda p, l: self.sig_progress.emit(p, l),
                log_cb=lambda m: self.sig_log.emit(m),
            )
        except Exception:
            success, result = False, traceback.format_exc()
        self.sig_done.emit(success, result)


class GenerateWorker(QtCore.QThread):
    sig_vertices = Signal(object)   # numpy [N,3] or None

    def __init__(self, identity, expression, rotations, parent=None):
        super().__init__(parent)
        self._identity = identity.copy()
        self._expression = expression.copy()
        self._rotations = rotations.copy()

    def run(self):
        vertices, _tris, _uvs = gnm_bridge.generate_head(
            identity=self._identity,
            expression=self._expression,
            rotations=self._rotations,
        )
        self.sig_vertices.emit(vertices)


class BatchWorker(QtCore.QThread):
    sig_head = Signal(object, object, object, str, tuple, object)  # verts, tris, uvs, name, pos, identity
    sig_progress = Signal(int, int)
    sig_done = Signal()

    def __init__(self, count, cols, spacing_x, spacing_y, seed, parent=None):
        super().__init__(parent)
        self._count = count
        self._cols = cols
        self._spacing_x = spacing_x
        self._spacing_y = spacing_y
        self._seed = seed

    def run(self):
        import numpy as _np
        rng = _np.random.default_rng(self._seed)
        for i in range(self._count):
            identity = rng.standard_normal(253).astype(_np.float32)
            vertices, triangles, triangle_uvs = gnm_bridge.generate_head(identity=identity)
            if vertices is not None:
                col = i % self._cols
                row = i // self._cols
                pos = (col * self._spacing_x, row * self._spacing_y, 0.0)
                self.sig_head.emit(vertices, triangles, triangle_uvs, f"GNM_Head_{i+1:02d}", pos, identity)
            self.sig_progress.emit(i + 1, self._count)
        self.sig_done.emit()


# ─── Main Widget ─────────────────────────────────────────────────────────────

class GNMTool(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(constants.TOOL_NAME)
        self.setMinimumWidth(380)
        self.setStyleSheet(constants.STYLESHEET)

        self.logger = QLogger(self)
        self.logger.sig_log.connect(self._append_log)

        # State that _build_ui / _build_expression_tab etc. reference must exist
        # before any widget is built, even when numpy is not yet available.
        if _NUMPY_OK:
            self._identity = np.zeros(253, dtype=np.float32)
            self._expression = np.zeros(383, dtype=np.float32)
            self._rotations = np.zeros((4, 3), dtype=np.float32)
        else:
            self._identity = None
            self._expression = None
            self._rotations = None
        self._current_mesh = None
        self._gen_worker = None
        self._batch_worker = None
        self._pending = False
        self._setup_worker = None
        self._expr_region_data = []
        self._gender_strength = 0.0   # -1.0 = full female, +1.0 = full male
        self._anim_keyframes = []     # list of dicts sorted by frame
        self._anim_armed = False      # True when time callback is registered

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._trigger_update)

        self._lbl_update_status = QtWidgets.QLabel("")  # created here so timer always works
        self._status_clear = QtCore.QTimer(self)
        self._status_clear.setSingleShot(True)
        self._status_clear.setInterval(2000)
        self._status_clear.timeout.connect(lambda: self._lbl_update_status.setText(""))

        self._build_ui()

        if not _NUMPY_OK:
            self._show_setup_state()
            QtCore.QTimer.singleShot(0, lambda: self.logger.warning(
                "numpy not found in this version of 3ds Max. "
                "Click 'Install & Setup GNM' to install all dependencies."
            ))
            return

        is_ready, _ = setup_manager.check_status()
        self._show_ready_state() if is_ready else self._show_setup_state()

    # ─── UI Build ────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._main_layout = QtWidgets.QVBoxLayout(self)
        self._main_layout.setContentsMargins(10, 10, 10, 10)
        self._main_layout.setSpacing(8)

        title_row = QtWidgets.QHBoxLayout()
        lbl_title = QtWidgets.QLabel(constants.TOOL_NAME)
        lbl_title.setObjectName("lbl_title")
        title_row.addWidget(lbl_title)
        title_row.addStretch()
        self._lbl_status_badge = QtWidgets.QLabel("")
        title_row.addWidget(self._lbl_status_badge)
        self._btn_settings = QtWidgets.QPushButton("⚙")
        self._btn_settings.setFixedSize(28, 28)
        self._btn_settings.setToolTip("Settings")
        self._btn_settings.clicked.connect(self._show_settings_menu)
        title_row.addWidget(self._btn_settings)
        self._main_layout.addLayout(title_row)

        self._stack = QtWidgets.QStackedWidget()
        self._main_layout.addWidget(self._stack)

        self._page_setup = self._build_setup_page()
        self._stack.addWidget(self._page_setup)

        # Ready page contains numpy-dependent slider widgets — only build it
        # when numpy is actually available, otherwise a placeholder is used.
        if _NUMPY_OK:
            self._page_ready = self._build_ready_page()
        else:
            self._page_ready = QtWidgets.QWidget()  # empty placeholder
        self._stack.addWidget(self._page_ready)

        self._log_view = QtWidgets.QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMinimumHeight(80)
        self._log_view.setMaximumHeight(120)
        self._main_layout.addWidget(self._log_view)

        bottom_row = QtWidgets.QHBoxLayout()
        lbl_ver = QtWidgets.QLabel(f"v{constants.VERSION}  |  {constants.AUTHOR}")
        btn_clear = QtWidgets.QPushButton("Clear Log")
        btn_clear.setFixedHeight(22)
        btn_clear.clicked.connect(self._log_view.clear)
        bottom_row.addWidget(lbl_ver)
        bottom_row.addStretch()
        bottom_row.addWidget(btn_clear)
        self._main_layout.addLayout(bottom_row)

    # ─── Setup page ──────────────────────────────────────────────────────────

    def _build_setup_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        info_box = QtWidgets.QFrame()
        info_box.setStyleSheet(
            "QFrame { background-color: #2a2200; border: 1px solid #665500; border-radius: 4px; }"
        )
        info_layout = QtWidgets.QVBoxLayout(info_box)
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.setSpacing(3)

        lbl_warn = QtWidgets.QLabel("⚠  Please read before installing:")
        lbl_warn.setStyleSheet("color: #FFD700; font-weight: bold;")
        info_layout.addWidget(lbl_warn)

        info = constants.INSTALL_INFO
        for line in [
            f"•  Download size:   {info['download_mb']} MB",
            f"•  Disk space:      {info['disk_mb']} MB",
            f"•  Estimated time:  {info['time_min']} minutes",
            "•  Internet connection required",
            "•  Files are stored next to this script",
        ]:
            lbl = QtWidgets.QLabel(line)
            lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
            info_layout.addWidget(lbl)

        layout.addWidget(info_box)

        self._btn_setup = QtWidgets.QPushButton("Install && Setup GNM")
        self._btn_setup.setObjectName("btn_primary")
        self._btn_setup.setMinimumHeight(34)
        self._btn_setup.clicked.connect(self._on_start_setup)
        layout.addWidget(self._btn_setup)

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar { border:1px solid #555; border-radius:3px; background:#222; height:16px; }"
            "QProgressBar::chunk { background-color:#00AAFF; border-radius:2px; }"
        )
        layout.addWidget(self._progress_bar)

        self._lbl_progress_text = QtWidgets.QLabel("")
        self._lbl_progress_text.setVisible(False)
        self._lbl_progress_text.setStyleSheet("color:#aaaaaa; font-size:11px;")
        layout.addWidget(self._lbl_progress_text)

        layout.addStretch()
        return page

    # ─── Ready page (tabbed) ─────────────────────────────────────────────────

    def _build_ready_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #444; border-radius: 3px; }"
            "QTabBar::tab { background: #2a2a2a; color: #aaaaaa; padding: 5px 12px; border: 1px solid #444; border-bottom: none; border-radius: 3px 3px 0 0; }"
            "QTabBar::tab:selected { background: #1e1e1e; color: #00AAFF; border-color: #00AAFF; }"
            "QTabBar::tab:hover:!selected { background: #333; color: #dddddd; }"
        )
        tabs.addTab(self._build_shape_tab(), "Shape")
        tabs.addTab(self._build_expression_tab(), "Expression")
        tabs.addTab(self._build_presets_tab(), "Presets")
        tabs.addTab(self._build_population_tab(), "Population")
        tabs.addTab(self._build_animation_tab(), "Animation")
        layout.addWidget(tabs)

        self._lbl_update_status.setStyleSheet("font-size: 11px;")
        layout.addWidget(self._lbl_update_status)

        return page

    # ─── Shape tab ───────────────────────────────────────────────────────────

    def _build_shape_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        top_row = QtWidgets.QHBoxLayout()
        lbl_name = QtWidgets.QLabel("Name:")
        lbl_name.setFixedWidth(38)
        self._edit_name = QtWidgets.QLineEdit("GNM_Head")
        self._btn_new_mesh = QtWidgets.QPushButton("Create New Mesh")
        self._btn_new_mesh.setObjectName("btn_primary")
        self._btn_new_mesh.clicked.connect(self._on_new_mesh)
        top_row.addWidget(lbl_name)
        top_row.addWidget(self._edit_name)
        top_row.addWidget(self._btn_new_mesh)
        layout.addLayout(top_row)

        self._btn_use_selected = QtWidgets.QPushButton("Use Selected")
        self._btn_use_selected.setToolTip("Select a GNM mesh in the viewport, then click to edit it")
        self._btn_use_selected.clicked.connect(self._on_use_selected)
        layout.addWidget(self._btn_use_selected)

        # ── Gender slider ────────────────────────────────────────────────
        gender_grp = QtWidgets.QGroupBox("Gender")
        gender_lay = QtWidgets.QVBoxLayout(gender_grp)
        gender_lay.setContentsMargins(8, 8, 8, 6)
        gender_lay.setSpacing(4)

        gender_row = QtWidgets.QHBoxLayout()
        lbl_f = QtWidgets.QLabel("Female")
        lbl_f.setStyleSheet("color:#FF88BB; font-size:11px;")
        lbl_n = QtWidgets.QLabel("Neutral")
        lbl_n.setStyleSheet("color:#888; font-size:11px;")
        lbl_n.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_m = QtWidgets.QLabel("Male")
        lbl_m.setStyleSheet("color:#88AAFF; font-size:11px;")
        lbl_m.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        gender_row.addWidget(lbl_f)
        gender_row.addWidget(lbl_n, 1)
        gender_row.addWidget(lbl_m)
        gender_lay.addLayout(gender_row)

        self._sld_gender = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._sld_gender.setRange(-100, 100)
        self._sld_gender.setValue(0)
        self._sld_gender.setStyleSheet(
            "QSlider::groove:horizontal { background: qlineargradient("
            "x1:0,y1:0,x2:1,y2:0, stop:0 #FF88BB, stop:0.5 #444, stop:1 #88AAFF);"
            "height:6px; border-radius:3px; }"
            "QSlider::handle:horizontal { background:#ddd; width:14px; height:14px; "
            "margin:-4px 0; border-radius:7px; }"
            "QSlider::sub-page:horizontal { background:transparent; }"
            "QSlider::add-page:horizontal { background:transparent; }"
        )
        self._sld_gender.valueChanged.connect(self._on_gender_slider_changed)
        gender_lay.addWidget(self._sld_gender)
        layout.addWidget(gender_grp)

        grp = QtWidgets.QGroupBox("Identity — Head Shape")
        grp_layout = QtWidgets.QVBoxLayout(grp)
        grp_layout.setSpacing(4)
        grp_layout.setContentsMargins(6, 8, 6, 6)

        btn_row = QtWidgets.QHBoxLayout()
        btn_reset = QtWidgets.QPushButton("Reset All")
        btn_reset.setFixedHeight(24)
        btn_rand = QtWidgets.QPushButton("Randomize")
        btn_rand.setFixedHeight(24)
        btn_reset.clicked.connect(self._on_reset_sliders)
        btn_rand.clicked.connect(self._on_randomize_sliders)
        btn_row.addWidget(btn_reset)
        btn_row.addWidget(btn_rand)
        grp_layout.addLayout(btn_row)

        self._sliders, self._slider_labels, self._slider_name_edits = self._build_slider_list(
            grp_layout, _NUM_IDENTITY_SLIDERS, "IC",
            lambda i, v: self._on_identity_slider_changed(i, v),
            default_names=_IDENTITY_NAMES_DEFAULT,
        )

        layout.addWidget(grp)
        layout.addStretch()
        return tab

    # ─── Expression tab ──────────────────────────────────────────────────────

    def _build_expression_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Pose Controls (jaw / head rotation) ─────────────────────────────
        pose_grp = QtWidgets.QGroupBox("Pose Controls")
        pose_layout = QtWidgets.QVBoxLayout(pose_grp)
        pose_layout.setSpacing(6)
        pose_layout.setContentsMargins(8, 10, 8, 8)

        pose_reset = QtWidgets.QPushButton("Reset Pose")
        pose_reset.setFixedHeight(24)
        pose_reset.clicked.connect(self._on_reset_pose)
        pose_layout.addWidget(pose_reset)

        self._pose_sliders = {}
        # GNM HEAD joints: 0=head, 1=neck, 2=left_eye, 3=right_eye (no jaw joint)
        # axis-angle in radians: axis 0=X (nod/pitch), 1=Y (turn/yaw), 2=Z (tilt/roll)
        pose_defs = [
            ("head_nod",    "Head Nod",      0, 0, -35, 35, 0, "Head nod up/down"),
            ("head_turn",   "Head Turn",     0, 1, -45, 45, 0, "Head left/right"),
            ("head_tilt",   "Head Tilt",     0, 2, -30, 30, 0, "Head side-tilt"),
            ("neck_nod",    "Neck Nod",      1, 0, -20, 20, 0, "Neck forward/back"),
            ("eye_l_ud",    "Eye L Up/Down", 2, 0, -25, 25, 0, "Left eye up/down"),
            ("eye_l_lr",    "Eye L Left/Right", 2, 1, -30, 30, 0, "Left eye left/right"),
            ("eye_r_ud",    "Eye R Up/Down", 3, 0, -25, 25, 0, "Right eye up/down"),
            ("eye_r_lr",    "Eye R Left/Right", 3, 1, -30, 30, 0, "Right eye left/right"),
        ]

        for key, label, rot_idx, axis_idx, lo, hi, default, tip in pose_defs:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(6)

            lbl = QtWidgets.QLabel(label)
            lbl.setFixedWidth(72)
            lbl.setStyleSheet("color:#aaa; font-size:11px;")
            lbl.setToolTip(tip)

            sld = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            sld.setRange(lo, hi)
            sld.setValue(default)
            sld.setFixedHeight(14)
            sld.setStyleSheet(_SLIDER_STYLE)

            val_lbl = QtWidgets.QLabel(f"{default}°")
            val_lbl.setFixedWidth(38)
            val_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet("color:#666; font-size:12px; font-family:Consolas; font-weight:bold;")

            def make_pose_handler(k, ri, ai, lbl_w):
                def on_change(deg):
                    self._rotations[ri, ai] = math.radians(deg)
                    color = "#00AAFF" if deg > 0 else ("#FF5555" if deg < 0 else "#555")
                    lbl_w.setText(f"{deg}°")
                    lbl_w.setStyleSheet(f"color:{color}; font-size:12px; font-family:Consolas; font-weight:bold;")
                    self._debounce.start()
                return on_change

            sld.valueChanged.connect(make_pose_handler(key, rot_idx, axis_idx, val_lbl))
            self._pose_sliders[key] = sld

            row.addWidget(lbl)
            row.addWidget(sld)
            row.addWidget(val_lbl)
            pose_layout.addLayout(row)

        layout.addWidget(pose_grp)

        # ── Expression regions ───────────────────────────────────────────────
        expr_top_row = QtWidgets.QHBoxLayout()
        btn_reset_all = QtWidgets.QPushButton("Reset All")
        btn_reset_all.setFixedHeight(24)
        btn_rand_all = QtWidgets.QPushButton("Randomize")
        btn_rand_all.setFixedHeight(24)
        btn_reset_all.clicked.connect(self._on_reset_expr_sliders)
        btn_rand_all.clicked.connect(self._on_randomize_expr_sliders)
        expr_top_row.addWidget(btn_reset_all)
        expr_top_row.addWidget(btn_rand_all)
        layout.addLayout(expr_top_row)

        # Scrollable container for all region groups
        outer_scroll = QtWidgets.QScrollArea()
        outer_scroll.setWidgetResizable(True)
        outer_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        outer_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        outer_container = QtWidgets.QWidget()
        outer_container.setStyleSheet("background: transparent;")
        outer_layout = QtWidgets.QVBoxLayout(outer_container)
        outer_layout.setSpacing(6)
        outer_layout.setContentsMargins(0, 0, 4, 0)

        self._expr_region_data = []   # [(start_dim, [sliders], [labels])]
        saved_names = _load_slider_names().get("expr_region_names", {})

        for region_label, start_dim, count, prefix in _EXPRESSION_REGIONS:
            # Collapsible group
            grp_frame = QtWidgets.QFrame()
            grp_frame.setStyleSheet(
                "QFrame { border:1px solid #3a3a3a; border-radius:4px; background:#1e1e1e; }"
            )
            grp_v = QtWidgets.QVBoxLayout(grp_frame)
            grp_v.setContentsMargins(0, 0, 0, 0)
            grp_v.setSpacing(0)

            toggle_btn = QtWidgets.QPushButton(f"▼  {region_label}")
            toggle_btn.setCheckable(True)
            toggle_btn.setChecked(True)
            toggle_btn.setStyleSheet(
                "QPushButton { background:#2a2a2a; border:none; border-radius:3px; "
                "color:#00AAFF; font-weight:bold; font-size:11px; padding:5px 8px; text-align:left; }"
                "QPushButton:hover { background:#333; }"
            )
            grp_v.addWidget(toggle_btn)

            content = QtWidgets.QWidget()
            content.setStyleSheet("background: transparent; border: none;")
            content_layout = QtWidgets.QVBoxLayout(content)
            content_layout.setContentsMargins(4, 4, 4, 4)
            content_layout.setSpacing(2)

            sliders = []
            labels = []

            for i in range(count):
                dim_idx = start_dim + i
                name_key = f"{prefix}-{i+1:02d}"
                saved_name = saved_names.get(name_key)

                row_top = QtWidgets.QHBoxLayout()
                row_top.setSpacing(4)

                lbl_idx = QtWidgets.QLabel(f"{prefix}-{i+1:02d}")
                lbl_idx.setFixedWidth(38)
                lbl_idx.setStyleSheet(
                    "color:#555; font-size:10px; font-family:Consolas; "
                    "background:#1a1a1a; border-radius:3px; padding:1px 3px; border:none;"
                )

                edit_name = QtWidgets.QLineEdit(saved_name or f"{prefix}-{i+1:02d}")
                edit_name.setStyleSheet(
                    "QLineEdit { background:#222; border:1px solid #3a3a3a; border-radius:3px; "
                    "color:#aaa; font-size:11px; padding:1px 4px; }"
                    "QLineEdit:focus { border-color:#00AAFF; color:#ddd; }"
                )
                edit_name.setFixedHeight(20)

                lbl_val = QtWidgets.QLabel("0.0")
                lbl_val.setFixedWidth(38)
                lbl_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                lbl_val.setStyleSheet("color:#555; font-size:12px; font-family:Consolas; font-weight:bold; border:none;")

                row_top.addWidget(lbl_idx)
                row_top.addWidget(edit_name)
                row_top.addWidget(lbl_val)

                sld = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                sld.setRange(-30, 30)
                sld.setValue(0)
                sld.setFixedHeight(14)
                sld.setStyleSheet(_SLIDER_STYLE)

                def make_expr_handler(d_idx, lbl_w):
                    def on_change(val):
                        v = val / _SLIDER_SCALE
                        self._expression[d_idx] = v
                        _set_val_label(lbl_w, v)
                        self._debounce.start()
                    return on_change

                sld.valueChanged.connect(make_expr_handler(dim_idx, lbl_val))

                def make_expr_name_saver(pref, idx):
                    def save():
                        data = _load_slider_names()
                        rn = data.get("expr_region_names", {})
                        widget = content_layout.itemAt(idx).layout().itemAt(1).widget()
                        rn[f"{pref}-{idx+1:02d}"] = widget.text().strip() or None
                        data["expr_region_names"] = rn
                        _save_slider_names(data)
                    return save

                edit_name.editingFinished.connect(make_expr_name_saver(prefix, i))

                item = QtWidgets.QVBoxLayout()
                item.setSpacing(1)
                item.setContentsMargins(0, 1, 0, 1)
                item.addLayout(row_top)
                item.addWidget(sld)
                content_layout.addLayout(item)

                sliders.append(sld)
                labels.append(lbl_val)

            grp_v.addWidget(content)
            self._expr_region_data.append((start_dim, sliders, labels))

            def make_toggle(btn, w):
                def on_toggle(checked):
                    w.setVisible(checked)
                    btn.setText(f"{'▼' if checked else '▶'}  {btn.text()[3:]}")
                return on_toggle

            toggle_btn.toggled.connect(make_toggle(toggle_btn, content))
            # Start Lower Face open, others collapsed to save space
            if region_label != "Lower Face — Mouth & Jaw":
                toggle_btn.setChecked(False)
                content.setVisible(False)
                toggle_btn.setText(f"▶  {region_label}")

            outer_layout.addWidget(grp_frame)

        outer_layout.addStretch()
        outer_scroll.setWidget(outer_container)
        layout.addWidget(outer_scroll)
        return tab

    # ─── Shared slider builder ───────────────────────────────────────────────

    def _build_slider_list(self, parent_layout, count, prefix, on_change_fn, default_names=None):
        """Build a scroll area with editable-name sliders. Returns (sliders, val_labels, name_edits)."""
        saved = _load_slider_names()
        names_key = f"{prefix}_names"

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        container = QtWidgets.QWidget()
        container.setStyleSheet("background: transparent;")
        c_layout = QtWidgets.QVBoxLayout(container)
        c_layout.setSpacing(2)
        c_layout.setContentsMargins(0, 0, 4, 0)

        sliders = []
        val_labels = []
        name_edits = []

        for i in range(count):
            # Outer row: index badge + name field + value badge
            top_row = QtWidgets.QHBoxLayout()
            top_row.setSpacing(4)

            lbl_idx = QtWidgets.QLabel(f"{i+1:02d}")
            lbl_idx.setFixedWidth(20)
            lbl_idx.setStyleSheet(
                "color:#555; font-size:10px; font-family:Consolas; "
                "background:#1a1a1a; border-radius:3px; padding:1px 3px;"
            )
            lbl_idx.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            saved_name = saved.get(names_key, [None] * count)[i] if i < len(saved.get(names_key, [])) else None
            default_name = (default_names[i] if default_names and i < len(default_names)
                            else f"{prefix}-{i+1:02d}")
            edit_name = QtWidgets.QLineEdit(saved_name or default_name)
            edit_name.setStyleSheet(
                "QLineEdit { background:#222; border:1px solid #3a3a3a; border-radius:3px; "
                "color:#aaaaaa; font-size:11px; padding:1px 4px; }"
                "QLineEdit:focus { border-color:#00AAFF; color:#dddddd; }"
            )
            edit_name.setFixedHeight(20)

            lbl_val = QtWidgets.QLabel("0.0")
            lbl_val.setFixedWidth(38)
            lbl_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            lbl_val.setStyleSheet("color:#666; font-size:12px; font-family:Consolas; font-weight:bold;")

            top_row.addWidget(lbl_idx)
            top_row.addWidget(edit_name)
            top_row.addWidget(lbl_val)

            # Slider row
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(-30, 30)
            slider.setValue(0)
            slider.setFixedHeight(14)
            slider.setStyleSheet(_SLIDER_STYLE)

            def make_handler(index, label, cb):
                def on_change(val):
                    v = val / _SLIDER_SCALE
                    _set_val_label(label, v)
                    cb(index, v)
                return on_change

            slider.valueChanged.connect(make_handler(i, lbl_val, on_change_fn))

            def make_name_saver(key, idx, all_edits):
                def on_name_changed():
                    data = _load_slider_names()
                    current = data.get(key, [None] * count)
                    while len(current) < count:
                        current.append(None)
                    current[idx] = all_edits[idx].text().strip() or None
                    data[key] = current
                    _save_slider_names(data)
                return on_name_changed

            # Connect name save after all edits are built (deferred below)
            name_edits.append(edit_name)

            item_layout = QtWidgets.QVBoxLayout()
            item_layout.setSpacing(1)
            item_layout.setContentsMargins(0, 2, 0, 2)
            item_layout.addLayout(top_row)
            item_layout.addWidget(slider)

            sep = QtWidgets.QFrame()
            sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            sep.setStyleSheet("color:#252525;")
            sep.setFixedHeight(1)
            item_layout.addWidget(sep)

            c_layout.addLayout(item_layout)
            sliders.append(slider)
            val_labels.append(lbl_val)

        # Wire name-save signals now that name_edits list is complete
        for i, edit in enumerate(name_edits):
            edit.editingFinished.connect(make_name_saver(names_key, i, name_edits))

        c_layout.addStretch()
        scroll.setWidget(container)
        scroll.setMinimumHeight(min(count * 38 + 10, 320))
        parent_layout.addWidget(scroll)

        return sliders, val_labels, name_edits

    # ─── Presets tab ─────────────────────────────────────────────────────────

    def _build_presets_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Save row ────────────────────────────────────────────────────────
        save_grp = QtWidgets.QGroupBox("Save Current Preset")
        save_form = QtWidgets.QFormLayout(save_grp)
        save_form.setSpacing(5)
        save_form.setContentsMargins(8, 10, 8, 8)

        self._edit_preset_name = QtWidgets.QLineEdit()
        self._edit_preset_name.setPlaceholderText("Preset name…")
        save_form.addRow("Name:", self._edit_preset_name)

        self._edit_preset_category = QtWidgets.QLineEdit()
        self._edit_preset_category.setPlaceholderText("Category (optional)…")
        save_form.addRow("Category:", self._edit_preset_category)

        btn_save = QtWidgets.QPushButton("Save Preset")
        btn_save.setObjectName("btn_primary")
        btn_save.clicked.connect(self._on_save_preset)
        save_form.addRow("", btn_save)

        layout.addWidget(save_grp)

        # ── Header ──────────────────────────────────────────────────────
        hdr = QtWidgets.QHBoxLayout()
        lbl_saved = QtWidgets.QLabel("Saved Presets:")
        lbl_saved.setStyleSheet("color:#aaa; font-size:11px;")
        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_refresh.setFixedHeight(22)
        btn_refresh.clicked.connect(self._refresh_presets)
        hdr.addWidget(lbl_saved)
        hdr.addStretch()
        hdr.addWidget(btn_refresh)
        layout.addLayout(hdr)

        # ── Gallery scroll area (categories + icon grids) ────────────────
        self._presets_scroll = QtWidgets.QScrollArea()
        self._presets_scroll.setWidgetResizable(True)
        self._presets_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._presets_scroll.setStyleSheet("QScrollArea { background:#1a1a1a; }")
        self._presets_container = QtWidgets.QWidget()
        self._presets_container.setStyleSheet("background:#1a1a1a;")
        self._presets_container_layout = QtWidgets.QVBoxLayout(self._presets_container)
        self._presets_container_layout.setContentsMargins(4, 4, 4, 4)
        self._presets_container_layout.setSpacing(8)
        self._presets_container_layout.addStretch()
        self._presets_scroll.setWidget(self._presets_container)
        layout.addWidget(self._presets_scroll, 1)

        # maps category name → QListWidget for quick lookup
        self._preset_lists = {}   # cat_name -> QListWidget

        self._refresh_presets()
        return tab

    # ─── Animation tab ───────────────────────────────────────────────────────

    def _build_animation_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Arm toggle ────────────────────────────────────────────────────
        self._btn_anim_arm = QtWidgets.QPushButton("Arm Animation")
        self._btn_anim_arm.setCheckable(True)
        self._btn_anim_arm.setMinimumHeight(32)
        self._btn_anim_arm.setStyleSheet(
            "QPushButton { background:#333; border:1px solid #555; border-radius:4px; "
            "color:#ddd; font-weight:bold; }"
            "QPushButton:checked { background:#7b0000; border-color:#ff4444; color:#ff8888; }"
            "QPushButton:checked:hover { background:#990000; }"
            "QPushButton:hover { background:#444; border-color:#00AAFF; }"
        )
        self._btn_anim_arm.toggled.connect(self._on_arm_animation)
        layout.addWidget(self._btn_anim_arm)

        lbl_hint = QtWidgets.QLabel(
            "Set sliders to a pose, move Max timeline to a frame, then Add Keyframe."
        )
        lbl_hint.setStyleSheet("color:#666; font-size:10px;")
        lbl_hint.setWordWrap(True)
        layout.addWidget(lbl_hint)

        # ── Keyframe list ────────────────────────────────────────────────
        self._list_keyframes = QtWidgets.QListWidget()
        self._list_keyframes.setStyleSheet(
            "QListWidget { background:#1a1a1a; border:1px solid #444; border-radius:3px; }"
            "QListWidget::item { padding:4px 8px; color:#ccc; font-family:Consolas; font-size:11px; }"
            "QListWidget::item:selected { background:#005588; color:#fff; }"
            "QListWidget::item:hover { background:#2a3a4a; }"
        )
        layout.addWidget(self._list_keyframes, 1)

        # ── Action buttons ───────────────────────────────────────────────
        btn_row1 = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("Add Keyframe")
        btn_add.setObjectName("btn_primary")
        btn_add.clicked.connect(self._on_add_keyframe)
        btn_goto = QtWidgets.QPushButton("Go To Frame")
        btn_goto.clicked.connect(self._on_goto_keyframe)
        btn_row1.addWidget(btn_add)
        btn_row1.addWidget(btn_goto)
        layout.addLayout(btn_row1)

        btn_row2 = QtWidgets.QHBoxLayout()
        btn_delete = QtWidgets.QPushButton("Delete")
        btn_delete.clicked.connect(self._on_delete_keyframe)
        btn_clear = QtWidgets.QPushButton("Clear All")
        btn_clear.clicked.connect(self._on_clear_keyframes)
        btn_row2.addWidget(btn_delete)
        btn_row2.addWidget(btn_clear)
        layout.addLayout(btn_row2)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color:#444;")
        layout.addWidget(sep)

        # ── Save row with name field ─────────────────────────────────────
        save_row = QtWidgets.QHBoxLayout()
        self._edit_anim_name = QtWidgets.QLineEdit()
        self._edit_anim_name.setPlaceholderText("Animation name…")
        btn_save_anim = QtWidgets.QPushButton("Save")
        btn_save_anim.setFixedWidth(50)
        btn_save_anim.clicked.connect(self._on_save_animation)
        save_row.addWidget(self._edit_anim_name)
        save_row.addWidget(btn_save_anim)
        layout.addLayout(save_row)

        # ── Saved animations list ────────────────────────────────────────
        self._list_saved_anims = QtWidgets.QListWidget()
        self._list_saved_anims.setMaximumHeight(80)
        self._list_saved_anims.setStyleSheet(
            "QListWidget { background:#111; border:1px solid #444; border-radius:3px; }"
            "QListWidget::item { padding:2px 6px; color:#aaa; font-size:11px; }"
            "QListWidget::item:selected { background:#005588; color:#fff; }"
        )
        self._list_saved_anims.itemDoubleClicked.connect(
            lambda item: self._load_animation_by_name(item.text()))
        layout.addWidget(self._list_saved_anims)

        btn_row3 = QtWidgets.QHBoxLayout()
        btn_load_anim = QtWidgets.QPushButton("Load Selected")
        btn_load_anim.clicked.connect(self._on_load_animation)
        btn_del_anim = QtWidgets.QPushButton("Delete")
        btn_del_anim.clicked.connect(self._on_delete_saved_animation)
        btn_row3.addWidget(btn_load_anim)
        btn_row3.addWidget(btn_del_anim)
        layout.addLayout(btn_row3)
        self._refresh_saved_anims()

        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#444;")
        layout.addWidget(sep2)

        self._btn_bake = QtWidgets.QPushButton("Bake to Timeline")
        self._btn_bake.setMinimumHeight(32)
        self._btn_bake.setToolTip(
            "Render every frame in the scene timeline and bake vertex positions "
            "directly into Max's timeline as animated vertex keys."
        )
        self._btn_bake.setStyleSheet(
            "QPushButton { background:#1a3300; border:1px solid #336600; border-radius:4px; "
            "color:#88ff44; font-weight:bold; }"
            "QPushButton:hover { background:#224400; border-color:#66cc00; }"
            "QPushButton:disabled { background:#222; color:#555; border-color:#444; }"
        )
        self._btn_bake.clicked.connect(self._on_bake_pc2)
        layout.addWidget(self._btn_bake)

        self._bake_progress = QtWidgets.QProgressBar()
        self._bake_progress.setRange(0, 100)
        self._bake_progress.setValue(0)
        self._bake_progress.setVisible(False)
        self._bake_progress.setStyleSheet(
            "QProgressBar { border:1px solid #555; border-radius:3px; background:#222; height:14px; }"
            "QProgressBar::chunk { background:#66cc00; border-radius:2px; }"
        )
        layout.addWidget(self._bake_progress)

        return tab

    # ─── Population tab ──────────────────────────────────────────────────────

    def _build_population_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        grp = QtWidgets.QGroupBox("Grid Settings")
        form = QtWidgets.QFormLayout(grp)
        form.setSpacing(6)
        form.setContentsMargins(8, 10, 8, 8)

        spin_style = "QSpinBox, QDoubleSpinBox { background:#2a2a2a; border:1px solid #555; border-radius:3px; padding:2px 4px; }"

        self._spin_count = QtWidgets.QSpinBox()
        self._spin_count.setRange(1, 100)
        self._spin_count.setValue(9)
        self._spin_count.setStyleSheet(spin_style)
        form.addRow("Count:", self._spin_count)

        self._spin_cols = QtWidgets.QSpinBox()
        self._spin_cols.setRange(1, 20)
        self._spin_cols.setValue(3)
        self._spin_cols.setStyleSheet(spin_style)
        form.addRow("Columns:", self._spin_cols)

        self._spin_spacing_x = QtWidgets.QDoubleSpinBox()
        self._spin_spacing_x.setRange(1.0, 1000.0)
        self._spin_spacing_x.setValue(20.0)
        self._spin_spacing_x.setSuffix(" cm")
        self._spin_spacing_x.setStyleSheet(spin_style)
        form.addRow("Spacing X:", self._spin_spacing_x)

        self._spin_spacing_y = QtWidgets.QDoubleSpinBox()
        self._spin_spacing_y.setRange(1.0, 1000.0)
        self._spin_spacing_y.setValue(25.0)
        self._spin_spacing_y.setSuffix(" cm")
        self._spin_spacing_y.setStyleSheet(spin_style)
        form.addRow("Spacing Y:", self._spin_spacing_y)

        self._spin_seed = QtWidgets.QSpinBox()
        self._spin_seed.setRange(0, 99999)
        self._spin_seed.setValue(42)
        self._spin_seed.setStyleSheet(spin_style)
        form.addRow("Seed:", self._spin_seed)

        layout.addWidget(grp)

        self._btn_generate_grid = QtWidgets.QPushButton("Generate Grid")
        self._btn_generate_grid.setObjectName("btn_primary")
        self._btn_generate_grid.setMinimumHeight(34)
        self._btn_generate_grid.clicked.connect(self._on_generate_population)
        layout.addWidget(self._btn_generate_grid)

        self._pop_progress = QtWidgets.QProgressBar()
        self._pop_progress.setRange(0, 100)
        self._pop_progress.setValue(0)
        self._pop_progress.setVisible(False)
        self._pop_progress.setStyleSheet(
            "QProgressBar { border:1px solid #555; border-radius:3px; background:#222; height:16px; }"
            "QProgressBar::chunk { background-color:#00AAFF; border-radius:2px; }"
        )
        layout.addWidget(self._pop_progress)

        self._lbl_pop_status = QtWidgets.QLabel("")
        self._lbl_pop_status.setStyleSheet("color:#aaaaaa; font-size:11px;")
        layout.addWidget(self._lbl_pop_status)

        layout.addStretch()
        return tab

    # ─── State switching ─────────────────────────────────────────────────────

    def _show_setup_state(self):
        self._stack.setCurrentWidget(self._page_setup)
        self._lbl_status_badge.setText("Not installed")
        self._lbl_status_badge.setStyleSheet("color: #FF4444; font-size: 11px;")
        self._btn_settings.setVisible(False)

    def _show_ready_state(self):
        self._stack.setCurrentWidget(self._page_ready)
        self._lbl_status_badge.setText("✓ Ready")
        self._lbl_status_badge.setStyleSheet("color: #00FF00; font-size: 11px;")
        self._btn_settings.setVisible(True)

    # ─── Setup flow ──────────────────────────────────────────────────────────

    def _on_start_setup(self):
        import traceback
        try:
            self._btn_setup.setEnabled(False)
            self._btn_setup.setText("Setting up...")
            self._progress_bar.setVisible(True)
            self._lbl_progress_text.setVisible(True)
            self._progress_bar.setValue(0)

            self._setup_worker = SetupWorker(self)
            self._setup_worker.sig_progress.connect(lambda p, l: (
                self._progress_bar.setValue(p),
                self._lbl_progress_text.setText(l),
            ))
            self._setup_worker.sig_log.connect(self.logger.info)
            self._setup_worker.sig_done.connect(self._on_setup_done)
            self._setup_worker.start()
            self.logger.info("Setup worker started.")
        except Exception:
            self.logger.error(f"Failed to start setup:\n{traceback.format_exc()}")

    def _on_setup_done(self, success: bool, result: str):
        self._btn_setup.setEnabled(True)
        self._btn_setup.setText("Install && Setup GNM")
        if success:
            self._progress_bar.setValue(100)
            self.logger.info("Setup complete.")
            gnm_bridge.reset_model()
            if not _NUMPY_OK:
                # numpy was just installed — must relaunch so the new import takes effect
                self.logger.info("Relaunching tool to activate newly installed packages…")
                QtCore.QTimer.singleShot(500, self._relaunch)
            else:
                self._show_ready_state()
        else:
            self.logger.error(f"Setup failed: {result}")
            self._lbl_progress_text.setText("Error — check the log.")

    def _relaunch(self):
        # launch.launch() handles close + module eviction + fresh import internally
        try:
            import GNM.launch as _launch
            _launch.launch()
        except Exception as e:
            print(f"[GNM] Relaunch failed: {e}. Please run the launch script again manually.")

    # ─── New mesh ────────────────────────────────────────────────────────────

    def _on_new_mesh(self):
        if not gnm_bridge.is_gnm_available():
            self.logger.error("GNM not loaded. Please reopen the tool.")
            return

        # Stop debounce while creating to avoid re-entrant calls
        self._debounce.stop()
        self._pending = False

        name = self._edit_name.text().strip() or "GNM_Head"
        self._btn_new_mesh.setEnabled(False)
        self._btn_new_mesh.setText("Creating...")

        eff_identity = self._get_gendered_identity()
        vertices, triangles, triangle_uvs = gnm_bridge.generate_head(
            identity=eff_identity, expression=self._expression, rotations=self._rotations)
        if vertices is not None:
            self._current_mesh = gnm_bridge.create_max_mesh(
                vertices, triangles, triangle_uvs=triangle_uvs, name=name,
                identity=self._identity, logger=self.logger
            )
        else:
            self.logger.error("Mesh creation failed.")

        self._btn_new_mesh.setEnabled(True)
        self._btn_new_mesh.setText("Create New Mesh")
        QtCore.QTimer.singleShot(0, self._redraw_views)

    # ─── Live update flow ────────────────────────────────────────────────────

    def _trigger_update(self):
        if self._gen_worker is not None and self._gen_worker.isRunning():
            self._pending = True
            return

        if not self._check_mesh_valid():
            self._on_new_mesh()
            return

        self._set_update_status("⟳ Updating...", "#FFA500")
        worker = GenerateWorker(self._get_gendered_identity(), self._expression, self._rotations, parent=self)
        worker.sig_vertices.connect(self._on_vertices_ready)
        worker.finished.connect(self._on_worker_finished)
        self._gen_worker = worker
        worker.start()

    def _on_vertices_ready(self, vertices):
        if vertices is None or not self._check_mesh_valid():
            return
        gnm_bridge.update_max_mesh_vertices(self._current_mesh, vertices)
        gnm_bridge.save_identity_to_mesh(self._current_mesh, self._gen_worker._identity)
        self._log_active_params()
        # Defer redraw outside Max's notification stack (fixes DirtyNotificationEventMonitor on Max 2027)
        QtCore.QTimer.singleShot(0, self._redraw_views)

    def _redraw_views(self):
        try:
            import pymxs
            pymxs.runtime.redrawViews()
        except Exception:
            pass

    def _on_worker_finished(self):
        self._set_update_status("✓ Updated", "#00FF00")
        self._status_clear.start()

        if self._pending:
            self._pending = False
            self._trigger_update()

    def _on_use_selected(self):
        try:
            import pymxs
            rt = pymxs.runtime
            sel = rt.selection
            if not sel or len(sel) == 0:
                self.logger.warning("Nothing selected. Select a GNM mesh in the viewport first.")
                return
            node = sel[0]
            if not gnm_bridge.is_gnm_mesh(node):
                self.logger.warning(f'"{node.name}" is not a GNM mesh.')
                return
            identity = gnm_bridge.load_identity_from_mesh(node)
            if identity is None:
                self.logger.warning("Could not read identity data from mesh.")
                return
            self._current_mesh = node
            self._identity[:] = 0.0
            self._identity[:len(identity)] = identity[:len(self._identity)]
            self._apply_identity_to_sliders()
            self.logger.info(f'Now editing "{node.name}".')
        except Exception as e:
            self.logger.error(f"Use Selected error: {e}")

    # ─── Slider callbacks ────────────────────────────────────────────────────

    def _on_gender_slider_changed(self, value: int):
        self._gender_strength = value / 100.0   # -1.0 … +1.0
        self._debounce.start()

    def _get_gendered_identity(self):
        """Return identity blended with gender average. Does not modify self._identity."""
        if self._gender_strength == 0.0:
            return self._identity

        result = gnm_bridge.get_gender_vector(logger=self.logger)
        if result is None:
            return self._identity

        male_avg, female_avg, _direction = result
        t = self._gender_strength   # -1=female, 0=neutral, +1=male

        if t > 0:
            # blend toward male average
            gender_base = male_avg * t
        else:
            # blend toward female average
            gender_base = female_avg * (-t)

        # Add gender offset on top of the current identity sliders
        return (self._identity + gender_base * 0.5).astype(np.float32)

    def _on_identity_slider_changed(self, index, value):
        self._identity[index] = value
        self._debounce.start()

    def _on_expression_slider_changed(self, index, value):
        # legacy — no longer called (region sliders use inline handlers)
        self._expression[index] = value
        self._debounce.start()

    def _apply_identity_to_sliders(self):
        for i, (s, lbl) in enumerate(zip(self._sliders, self._slider_labels)):
            v = float(self._identity[i])
            s.blockSignals(True)
            s.setValue(int(round(v * _SLIDER_SCALE)))
            s.blockSignals(False)
            _set_val_label(lbl, v)

    def _apply_expression_to_sliders(self):
        for start_dim, sliders, labels in self._expr_region_data:
            for offset, (s, lbl) in enumerate(zip(sliders, labels)):
                v = float(self._expression[start_dim + offset])
                s.blockSignals(True)
                s.setValue(int(round(v * _SLIDER_SCALE)))
                s.blockSignals(False)
                _set_val_label(lbl, v)

    def _log_active_params(self):
        """Log non-zero identity, expression, and rotation values to the log panel."""
        parts = []

        id_active = [(i, float(self._identity[i]))
                     for i in range(_NUM_IDENTITY_SLIDERS) if abs(self._identity[i]) > 0.05]
        if id_active:
            id_names = _load_slider_names().get("IC_names", [None] * _NUM_IDENTITY_SLIDERS)
            id_strs = []
            for i, v in id_active:
                name = (id_names[i] if i < len(id_names) and id_names[i] else f"IC-{i+1:02d}")
                id_strs.append(f"{name}={v:+.1f}")
            parts.append("Identity: " + ", ".join(id_strs))

        ex_strs = []
        saved_rnames = _load_slider_names().get("expr_region_names", {})
        for region_label, start_dim, count, prefix in _EXPRESSION_REGIONS:
            for offset in range(count):
                v = float(self._expression[start_dim + offset])
                if abs(v) > 0.05:
                    key = f"{prefix}-{offset+1:02d}"
                    name = saved_rnames.get(key) or key
                    ex_strs.append(f"{name}={v:+.1f}")
        if ex_strs:
            parts.append("Expr: " + ", ".join(ex_strs))

        rot_labels = [
            ("Head", 0), ("Neck", 1), ("Eye-L", 2), ("Eye-R", 3)
        ]
        rot_axis = ["X", "Y", "Z"]
        rot_strs = []
        for lbl, ri in rot_labels:
            for ai in range(3):
                deg = math.degrees(float(self._rotations[ri, ai]))
                if abs(deg) > 0.5:
                    rot_strs.append(f"{lbl}.{rot_axis[ai]}={deg:.0f}°")
        if rot_strs:
            parts.append("Pose: " + ", ".join(rot_strs))

        if parts:
            self.logger.info(" | ".join(parts))

    # ─── Slider controls — Identity ──────────────────────────────────────────

    def _on_reset_sliders(self):
        for s in self._sliders:
            s.blockSignals(True)
            s.setValue(0)
            s.blockSignals(False)
        for lbl in self._slider_labels:
            _set_val_label(lbl, 0.0)
        self._identity[:_NUM_IDENTITY_SLIDERS] = 0.0
        self._debounce.start()

    def _on_randomize_sliders(self):
        vals = (np.random.randn(_NUM_IDENTITY_SLIDERS) * 10).clip(-30, 30).astype(int)
        for i, (s, lbl) in enumerate(zip(self._sliders, self._slider_labels)):
            s.blockSignals(True)
            s.setValue(int(vals[i]))
            s.blockSignals(False)
            v = vals[i] / _SLIDER_SCALE
            _set_val_label(lbl, v)
            self._identity[i] = v
        self._debounce.start()

    # ─── Pose controls ───────────────────────────────────────────────────────

    def _on_reset_pose(self):
        self._rotations[:] = 0.0
        for sld in self._pose_sliders.values():
            sld.blockSignals(True)
            sld.setValue(0)
            sld.blockSignals(False)
        self._debounce.start()

    # ─── Slider controls — Expression ────────────────────────────────────────

    def _on_reset_expr_sliders(self):
        self._expression[:] = 0.0
        for _, sliders, labels in self._expr_region_data:
            for s in sliders:
                s.blockSignals(True)
                s.setValue(0)
                s.blockSignals(False)
            for lbl in labels:
                _set_val_label(lbl, 0.0)
        self._debounce.start()

    def _on_randomize_expr_sliders(self):
        # Only randomize lower_face (200-224) and eye dims (0-19, 100-119)
        self._expression[:] = 0.0
        randomize_ranges = [(200, 25), (0, 20), (100, 20)]
        for start, count in randomize_ranges:
            vals = np.random.uniform(-3.0, 3.0, count).astype(np.float32)
            self._expression[start:start + count] = vals

        # Reflect in sliders
        for start_dim, sliders, labels in self._expr_region_data:
            for offset, (s, lbl) in enumerate(zip(sliders, labels)):
                v = float(self._expression[start_dim + offset])
                s.blockSignals(True)
                s.setValue(int(round(v * _SLIDER_SCALE)))
                s.blockSignals(False)
                _set_val_label(lbl, v)
        self._debounce.start()

    # ─── Presets logic ───────────────────────────────────────────────────────

    _THUMB_W = 128   # icon display size in the gallery

    def _refresh_presets(self):
        """Rebuild the gallery: one collapsible section per category, 3-per-row icon grid."""
        # Clear existing sections
        layout = self._presets_container_layout
        while layout.count() > 1:           # keep the trailing stretch
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._preset_lists = {}

        if not _PRESETS_DIR.exists():
            return

        # Collect presets grouped by category
        categories = {}   # cat -> [(name, path, thumb_path)]
        for f in sorted(_PRESETS_DIR.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            cat = data.get("category", "").strip() or "Uncategorized"
            preset_name = data.get("name", f.stem)
            categories.setdefault(cat, []).append((preset_name, str(f), f.with_suffix(".png")))

        insert_pos = 0
        for cat, presets in sorted(categories.items()):
            # ── Category header with rename button ───────────────────────
            cat_hdr = QtWidgets.QWidget()
            cat_hdr.setStyleSheet("background:#252525; border-radius:3px;")
            hdr_row = QtWidgets.QHBoxLayout(cat_hdr)
            hdr_row.setContentsMargins(8, 4, 4, 4)
            hdr_row.setSpacing(4)

            btn_toggle = QtWidgets.QPushButton(f"▼  {cat}")
            btn_toggle.setCheckable(True)
            btn_toggle.setChecked(True)
            btn_toggle.setStyleSheet(
                "QPushButton { background:transparent; border:none; color:#00AAFF; "
                "font-weight:bold; font-size:11px; text-align:left; padding:0; }"
                "QPushButton:hover { color:#33ccff; }"
            )

            btn_rename_cat = QtWidgets.QPushButton("✎")
            btn_rename_cat.setFixedSize(22, 22)
            btn_rename_cat.setToolTip("Rename category")
            btn_rename_cat.setStyleSheet(
                "QPushButton { background:#333; border:1px solid #555; border-radius:3px; "
                "color:#aaa; font-size:11px; }"
                "QPushButton:hover { color:#fff; border-color:#00AAFF; }"
            )

            hdr_row.addWidget(btn_toggle, 1)
            hdr_row.addWidget(btn_rename_cat)

            # ── Icon grid (QListWidget in icon mode) ─────────────────────
            lst = QtWidgets.QListWidget()
            lst.setViewMode(QtWidgets.QListWidget.ViewMode.IconMode)
            lst.setIconSize(QtCore.QSize(self._THUMB_W, self._THUMB_W))
            lst.setGridSize(QtCore.QSize(self._THUMB_W + 20, self._THUMB_W + 30))
            lst.setResizeMode(QtWidgets.QListWidget.ResizeMode.Adjust)
            lst.setMovement(QtWidgets.QListWidget.Movement.Static)
            lst.setUniformItemSizes(True)
            lst.setSpacing(4)
            lst.setWordWrap(True)
            lst.setStyleSheet(
                "QListWidget { background:#1e1e1e; border:none; }"
                "QListWidget::item { color:#cccccc; font-size:10px; border-radius:4px; }"
                "QListWidget::item:selected { background:#005588; color:#fff; }"
                "QListWidget::item:hover { background:#2a3a4a; }"
            )

            for preset_name, path_str, thumb_path in presets:
                list_item = QtWidgets.QListWidgetItem(preset_name)
                list_item.setData(QtCore.Qt.ItemDataRole.UserRole, path_str)
                list_item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, cat)
                if thumb_path.exists():
                    pix = QtGui.QPixmap(str(thumb_path))
                    if not pix.isNull():
                        list_item.setIcon(QtGui.QIcon(pix.scaled(
                            self._THUMB_W, self._THUMB_W,
                            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                            QtCore.Qt.TransformationMode.SmoothTransformation,
                        )))
                lst.addItem(list_item)

            # Row height = enough for thumb + label
            row_h = self._THUMB_W + 34
            n_rows = max(1, math.ceil(len(presets) / 3))
            lst.setFixedHeight(n_rows * row_h + 8)

            # Wire signals
            lst.itemDoubleClicked.connect(self._on_preset_item_double_clicked)
            lst.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            lst.customContextMenuRequested.connect(
                lambda pos, l=lst: self._on_preset_grid_context_menu(pos, l)
            )
            btn_toggle.toggled.connect(
                lambda checked, l=lst, b=btn_toggle, c=cat: (
                    l.setVisible(checked),
                    b.setText(f"{'▼' if checked else '▶'}  {c}"),
                )
            )
            btn_rename_cat.clicked.connect(
                lambda _, c=cat: self._rename_category(c)
            )

            layout.insertWidget(insert_pos, cat_hdr)
            layout.insertWidget(insert_pos + 1, lst)
            insert_pos += 2
            self._preset_lists[cat] = lst

    def _selected_preset_path(self):
        """Return (path_str, name) for the currently selected preset leaf, or (None, None)."""
        item = self._tree_presets.currentItem()
        if item is None:
            return None, None
        path = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if path is None:
            return None, None  # category node
        return path, item.text(0)

    def _load_preset_from_path(self, path: str):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            id_arr = np.array(data.get("identity", []), dtype=np.float32)
            ex_arr = np.array(data.get("expression", []), dtype=np.float32)
            n_id = min(len(id_arr), len(self._identity))
            n_ex = min(len(ex_arr), len(self._expression))
            self._identity[:] = 0.0
            self._identity[:n_id] = id_arr[:n_id]
            self._expression[:] = 0.0
            self._expression[:n_ex] = ex_arr[:n_ex]
            self._apply_identity_to_sliders()
            self._apply_expression_to_sliders()
            self._debounce.start()
            self.logger.info(f'Preset "{data.get("name", "")}" loaded.')
        except Exception as e:
            self.logger.error(f"Failed to load preset: {e}")

    def _on_preset_item_double_clicked(self, item):
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if path:
            self._load_preset_from_path(path)

    def _on_preset_grid_context_menu(self, pos, lst: QtWidgets.QListWidget):
        item = lst.itemAt(pos)
        if item is None:
            return
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not path:
            return

        menu = QtWidgets.QMenu(self)
        act_load = menu.addAction("Load")
        act_rename = menu.addAction("Rename…")
        act_move = menu.addAction("Set Category…")
        menu.addSeparator()
        act_delete = menu.addAction("Delete")

        chosen = menu.exec(lst.viewport().mapToGlobal(pos))
        if chosen == act_load:
            self._load_preset_from_path(path)
        elif chosen == act_rename:
            self._rename_preset(path, item.text())
        elif chosen == act_move:
            self._set_preset_category(path, item.text())
        elif chosen == act_delete:
            self._delete_preset_at(path, item.text())

    def _rename_category(self, old_name: str):
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Category", "New category name:", text=old_name)
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name == old_name:
            return
        count = 0
        for f in _PRESETS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                current_cat = data.get("category", "").strip() or "Uncategorized"
                if current_cat == old_name:
                    data["category"] = new_name
                    f.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    count += 1
            except Exception:
                continue
        self._refresh_presets()
        self.logger.info(f'Category renamed from "{old_name}" to "{new_name}" ({count} presets updated).')

    def _rename_preset(self, path: str, current_name: str):
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Preset", "New name:", text=current_name)
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        try:
            old_path = Path(path)
            data = json.loads(old_path.read_text(encoding="utf-8"))
            data["name"] = new_name
            new_path = _PRESETS_DIR / f"{new_name}.json"
            new_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            # rename thumbnail too
            old_thumb = old_path.with_suffix(".png")
            new_thumb = new_path.with_suffix(".png")
            if old_thumb.exists() and old_path != new_path:
                old_thumb.rename(new_thumb)
            if old_path != new_path:
                old_path.unlink(missing_ok=True)
            self._refresh_presets()
            self.logger.info(f'Preset renamed to "{new_name}".')
        except Exception as e:
            self.logger.error(f"Failed to rename preset: {e}")

    def _set_preset_category(self, path: str, preset_name: str):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.error(f"Failed to read preset: {e}")
            return
        current_cat = data.get("category", "")
        new_cat, ok = QtWidgets.QInputDialog.getText(
            self, "Set Category", "Category name:", text=current_cat)
        if not ok:
            return
        data["category"] = new_cat.strip()
        try:
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._refresh_presets()
            self.logger.info(f'Preset "{preset_name}" moved to category "{data["category"] or "Uncategorized"}".')
        except Exception as e:
            self.logger.error(f"Failed to update preset: {e}")

    def _delete_preset_at(self, path: str, name: str):
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Preset", f'Delete "{name}"?',
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            try:
                p = Path(path)
                p.unlink(missing_ok=True)
                p.with_suffix(".png").unlink(missing_ok=True)  # delete thumbnail too
                self._refresh_presets()
                self.logger.info(f'Preset "{name}" deleted.')
            except Exception as e:
                self.logger.error(f"Failed to delete preset: {e}")

    def _capture_viewport_thumbnail(self, dest_path: Path) -> bool:
        """Grab the active Max viewport as a PNG thumbnail. Returns True on success."""
        import tempfile
        try:
            import pymxs
            rt = pymxs.runtime
            # gw.getViewportDib() captures the current viewport instantly (no render)
            tmp = Path(tempfile.mktemp(suffix=".bmp"))
            tmp_str = str(tmp).replace("\\", "/")
            rt.execute(
                f'(local d = gw.getViewportDib(); d.filename = "{tmp_str}"; save d; close d)'
            )
            if tmp.exists():
                pix = QtGui.QPixmap(str(tmp))
                tmp.unlink(missing_ok=True)
                if not pix.isNull():
                    # Crop to square from center, then scale to 256×256
                    w, h = pix.width(), pix.height()
                    side = min(w, h)
                    square = pix.copy((w - side) // 2, (h - side) // 2, side, side)
                    thumb = square.scaled(
                        256, 256,
                        QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                    thumb.save(str(dest_path), "PNG")
                    return True
        except Exception as e:
            self.logger.warning(f"Thumbnail capture failed: {e}")
        return False

    def _on_save_preset(self):
        name = self._edit_preset_name.text().strip()
        if not name:
            self.logger.warning("Enter a preset name first.")
            return
        category = self._edit_preset_category.text().strip()
        _PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        dest = _PRESETS_DIR / f"{name}.json"
        data = {
            "name": name,
            "category": category,
            "identity": self._identity.tolist(),
            "expression": self._expression.tolist(),
        }
        try:
            dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
            # Capture viewport thumbnail alongside the JSON
            thumb_path = dest.with_suffix(".png")
            self._capture_viewport_thumbnail(thumb_path)
            self.logger.info(f'Preset "{name}" saved{f" in {category!r}" if category else ""}.')
            self._refresh_presets()
        except Exception as e:
            self.logger.error(f"Failed to save preset: {e}")

    # ─── Animation logic ─────────────────────────────────────────────────────

    _ANIM_DIR = Path(__file__).parent / "animation"

    def _on_arm_animation(self, checked: bool):
        import GNM.launch as _launch
        if checked:
            if not self._check_mesh_valid():
                self.logger.warning("No active GNM mesh — create one first.")
                self._btn_anim_arm.blockSignals(True)
                self._btn_anim_arm.setChecked(False)
                self._btn_anim_arm.blockSignals(False)
                return
            _launch._register_time_callback(self)
            self._anim_armed = True
            self._btn_anim_arm.setText("Disarm Animation")
            self.logger.info("Animation armed — scrub the Max timeline to update the mesh.")
        else:
            _launch._unregister_time_callback()
            self._anim_armed = False
            self._btn_anim_arm.setText("Arm Animation")
            self.logger.info("Animation disarmed.")

    def _on_add_keyframe(self):
        if not self._check_mesh_valid():
            self.logger.warning("No active GNM mesh.")
            return
        try:
            import pymxs
            frame = int(pymxs.runtime.currentTime.frame)
        except Exception:
            self.logger.error("Could not read current frame from Max.")
            return

        # Replace existing keyframe at this frame if present
        self._anim_keyframes = [k for k in self._anim_keyframes if k["frame"] != frame]
        self._anim_keyframes.append({
            "frame":      frame,
            "identity":   self._get_gendered_identity().tolist(),
            "expression": self._expression.tolist(),
            "rotations":  self._rotations.tolist(),
            "gender":     self._gender_strength,
        })
        self._anim_keyframes.sort(key=lambda k: k["frame"])
        self._refresh_keyframe_list()
        self.logger.info(f"Keyframe added at frame {frame}.")

    def _on_delete_keyframe(self):
        row = self._list_keyframes.currentRow()
        if row < 0:
            return
        frame = self._anim_keyframes[row]["frame"]
        del self._anim_keyframes[row]
        self._refresh_keyframe_list()
        self.logger.info(f"Keyframe at frame {frame} deleted.")

    def _on_clear_keyframes(self):
        reply = QtWidgets.QMessageBox.question(
            self, "Clear All", "Delete all keyframes?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self._anim_keyframes.clear()
            self._refresh_keyframe_list()
            self.logger.info("All keyframes cleared.")

    def _on_goto_keyframe(self):
        row = self._list_keyframes.currentRow()
        if row < 0:
            return
        frame = self._anim_keyframes[row]["frame"]
        try:
            import pymxs
            pymxs.runtime.sliderTime = pymxs.runtime.time(frame)
        except Exception as e:
            self.logger.error(f"Could not set frame: {e}")

    def _refresh_keyframe_list(self):
        self._list_keyframes.clear()
        for kf in self._anim_keyframes:
            # Build a compact summary of non-zero dims
            id_arr = np.array(kf["identity"], dtype=np.float32)
            ex_arr = np.array(kf["expression"], dtype=np.float32)
            nonzero_id = int(np.count_nonzero(np.abs(id_arr) > 0.05))
            nonzero_ex = int(np.count_nonzero(np.abs(ex_arr) > 0.05))
            g = kf.get("gender", 0.0)
            gender_str = f"  G:{g:+.1f}" if abs(g) > 0.01 else ""
            label = (f"Frame {kf['frame']:>4}  —  "
                     f"IC:{nonzero_id}  EX:{nonzero_ex}{gender_str}")
            self._list_keyframes.addItem(label)

    def _interpolate_at_frame(self, frame: int):
        """Return (identity, expression, rotations) interpolated at the given frame."""
        kfs = self._anim_keyframes
        if not kfs:
            return None, None, None
        if frame <= kfs[0]["frame"]:
            k = kfs[0]
        elif frame >= kfs[-1]["frame"]:
            k = kfs[-1]
        else:
            # Find surrounding pair
            prev = next(k for k in reversed(kfs) if k["frame"] <= frame)
            nxt  = next(k for k in kfs if k["frame"] > frame)
            t = (frame - prev["frame"]) / (nxt["frame"] - prev["frame"])
            identity   = (np.array(prev["identity"],   dtype=np.float32) * (1 - t)
                        + np.array(nxt["identity"],    dtype=np.float32) * t)
            expression = (np.array(prev["expression"], dtype=np.float32) * (1 - t)
                        + np.array(nxt["expression"],  dtype=np.float32) * t)
            rotations  = (np.array(prev["rotations"],  dtype=np.float32) * (1 - t)
                        + np.array(nxt["rotations"],   dtype=np.float32) * t)
            return identity, expression, rotations
        return (np.array(k["identity"],   dtype=np.float32),
                np.array(k["expression"], dtype=np.float32),
                np.array(k["rotations"],  dtype=np.float32))

    def apply_params_at_frame(self, frame: int):
        """Called by the MAXScript time callback on every frame change."""
        if not self._anim_armed or not self._check_mesh_valid():
            return
        identity, expression, rotations = self._interpolate_at_frame(frame)
        if identity is None:
            return
        gnm_bridge.interpolate_and_update(
            self._current_mesh, identity, expression, rotations, logger=self.logger)
        QtCore.QTimer.singleShot(0, self._redraw_views)

    def _refresh_saved_anims(self):
        self._list_saved_anims.clear()
        if self._ANIM_DIR.exists():
            for f in sorted(self._ANIM_DIR.glob("*.json")):
                self._list_saved_anims.addItem(f.stem)

    def _on_save_animation(self):
        name = self._edit_anim_name.text().strip()
        if not name:
            name = self._current_mesh.name if self._check_mesh_valid() else "animation"
        self._ANIM_DIR.mkdir(parents=True, exist_ok=True)
        dest = self._ANIM_DIR / f"{name}.json"
        try:
            dest.write_text(json.dumps(self._anim_keyframes, indent=2), encoding="utf-8")
            self.logger.info(f"Animation '{name}' saved ({len(self._anim_keyframes)} keyframes).")
            self._refresh_saved_anims()
        except Exception as e:
            self.logger.error(f"Failed to save animation: {e}")

    def _load_animation_by_name(self, name: str):
        src = self._ANIM_DIR / f"{name}.json"
        if not src.exists():
            self.logger.warning(f"Animation file not found: {src.name}")
            return
        try:
            self._anim_keyframes = json.loads(src.read_text(encoding="utf-8"))
            self._anim_keyframes.sort(key=lambda k: k["frame"])
            self._refresh_keyframe_list()
            self.logger.info(f"Animation '{name}' loaded: {len(self._anim_keyframes)} keyframes.")
            if self._anim_keyframes:
                kf = self._anim_keyframes[0]
                id_arr = np.array(kf["identity"],   dtype=np.float32)
                ex_arr = np.array(kf["expression"], dtype=np.float32)
                n_id = min(len(id_arr), len(self._identity))
                n_ex = min(len(ex_arr), len(self._expression))
                self._identity[:] = 0.0;  self._identity[:n_id] = id_arr[:n_id]
                self._expression[:] = 0.0; self._expression[:n_ex] = ex_arr[:n_ex]
                self._rotations[:] = np.array(kf["rotations"], dtype=np.float32)
                self._gender_strength = float(kf.get("gender", 0.0))
                self._apply_identity_to_sliders()
                self._apply_expression_to_sliders()
        except Exception as e:
            self.logger.error(f"Failed to load animation: {e}")

    def _on_load_animation(self):
        item = self._list_saved_anims.currentItem()
        if item:
            self._load_animation_by_name(item.text())
        else:
            self.logger.warning("Select an animation from the list, or double-click it.")

    def _on_delete_saved_animation(self):
        item = self._list_saved_anims.currentItem()
        if not item:
            return
        name = item.text()
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Animation", f'Delete "{name}"?',
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            try:
                (self._ANIM_DIR / f"{name}.json").unlink(missing_ok=True)
                self._refresh_saved_anims()
                self.logger.info(f"Animation '{name}' deleted.")
            except Exception as e:
                self.logger.error(f"Failed to delete: {e}")

    def _on_bake_pc2(self):
        """Bake GNM keyframes to Max timeline using Morpher modifier.

        Strategy:
        1. Generate a mesh snapshot for each defined GNM keyframe
        2. Add a Morpher modifier to the base mesh
        3. Load each snapshot as a morph target (channel)
        4. Key each channel's weight: 100 at its frame, 0 elsewhere
        This produces real keyframes visible on the Max timeline.
        """
        if not self._check_mesh_valid():
            self.logger.warning("No active GNM mesh.")
            return
        if len(self._anim_keyframes) < 2:
            self.logger.warning("Need at least 2 keyframes to bake.")
            return

        import pymxs
        rt = pymxs.runtime

        try:
            start_frame = int(rt.animationRange.start.frame)
            end_frame   = int(rt.animationRange.end.frame)
        except Exception as e:
            self.logger.error(f"Could not read scene timeline: {e}")
            return

        mesh_obj  = self._current_mesh
        mesh_name = mesh_obj.name
        kfs = sorted(self._anim_keyframes, key=lambda k: k["frame"])
        n_kf = len(kfs)

        self._btn_bake.setEnabled(False)
        self._bake_progress.setVisible(True)
        self._bake_progress.setValue(0)
        QtWidgets.QApplication.processEvents()

        target_names = []
        try:
            # Step 1 — Generate snapshot mesh for every keyframe
            self.logger.info("Baking: creating morph target meshes…")
            for fi, kf in enumerate(kfs):
                id_arr = np.array(kf["identity"],   dtype=np.float32)
                ex_arr = np.array(kf["expression"], dtype=np.float32)
                ro_arr = np.array(kf["rotations"],  dtype=np.float32)
                vertices, triangles, _ = gnm_bridge.generate_head(
                    identity=id_arr, expression=ex_arr, rotations=ro_arr)
                if vertices is None:
                    self.logger.error(f"Keyframe {kf['frame']}: generate_head failed.")
                    return
                tname = f"_gnm_target_{fi}"
                target_names.append(tname)
                gnm_bridge.create_max_mesh(vertices, triangles, name=tname)
                self._bake_progress.setValue(int((fi + 1) / n_kf * 40))
                QtWidgets.QApplication.processEvents()

            # Step 2 — Add Morpher modifier via MAXScript
            self.logger.info("Baking: adding Morpher modifier…")
            rt.execute(
                f'(select (getNodeByName "{mesh_name}"); '
                f'max modify mode; '
                f'addModifier $ (Morpher()))'
            )

            # Step 3 — Load targets and key weights
            # WM3 functions require the node to be selected in Modify panel
            self.logger.info("Baking: keying morph weights…")
            for fi, (kf, tname) in enumerate(zip(kfs, target_names)):
                frame = kf["frame"]
                ch = fi + 1

                # Single MAXScript block: node is selected, modify panel active
                mxs = (
                    f'(\n'
                    f'  local m = $.modifiers[#Morpher]\n'
                    f'  local t = getNodeByName "{tname}"\n'
                    f'  WM3_MC_BuildFromNode m {ch} t\n'
                    f'  animate on (\n'
                    f'    at time {start_frame}f (WM3_MC_SetValue m {ch} 0.0)\n'
                    f'    at time {frame}f       (WM3_MC_SetValue m {ch} 100.0)\n'
                    f'    at time {end_frame}f   (WM3_MC_SetValue m {ch} 0.0)\n'
                    f'  )\n'
                    f')'
                )
                result = rt.execute(mxs)
                if result is False:
                    self.logger.error(f"Morph channel {ch} failed for frame {frame}")

                self._bake_progress.setValue(40 + int((fi + 1) / n_kf * 50))
                QtWidgets.QApplication.processEvents()

            # Step 4 — Delete temporary target meshes
            for tname in target_names:
                rt.execute(f'(local t = getNodeByName "{tname}"; if t != undefined do delete t)')

            QtCore.QTimer.singleShot(0, self._redraw_views)
            self._bake_progress.setValue(100)
            self.logger.info(
                f"Baked {n_kf} morph targets to timeline "
                f"(frames {start_frame}–{end_frame}). "
                f"Morpher modifier added to '{mesh_name}'.")

        except Exception as e:
            self.logger.error(f"Bake failed: {e}")
            for tname in target_names:
                try:
                    rt.execute(f'(local t = getNodeByName "{tname}"; if t != undefined do delete t)')
                except Exception:
                    pass
        finally:
            self._btn_bake.setEnabled(True)
            self._bake_progress.setVisible(False)

    # ─── Population logic ────────────────────────────────────────────────────

    def _on_generate_population(self):
        if not gnm_bridge.is_gnm_available():
            self.logger.error("GNM not loaded. Please reopen the tool.")
            return
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self.logger.warning("Generation already in progress.")
            return

        count = self._spin_count.value()
        cols = self._spin_cols.value()
        sx = self._spin_spacing_x.value()
        sy = self._spin_spacing_y.value()
        seed = self._spin_seed.value()

        self._btn_generate_grid.setEnabled(False)
        self._btn_generate_grid.setText("Generating...")
        self._pop_progress.setValue(0)
        self._pop_progress.setVisible(True)
        self._lbl_pop_status.setText(f"0 / {count}")

        self._batch_worker = BatchWorker(count, cols, sx, sy, seed, parent=self)
        self._batch_worker.sig_head.connect(self._on_batch_head)
        self._batch_worker.sig_progress.connect(self._on_batch_progress)
        self._batch_worker.sig_done.connect(self._on_batch_done)
        self._batch_worker.start()

    def _on_batch_head(self, vertices, triangles, triangle_uvs, name, pos, identity):
        gnm_bridge.create_max_mesh(
            vertices, triangles, triangle_uvs=triangle_uvs, name=name,
            identity=identity, position=pos, logger=self.logger
        )

    def _on_batch_progress(self, done, total):
        pct = int(done * 100 / total)
        self._pop_progress.setValue(pct)
        self._lbl_pop_status.setText(f"{done} / {total}")

    def _on_batch_done(self):
        self._btn_generate_grid.setEnabled(True)
        self._btn_generate_grid.setText("Generate Grid")
        self._pop_progress.setVisible(False)
        count = self._spin_count.value()
        self._lbl_pop_status.setText(f"Done — {count} heads created.")
        self.logger.info(f"Batch generation complete: {count} heads.")

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _check_mesh_valid(self) -> bool:
        if self._current_mesh is None:
            return False
        try:
            import pymxs
            return bool(pymxs.runtime.isValidNode(self._current_mesh))
        except Exception:
            return False

    def _set_update_status(self, text: str, color: str):
        self._lbl_update_status.setText(text)
        self._lbl_update_status.setStyleSheet(f"font-size: 11px; color: {color};")

    # ─── Settings menu ───────────────────────────────────────────────────────

    def _show_settings_menu(self):
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color:#2a2a2a; color:#dddddd; border:1px solid #555; }"
            "QMenu::item:selected { background-color:#005588; }"
        )
        act_path = menu.addAction("Reveal Install Path")
        menu.addSeparator()
        act_reinstall = menu.addAction("Reinstall")
        menu.addSeparator()
        act_about = menu.addAction("About")

        act_path.triggered.connect(self._on_reveal_install_path)
        act_reinstall.triggered.connect(self._on_reinstall)
        act_about.triggered.connect(self._on_show_about)

        menu.exec(self._btn_settings.mapToGlobal(self._btn_settings.rect().bottomLeft()))

    def _on_reveal_install_path(self):
        import subprocess
        path = setup_manager.get_install_path()
        from pathlib import Path
        p = Path(path)
        # Walk up until we find an existing directory to open
        while p and not p.exists():
            p = p.parent
        subprocess.Popen(f'explorer /select,"{path}"' if Path(path).exists()
                         else f'explorer "{p}"')

    def _on_show_about(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"About {constants.TOOL_NAME}")
        dlg.setFixedWidth(340)
        dlg.setStyleSheet(constants.STYLESHEET)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setSpacing(12)
        lay.setContentsMargins(20, 20, 20, 20)

        # Title + version
        lbl_title = QtWidgets.QLabel(constants.TOOL_NAME)
        lbl_title.setObjectName("lbl_title")
        lbl_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl_title)

        lbl_ver = QtWidgets.QLabel(f"Version {constants.VERSION}  ·  by {constants.AUTHOR}")
        lbl_ver.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_ver.setStyleSheet("color:#888; font-size:11px;")
        lay.addWidget(lbl_ver)

        lbl_desc = QtWidgets.QLabel(
            "Parametric 3D human head generator for 3ds Max,\n"
            "powered by Google's GNM model."
        )
        lbl_desc.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_desc.setStyleSheet("color:#aaa; font-size:11px;")
        lbl_desc.setWordWrap(True)
        lay.addWidget(lbl_desc)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color:#444;")
        lay.addWidget(sep)

        # GitHub button — #24292e (GitHub dark)
        btn_github = QtWidgets.QPushButton("  View on GitHub")
        btn_github.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_ComputerIcon))
        btn_github.setStyleSheet(
            "QPushButton { background:#24292e; color:#ffffff; border:1px solid #444; "
            "border-radius:5px; padding:7px 12px; font-size:12px; font-weight:bold; }"
            "QPushButton:hover { background:#2f363d; border-color:#58a6ff; }"
        )
        btn_github.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl("https://github.com/imanshirani/GNM-Bridge-for-3ds-Max/")
        ))
        lay.addWidget(btn_github)

        # Google GNM button — #4285F4 (Google blue)
        btn_gnm = QtWidgets.QPushButton("  Google GNM Model")
        btn_gnm.setStyleSheet(
            "QPushButton { background:#4285F4; color:#ffffff; border:none; "
            "border-radius:5px; padding:7px 12px; font-size:12px; font-weight:bold; }"
            "QPushButton:hover { background:#5a95f5; }"
        )
        btn_gnm.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl("https://github.com/google/GNM")
        ))
        lay.addWidget(btn_gnm)

        # PayPal donate button — #009cde (PayPal blue)
        btn_donate = QtWidgets.QPushButton("  Donate via PayPal")
        btn_donate.setStyleSheet(
            "QPushButton { background:#009cde; color:#ffffff; border:none; "
            "border-radius:5px; padding:7px 12px; font-size:12px; font-weight:bold; }"
            "QPushButton:hover { background:#00b2ff; }"
        )
        btn_donate.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl("https://www.paypal.com/donate/?hosted_button_id=LAMNRY6DDWDC4")
        ))
        lay.addWidget(btn_donate)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setFixedHeight(28)
        btn_close.clicked.connect(dlg.accept)
        lay.addWidget(btn_close)

        dlg.exec()

    def _on_reinstall(self):
        reply = QtWidgets.QMessageBox.question(
            self, "Reinstall",
            "The vendor folder and config will be removed.\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            setup_manager.reset_setup()
            gnm_bridge.reset_model()
            self._current_mesh = None
            self._progress_bar.setValue(0)
            self._progress_bar.setVisible(False)
            self._lbl_progress_text.setVisible(False)
            self._btn_setup.setText("Install && Setup GNM")
            self._btn_setup.setEnabled(True)
            self._show_setup_state()
            self.logger.info("Config cleared. You can reinstall now.")

    # ─── Log ─────────────────────────────────────────────────────────────────

    def _append_log(self, text: str, color: str):
        self._log_view.append(f'<span style="color:{color};">{text}</span>')
        sb = self._log_view.verticalScrollBar()
        sb.setValue(sb.maximum())
