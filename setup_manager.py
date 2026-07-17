"""Manages the one-time GNM download, extraction, and installation."""

import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from . import constants

_BASE_DIR = Path(__file__).parent
_VENDOR_DIR = _BASE_DIR / constants.GNM_VENDOR_DIR
_CONFIG_PATH = _BASE_DIR / constants.CONFIG_FILE

# numpy must be pinned to the correct major version for each Max release.
# It is installed LAST with --force-reinstall so scipy/trimesh cannot upgrade it.
#   Max 2025/2026 (Python ≤3.12): numpy 1.x — Max extensions use NumPy 1 ABI
#   Max 2027+     (Python 3.13+):  numpy 2.x — no 1.x wheels exist for Python 3.13
_NUMPY_REQ = "numpy<2" if sys.version_info < (3, 13) else "numpy>=2,<3"

# numpy is intentionally absent here — it is installed last (see install_deps)
REQUIRED_DEPS = [
    "h5py",
    "absl-py",
    "etils",
    "immutabledict",
    "importlib_resources",
    "opt-einsum",
    "rtree",
    "scipy",
    "trimesh",
    "tqdm",
    "typeguard",
    "opencv-python",
]


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_config(gnm_shape_path: str):
    data = {
        "gnm_installed": True,
        "gnm_path": str(gnm_shape_path),
    }
    _CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def check_status() -> tuple[bool, str]:
    """Returns (is_ready, gnm_shape_path_or_empty)."""
    cfg = load_config()
    if not cfg.get("gnm_installed"):
        return False, ""
    path = cfg.get("gnm_path", "")
    if path and Path(path).exists():
        return True, path
    return False, ""


# ─── Download ────────────────────────────────────────────────────────────────

def download_gnm(progress_cb=None) -> Path:
    """Download GNM zip from GitHub. Returns path to the zip file."""
    _VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    dest = _VENDOR_DIR / "gnm_repo.zip"

    def reporthook(count, block_size, total_size):
        if progress_cb and total_size > 0:
            pct = min(int(count * block_size * 100 / total_size), 100)
            progress_cb(pct, f"Downloading GNM... {pct}%")

    urllib.request.urlretrieve(constants.GNM_ZIP_URL, dest, reporthook)
    return dest


# ─── Extract ─────────────────────────────────────────────────────────────────

def extract_gnm(zip_path: Path, progress_cb=None) -> Path:
    """Extract zip and return path to the gnm/shape directory."""
    if progress_cb:
        progress_cb(0, "Extracting files...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(_VENDOR_DIR)

    # GitHub names the folder GNM-main (or GNM-<hash>)
    shape_path = next(_VENDOR_DIR.glob("GNM-*/gnm/shape"), None)
    if shape_path is None:
        raise FileNotFoundError("gnm/shape folder not found inside zip.")

    if progress_cb:
        progress_cb(100, "Extraction complete.")

    return shape_path


# ─── Install ─────────────────────────────────────────────────────────────────

def _ensure_pip(log_cb=None, progress_cb=None) -> bool:
    """Bootstrap pip via ensurepip if it is not already available."""
    check = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True, text=True, timeout=30,
    )
    if check.returncode == 0:
        return True  # pip already present

    if log_cb:
        log_cb("pip not found — bootstrapping...")
    if progress_cb:
        progress_cb(62, "Bootstrapping pip...")

    # Try ensurepip --user first (avoids needing write access to Program Files)
    result = subprocess.run(
        [sys.executable, "-m", "ensurepip", "--user", "--upgrade"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        # Fallback: ensurepip without --user (works if Max was run as admin)
        result = subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            capture_output=True, text=True, timeout=60,
        )
    if result.returncode != 0:
        if log_cb:
            log_cb(f"ensurepip failed: {result.stderr.strip()[:400]}")
        return False

    # Verify pip is now importable
    verify = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True, text=True, timeout=15,
    )
    if verify.returncode != 0:
        if log_cb:
            log_cb(f"pip still not available after ensurepip: {verify.stderr.strip()[:300]}")
        return False

    if log_cb:
        log_cb("pip ready.")
    return True


def install_deps(log_cb=None, progress_cb=None) -> bool:
    """Install required Python packages into Max's Python, one by one."""
    if log_cb:
        log_cb(f"Python executable: {sys.executable}")
        log_cb(f"Python version: {sys.version}")

    if not _ensure_pip(log_cb=log_cb, progress_cb=progress_cb):
        return False

    total = len(REQUIRED_DEPS)
    for i, pkg in enumerate(REQUIRED_DEPS, 1):
        if log_cb:
            log_cb(f"Installing ({i}/{total}): {pkg}")
        if progress_cb:
            pct = 62 + int((i - 1) / total * 23)  # 62% → 85%
            progress_cb(pct, f"Installing ({i}/{total}): {pkg}")
        cmd = [sys.executable, "-m", "pip", "install",
               "--user", "--timeout", "120", "--retries", "3"]
        # numpy needs --force-reinstall in case a wrong major version was previously installed
        if pkg.startswith("numpy"):
            cmd.append("--force-reinstall")
        cmd.append(pkg)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if log_cb and result.stdout.strip():
            log_cb(result.stdout.strip()[:600])
        if result.returncode != 0:
            if log_cb:
                log_cb(f"STDERR: {result.stderr.strip()[:800]}")
            return False

    # Install numpy LAST with --force-reinstall so scipy/trimesh cannot pull in
    # a conflicting version during their own installation above.
    if log_cb:
        log_cb(f"Installing numpy (pinned): {_NUMPY_REQ}")
    if progress_cb:
        progress_cb(83, f"Installing numpy (pinned)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "--user", "--force-reinstall", "--timeout", "120", _NUMPY_REQ],
        capture_output=True, text=True, timeout=600,
    )
    if log_cb and result.stdout.strip():
        log_cb(result.stdout.strip()[:600])
    if result.returncode != 0:
        if log_cb:
            log_cb(f"STDERR: {result.stderr.strip()[:800]}")
        return False

    if log_cb:
        log_cb("All packages installed.")
    return True


def register_gnm_path(shape_path: Path, log_cb=None) -> bool:
    """Add the GNM repo root to sys.path so 'import gnm.shape' works without pip.

    shape_path = .../vendor/GNM-main/gnm/shape
    gnm_root   = .../vendor/GNM-main          ← needs to be on sys.path
    """
    gnm_root = str(shape_path.parent.parent)
    if gnm_root not in sys.path:
        sys.path.insert(0, gnm_root)
    # Verify the import works
    try:
        import importlib
        importlib.invalidate_caches()
        # clear any stale cached failure
        for key in list(sys.modules.keys()):
            if key == "gnm" or key.startswith("gnm."):
                del sys.modules[key]
        import gnm.shape  # noqa: F401
        if log_cb:
            log_cb(f"GNM importable from: {gnm_root}")
        return True
    except ImportError as e:
        if log_cb:
            log_cb(f"GNM import failed after path registration: {e}")
        return False


# ─── Full Setup (called from SetupWorker) ────────────────────────────────────

def _find_existing_shape_path() -> "Path | None":
    """Return the gnm/shape path if already extracted in vendor dir, else None."""
    return next(_VENDOR_DIR.glob("GNM-*/gnm/shape"), None)


def run_full_setup(progress_cb=None, log_cb=None) -> tuple[bool, str]:
    """
    Complete setup: download -> extract -> install deps -> install GNM -> save config.
    Skips download/extract if the repo is already present (e.g. installed by another Max version).
    Returns (success, gnm_shape_path_or_error_msg).
    """
    try:
        # Step 1 & 2: reuse existing extraction if present
        shape_path = _find_existing_shape_path()
        if shape_path is not None:
            if log_cb:
                log_cb(f"GNM repo already present at: {shape_path} — skipping download.")
            if progress_cb:
                progress_cb(60, "GNM repo found, skipping download...")
        else:
            # Download (0-50%)
            def dl_progress(pct, label):
                if progress_cb:
                    progress_cb(int(pct * 0.5), label)
            zip_path = download_gnm(progress_cb=dl_progress)

            # Extract (50-60%)
            if progress_cb:
                progress_cb(50, "Extracting files...")
            shape_path = extract_gnm(zip_path)
            if progress_cb:
                progress_cb(60, "Extraction complete.")

        # Step 3: install deps into THIS Max version's Python (60-85%)
        if progress_cb:
            progress_cb(60, "Installing packages...")
        ok = install_deps(log_cb=log_cb, progress_cb=progress_cb)
        if not ok:
            return False, "Dependency installation failed."
        if progress_cb:
            progress_cb(85, "Packages installed.")

        # Step 4: register GNM repo on sys.path (no pip install needed)
        if progress_cb:
            progress_cb(90, "Registering GNM...")
        ok = register_gnm_path(shape_path, log_cb=log_cb)
        if not ok:
            return False, "GNM import failed after path registration."
        if progress_cb:
            progress_cb(98, "Installation complete.")

        # Step 5: save config
        save_config(str(shape_path))
        if progress_cb:
            progress_cb(100, "Setup completed successfully.")
        if log_cb:
            log_cb("Setup complete. Tool is ready.")

        return True, str(shape_path)

    except Exception as e:
        msg = f"Setup error: {e}"
        if log_cb:
            log_cb(msg)
        return False, msg


# ─── Reset ───────────────────────────────────────────────────────────────────

def reset_setup():
    """Remove vendor dir and config — triggers Setup State on next launch."""
    if _VENDOR_DIR.exists():
        shutil.rmtree(_VENDOR_DIR, ignore_errors=True)
    if _CONFIG_PATH.exists():
        _CONFIG_PATH.unlink(missing_ok=True)


def get_install_path() -> str:
    cfg = load_config()
    return cfg.get("gnm_path", "Not installed")
