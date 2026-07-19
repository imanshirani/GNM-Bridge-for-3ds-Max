"""Bridge between Google GNM and 3ds Max."""

import subprocess
import sys
import importlib
from pathlib import Path


REQUIRED_DEPS = [
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

_gnm_model = None
_gender_vector = None        # None = not loaded yet, False = failed, tuple = loaded



def install_gnm(gnm_shape_path: str, logger=None) -> bool:
    """Install GNM dependencies into Max's Python. gnm_shape_path = path to GNM/gnm/shape/"""

    def log(msg):
        if logger:
            logger.info(msg)

    def log_err(msg):
        if logger:
            logger.error(msg)

    shape_path = Path(gnm_shape_path)
    if not shape_path.exists():
        log_err(f"Path not found: {gnm_shape_path}")
        return False

    log("Installing dependencies...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + REQUIRED_DEPS,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            log_err(f"Install error: {result.stderr[:500]}")
            return False
        log("Dependencies installed.")
    except subprocess.TimeoutExpired:
        log_err("Timeout while installing dependencies.")
        return False

    log("Installing GNM (no TensorFlow)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "-e", str(shape_path), "--no-deps"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            log_err(f"GNM install error: {result.stderr[:500]}")
            return False
        log("GNM installed successfully.")
        return True
    except subprocess.TimeoutExpired:
        log_err("Timeout while installing GNM.")
        return False


def _ensure_gnm_on_path():
    """Add the GNM repo root to sys.path if config records it."""
    try:
        from . import setup_manager
        cfg = setup_manager.load_config()
        gnm_path = cfg.get("gnm_path", "")
        if not gnm_path:
            return
        # shape_path = .../vendor/GNM-main/gnm/shape
        # gnm_root   = .../vendor/GNM-main
        gnm_root = str(Path(gnm_path).parent.parent)
        if gnm_root not in sys.path:
            sys.path.insert(0, gnm_root)
    except Exception:
        pass


def is_gnm_available() -> bool:
    _ensure_gnm_on_path()
    # Clear any previously failed import cache so Python retries the lookup
    for key in list(sys.modules.keys()):
        if key == "gnm" or key.startswith("gnm."):
            del sys.modules[key]
    importlib.invalidate_caches()
    try:
        import gnm.shape  # noqa: F401
        return True
    except ImportError:
        return False


def load_model(logger=None):
    global _gnm_model

    if _gnm_model is not None:
        return _gnm_model

    _ensure_gnm_on_path()

    if logger:
        logger.info("Loading GNM model...")

    try:
        from gnm.shape import gnm_numpy

        _gnm_model = gnm_numpy.GNM.from_local(
            version=gnm_numpy.GNMMajorVersion.V3,
            variant=gnm_numpy.GNMVariant.HEAD,
        )
        if logger:
            logger.info("GNM model loaded.")
        return _gnm_model
    except Exception as e:
        if logger:
            logger.error(f"Failed to load model: {e}")
        return None


def _load_identity_decoder():
    """Load identity_decoder_model.h5 weights and run inference with pure numpy.

    Architecture read from the h5 file:
      inputs: latent(64) + label(6)  →  concat(70)
      dense_4:  (70→64,  relu)
      dense_5:  (64→128, relu)
      dense_6:  (128→256, relu)
      dense_7:  (256→512, relu)
      dense_8:  (512→253, linear)   ← identity coefficients
    """
    import numpy as np
    import h5py
    from pathlib import Path
    import json

    cfg_path = Path(__file__).parent / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    gnm_shape_path = cfg.get("gnm_path", "")
    if not gnm_shape_path:
        raise FileNotFoundError("gnm_path not in config")

    h5_path = (Path(gnm_shape_path).parent.parent
               / "gnm" / "shape" / "data" / "semantic_sampler" / "identity_decoder_model.h5")

    layers = {}
    with h5py.File(str(h5_path), "r") as f:
        mw = f["model_weights"]
        for layer_name in mw.keys():
            try:
                inner = mw[layer_name][layer_name]
                if "kernel:0" not in inner or "bias:0" not in inner:
                    continue
                W = inner["kernel:0"][()]
                b = inner["bias:0"][()]
                layers[layer_name] = (W.astype(np.float32), b.astype(np.float32))
            except Exception:
                continue   # skip non-weight layers (Input, Concatenate, etc.)

    def relu(x):
        return np.maximum(0.0, x)

    def decode(latent, label):
        x = np.concatenate([latent, label], axis=-1)   # (70,)
        x = relu(x @ layers["dense_4"][0] + layers["dense_4"][1])
        x = relu(x @ layers["dense_5"][0] + layers["dense_5"][1])
        x = relu(x @ layers["dense_6"][0] + layers["dense_6"][1])
        x = relu(x @ layers["dense_7"][0] + layers["dense_7"][1])
        x =      x @ layers["dense_8"][0] + layers["dense_8"][1]
        return x  # (253,)

    return decode


def get_gender_vector(logger=None):
    """Return (male_avg, female_avg) as two (253,) arrays from the identity decoder.

    Uses pure numpy + h5py — no TensorFlow needed.
    Both arrays are the "average" face for each gender (zero latent vector).
    Cached after first call.  Returns (None, None) if decoder is unavailable.
    """
    global _gender_vector
    if _gender_vector is not None:
        return _gender_vector if _gender_vector is not False else None

    import numpy as np

    def _log(msg):
        if logger:
            logger.info(msg)

    try:
        decode = _load_identity_decoder()

        eth = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        z   = np.zeros(64, dtype=np.float32)   # zero latent = average face

        male_avg   = decode(z, np.array([0.0, 1.0, *eth], dtype=np.float32))
        female_avg = decode(z, np.array([1.0, 0.0, *eth], dtype=np.float32))

        direction = male_avg - female_avg
        norm = float(np.linalg.norm(direction))
        _log(f"Gender decoder loaded — norm={norm:.3f}")

        if norm < 1e-6:
            raise ValueError("Gender direction is zero")

        _gender_vector = (
            male_avg.astype(np.float32),
            female_avg.astype(np.float32),
            (direction / norm).astype(np.float32),
        )
        return _gender_vector

    except ImportError:
        _log("h5py not installed — gender slider unavailable")
    except Exception as e:
        _log(f"Gender decoder error: {e}")

    _gender_vector = False   # mark as permanently failed — don't retry
    return None


def generate_head(identity=None, expression=None, rotations=None, translation=None, logger=None):
    """Generate a head mesh. Returns (vertices [N,3], triangles [M,3]) or (None, None)."""
    import numpy as np

    model = load_model(logger=logger)
    if model is None:
        return None, None

    try:
        id_vec    = np.zeros((1, 253),  dtype=np.float32) if identity   is None else np.array(identity,    dtype=np.float32).reshape(1, 253)
        ex_vec    = np.zeros((1, 383),  dtype=np.float32) if expression is None else np.array(expression,  dtype=np.float32).reshape(1, 383)
        rot_vec   = np.zeros((1, 4, 3), dtype=np.float32) if rotations  is None else np.array(rotations,   dtype=np.float32).reshape(1, 4, 3)
        trans_vec = np.zeros((1, 3),    dtype=np.float32) if translation is None else np.array(translation, dtype=np.float32).reshape(1, 3)

        vertices = model(id_vec, ex_vec, rot_vec, trans_vec)
        v = np.array(vertices[0])  # [N, 3], Y-up

        # Y-up -> Z-up (3ds Max convention) + scale to cm
        from . import constants
        s = constants.GNM_MESH_SCALE
        out = np.empty_like(v)
        out[:, 0] =  v[:, 0] * s   # X -> X
        out[:, 1] = -v[:, 2] * s   # Z -> -Y
        out[:, 2] =  v[:, 1] * s   # Y -> Z
        vertices = out

        triangles = np.array(model.triangles)  # [M, 3], 0-based
        triangle_uvs = np.array(model.triangle_uvs)  # (T, 3, 2)
        face_mat_ids = _get_face_mat_ids(model, triangles)  # (T,) int, 0=skin

        if logger:
            logger.info(f"Mesh generated: {len(vertices)} verts, {len(triangles)} faces")

        return vertices, triangles, triangle_uvs, face_mat_ids

    except Exception as e:
        if logger:
            logger.error(f"Generate error: {e}")
        return None, None, None, None


_face_mat_ids_cache = None  # cached per model load — topology never changes

def _get_face_mat_ids(model, triangles):
    """Return per-face material ID array (0=skin, 1=upper_teeth, 2=lower_teeth, 3=tongue, 4=left_eye, 5=right_eye).

    Cached after first call since topology is constant.
    """
    global _face_mat_ids_cache
    import numpy as np

    if _face_mat_ids_cache is not None:
        return _face_mat_ids_cache

    try:
        vg_names = list(model.vertex_group_names)
        vg = np.array(model.vertex_groups)  # (46, N_verts)

        comp_vg = [
            vg_names.index('skin'),
            vg_names.index('upper_teeth_and_gums'),
            vg_names.index('lower_teeth_and_gums'),
            vg_names.index('tongue'),
            vg_names.index('left_eye'),
            vg_names.index('right_eye'),
        ]

        n_faces = len(triangles)
        ids = np.zeros(n_faces, dtype=np.int32)  # default = 0 (skin)

        for fi in range(n_faces):
            v0 = triangles[fi, 0]
            for mat_id, vg_idx in enumerate(comp_vg):
                if vg[vg_idx, v0] > 0.5:
                    ids[fi] = mat_id
                    break

        _face_mat_ids_cache = ids
        return ids
    except Exception:
        return np.zeros(len(triangles), dtype=np.int32)


def create_max_mesh(vertices, triangles, triangle_uvs=None, face_mat_ids=None,
                    name="GNM_Head", identity=None, position=None, logger=None):
    """Create an Editable Mesh object in 3ds Max with UV mapping and material IDs."""
    try:
        import pymxs
        rt = pymxs.runtime

        n_verts = len(vertices)
        n_faces = len(triangles)

        mxs_cmd = f'm = mesh numverts:{n_verts} numfaces:{n_faces}; m.name = "{name}"; m'

        with pymxs.undo(True):
            mesh_obj = rt.execute(mxs_cmd)

            for i, (x, y, z) in enumerate(vertices.tolist()):
                rt.setVert(mesh_obj, i + 1, rt.point3(float(x), float(y), float(z)))

            for i, (a, b, c) in enumerate(triangles.tolist()):
                rt.setFace(mesh_obj, i + 1, int(a) + 1, int(b) + 1, int(c) + 1)

            # Set face material IDs (1-based in Max)
            if face_mat_ids is not None:
                for i, mat_id in enumerate(face_mat_ids.tolist()):
                    rt.setFaceMatID(mesh_obj, i + 1, int(mat_id) + 1)

            rt.update(mesh_obj)

            if position is not None:
                mesh_obj.pos = rt.point3(float(position[0]), float(position[1]), float(position[2]))

            rt.select(mesh_obj)

        if identity is not None:
            save_identity_to_mesh(mesh_obj, identity)

        # UV mapping
        if triangle_uvs is not None:
            try:
                _apply_uvw(rt, mesh_obj, triangles, triangle_uvs)
            except Exception as uv_err:
                if logger:
                    logger.warning(f"UV mapping skipped: {uv_err}")


        # Multi/Sub-Object material — skin gets texture, rest get blank slots
        try:
            _apply_gnm_material(rt, mesh_obj, logger=logger)
        except Exception as mat_err:
            if logger:
                logger.warning(f"Material skipped: {mat_err}")

        if logger:
            logger.info(f'Mesh "{name}" created in scene.')

        return mesh_obj

    except Exception as e:
        if logger:
            logger.error(f"Failed to create mesh: {e}")
        return None


def _apply_uvw(rt, mesh_obj, triangles, triangle_uvs):
    """Apply per-triangle UV coordinates using the pymxs Python API (no MAXScript strings).

    GNM stores triangle_uvs as per-face-corner (T, 3, 2).  Adjacent faces that
    share a seam-free edge have identical UV coords at that edge.  We deduplicate
    those shared UV verts so 3ds Max builds proper UV islands with connected edges,
    matching the structured GNM UV layout.
    """
    import numpy as np

    n_faces  = len(triangles)
    uvs_flat = triangle_uvs.reshape(-1, 2)   # (T*3, 2)

    # Merge identical UV verts — scale to int for exact comparison
    uvs_int = np.round(uvs_flat * 1_000_000).astype(np.int64)
    packed  = uvs_int[:, 0] * 2_000_001 + uvs_int[:, 1]
    _, unique_idx, inverse_idx = np.unique(packed, return_index=True, return_inverse=True)

    unique_uvs  = uvs_flat[unique_idx]        # (U, 2) deduplicated
    n_unique    = len(unique_uvs)
    face_uv_idx = inverse_idx.reshape(n_faces, 3)  # (T, 3) 0-indexed

    # Allocate map channel 1
    rt.meshop.setNumMaps(mesh_obj, 2)
    rt.meshop.setMapSupport(mesh_obj, 1, True)
    rt.meshop.setNumMapVerts(mesh_obj, 1, n_unique)
    rt.meshop.setNumMapFaces(mesh_obj, 1, n_faces)

    # Write UV verts — GNM UV origin matches Max (V=0 at bottom), no flip needed
    for i, (u, v) in enumerate(unique_uvs.tolist()):
        rt.meshop.setMapVert(mesh_obj, 1, i + 1, rt.point3(float(u), float(v), 0.0))

    # Write UV faces with merged indices
    for i in range(n_faces):
        a = int(face_uv_idx[i, 0]) + 1
        b = int(face_uv_idx[i, 1]) + 1
        c = int(face_uv_idx[i, 2]) + 1
        rt.meshop.setMapFace(mesh_obj, 1, i + 1, rt.point3(a, b, c))

    rt.update(mesh_obj)


def _apply_gnm_material(rt, mesh_obj, logger=None):
    """Assign a Multi/Sub-Object material to mesh_obj.

    Slot 1 (MatID 1) = skin with GNM edgeflow texture
    Slots 2-6 = blank Standard materials (teeth/tongue/eye get no texture)
    Reuses existing 'GNM_Head_Mat' if already in scene.
    """
    from pathlib import Path

    mat_name = "GNM_Head_Mat"
    existing = rt.execute(f'(sceneMaterials["{mat_name}"])')
    if existing is not None:
        try:
            mesh_obj.material = existing
            return
        except Exception:
            pass

    tex_path = (Path(__file__).parent / "vendor" / "GNM-main" / "gnm" / "shape"
                / "data" / "textures" / "edgeflow_bw_4k.png")
    tex_str = str(tex_path).replace("\\", "/") if tex_path.exists() else ""

    # Build Multi/Sub-Object with 6 slots
    mxs = (
        f'(\n'
        f'  local msm = MultiMaterial numsubs:6\n'
        f'  msm.name = "{mat_name}"\n'
        f'  -- Slot 1: Skin with texture\n'
        f'  local skin = StandardMaterial()\n'
        f'  skin.name = "GNM_Skin"\n'
    )
    if tex_str:
        mxs += (
            f'  local bm = Bitmaptexture()\n'
            f'  bm.filename = @"{tex_str}"\n'
            f'  skin.diffusemap = bm\n'
            f'  skin.showInViewport = true\n'
        )
    mxs += (
        f'  msm.materialList[1] = skin\n'
        f'  -- Slots 2-6: blank materials for teeth/tongue/eyes\n'
        f'  local names = #("Teeth_Upper","Teeth_Lower","Tongue","Eye_Left","Eye_Right")\n'
        f'  for i = 2 to 6 do (\n'
        f'    local s = StandardMaterial()\n'
        f'    s.name = names[i-1]\n'
        f'    msm.materialList[i] = s\n'
        f'  )\n'
        f'  msm\n'
        f')'
    )
    mat = rt.execute(mxs)
    if mat is not None:
        mesh_obj.material = mat


_GNM_IDENTITY_PROP = "gnm_identity"


def save_identity_to_mesh(mesh_obj, identity):
    """Store the identity vector as a user property on the mesh node."""
    import pymxs
    import json
    rt = pymxs.runtime
    data = json.dumps(identity.tolist())
    rt.setUserProp(mesh_obj, _GNM_IDENTITY_PROP, data)


def load_identity_from_mesh(mesh_obj):
    """Read back the identity vector from a mesh's user properties. Returns numpy array or None."""
    import pymxs
    import json
    import numpy as np
    rt = pymxs.runtime
    raw = rt.getUserProp(mesh_obj, _GNM_IDENTITY_PROP)
    if raw is None:
        return None
    try:
        return np.array(json.loads(str(raw)), dtype=np.float32)
    except Exception:
        return None


def is_gnm_mesh(mesh_obj) -> bool:
    """Return True if this node was created by GNM (has identity property)."""
    try:
        import pymxs
        rt = pymxs.runtime
        return rt.getUserProp(mesh_obj, _GNM_IDENTITY_PROP) is not None
    except Exception:
        return False


def update_max_mesh_vertices(mesh_obj, vertices, logger=None):
    """Update vertex positions of an existing Editable Mesh.

    Does NOT call redrawViews() — caller is responsible for scheduling a redraw
    outside Max's notification stack to avoid DirtyNotificationEventMonitor errors.
    """
    try:
        import pymxs
        rt = pymxs.runtime

        for i, (x, y, z) in enumerate(vertices.tolist()):
            rt.setVert(mesh_obj, i + 1, rt.point3(float(x), float(y), float(z)))

        rt.update(mesh_obj)
    except Exception as e:
        if logger:
            logger.error(f"Failed to update mesh vertices: {e}")


def interpolate_and_update(mesh_obj, identity, expression, rotations, logger=None):
    """Generate vertices for interpolated params and update an existing mesh.

    Called from the MAXScript timeChangeCallback on every frame scrub.
    Does NOT call redrawViews() — caller must schedule it outside Max's
    notification stack (e.g. via QTimer.singleShot) to avoid
    DirtyNotificationEventMonitor errors.
    """
    vertices, _, _, _ = generate_head(
        identity=identity, expression=expression, rotations=rotations, logger=logger)
    if vertices is not None:
        update_max_mesh_vertices(mesh_obj, vertices, logger=logger)


def _prepare_wav(src_path: str, logger=None) -> str:
    """Convert a WAV to 16-bit PCM mono 16kHz if needed. Returns path to use.

    Rhubarb requires 16-bit PCM mono. Stereo, 24-bit, or 32-bit files fail.
    Uses only stdlib (wave + array) — no extra packages needed.
    Returns src_path unchanged if already compatible or conversion fails.
    """
    import wave
    import array
    import tempfile
    import os

    try:
        with wave.open(src_path, "rb") as wf:
            ch       = wf.getnchannels()
            sampw    = wf.getsampwidth()  # bytes per sample: 1=8bit, 2=16bit, 4=32bit
            rate     = wf.getframerate()
            n_frames = wf.getnframes()
            raw      = wf.readframes(n_frames)

        needs_convert = (ch != 1 or sampw != 2 or rate != 16000)
        if not needs_convert:
            return src_path  # already mono 16-bit 16kHz — no conversion needed

        if logger:
            logger.info(f"WAV: {ch}ch, {sampw*8}-bit, {rate}Hz — converting to mono 16-bit 16kHz...")

        # Decode samples
        if sampw == 1:
            # 8-bit unsigned → signed 16-bit
            samples = array.array("B", raw)
            samples16 = array.array("h", ((s - 128) * 256 for s in samples))
        elif sampw == 2:
            samples16 = array.array("h", raw)
        elif sampw == 4:
            # 32-bit signed → 16-bit (drop lower 2 bytes)
            samples32 = array.array("i", raw)
            samples16 = array.array("h", (s >> 16 for s in samples32))
        else:
            if logger:
                logger.warning(f"Unsupported sample width {sampw} — passing original to Rhubarb")
            return src_path

        # Mix down to mono if stereo
        if ch == 2:
            mono = array.array("h")
            for i in range(0, len(samples16), 2):
                mono.append((samples16[i] + samples16[i + 1]) // 2)
            samples16 = mono
        elif ch > 2:
            # take first channel
            step_ch = array.array("h")
            for i in range(0, len(samples16), ch):
                step_ch.append(samples16[i])
            samples16 = step_ch

        # Resample to exactly 16kHz (handles both upsample and downsample)
        out_rate = 16000
        if rate != 16000:
            import numpy as np
            src_arr = np.frombuffer(samples16.tobytes(), dtype=np.int16).astype(np.float32)
            new_len = int(len(src_arr) * 16000 / rate)
            x_old = np.linspace(0, 1, len(src_arr))
            x_new = np.linspace(0, 1, new_len)
            resampled = np.interp(x_new, x_old, src_arr).astype(np.int16)
            samples16 = array.array("h", resampled.tobytes())
            if logger:
                logger.info(f"Resampled {rate}Hz → 16000Hz ({new_len} samples)")

        # Write to a temp file
        tmp = tempfile.NamedTemporaryFile(suffix="_gnm_mono16.wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        with wave.open(tmp_path, "wb") as wo:
            wo.setnchannels(1)
            wo.setsampwidth(2)
            wo.setframerate(out_rate)
            wo.writeframes(samples16.tobytes())

        if logger:
            logger.info(f"Converted WAV saved to temp: {os.path.basename(tmp_path)}")
        return tmp_path

    except Exception as e:
        if logger:
            logger.warning(f"WAV conversion failed ({e}) — passing original to Rhubarb")
        return src_path


def run_wav2vec2(wav_path: str, logger=None) -> "list[dict] | None":
    """Run Facebook Wav2Vec2 on a WAV file and return phoneme cues.

    Each cue: {"start": float, "end": float, "value": Preston_Blair_code}
    Same format as run_rhubarb — compatible with _apply_lipsync_keyframes.
    Runs torch/transformers in a subprocess so 3ds Max never imports torch directly.
    Auto-installs transformers+torch on first use.
    Auto-downloads model (~378 MB) on first use, cached in vendor/wav2vec2/.
    """
    import subprocess, sys, json, importlib, importlib.util
    from pathlib import Path
    from . import constants as _c

    base_dir  = Path(__file__).parent
    cache_dir = str(base_dir / _c.WAV2VEC2_CACHE_DIR)
    worker    = base_dir / "wav2vec2_worker.py"

    # Auto-install transformers/torch if missing (find_spec = no import, no hang)
    for pkg, idx_url in [
        ("transformers", None),
        ("torch", "https://download.pytorch.org/whl/cpu"),
    ]:
        if importlib.util.find_spec(pkg) is None:
            if logger:
                logger.info(f"{pkg} not found — installing (may take a few minutes)…")
            cmd = [sys.executable, "-m", "pip", "install", "--user"]
            if idx_url:
                cmd += ["--index-url", idx_url]
            cmd.append(pkg)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                if logger:
                    logger.error(f"pip install {pkg} failed:\n{r.stderr.strip()[:600]}")
                return None
            if logger:
                logger.info(f"{pkg} installed.")
            importlib.invalidate_caches()

    # Resample WAV to 16kHz mono before passing to worker
    try:
        prepared = _prepare_wav(wav_path, logger=logger)
    except Exception as e:
        if logger:
            logger.error(f"Wav2Vec2: WAV prepare failed: {e}")
        return None

    if logger:
        logger.info("Wav2Vec2: starting worker process (torch loads here)…")

    import tempfile
    result_file = tempfile.NamedTemporaryFile(suffix="_gnm_w2v.json", delete=False)
    result_path = result_file.name
    result_file.close()

    # Build env with user site-packages on PYTHONPATH so --user installs are visible
    import os, site
    worker_env = os.environ.copy()
    extra_paths = []
    try:
        extra_paths.append(site.getusersitepackages())
    except Exception:
        pass
    try:
        extra_paths.extend(site.getsitepackages())
    except Exception:
        pass
    existing = worker_env.get("PYTHONPATH", "")
    all_paths = extra_paths + ([existing] if existing else [])
    worker_env["PYTHONPATH"] = os.pathsep.join(all_paths)
    # Pass current sys.path entries too
    worker_env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + all_paths
    )

    try:
        proc = subprocess.Popen(
            [sys.executable, str(worker), prepared, cache_dir, _c.WAV2VEC2_MODEL, result_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=worker_env,
        )

        import threading

        stderr_lines = []

        def _read_stderr():
            for line in proc.stderr:
                line = line.rstrip()
                stderr_lines.append(line)
                if line.startswith("LOG:") and logger:
                    logger.info("Wav2Vec2: " + line[4:])
                elif line and logger:
                    logger.info("Wav2Vec2 worker: " + line)

        t_err = threading.Thread(target=_read_stderr, daemon=True)
        t_err.start()
        proc.wait(timeout=600)
        t_err.join(timeout=5)

        if prepared != wav_path:
            import os; os.unlink(prepared)

        if proc.returncode != 0:
            if logger:
                last = [l for l in stderr_lines if l][-3:] if stderr_lines else []
                logger.error(
                    f"Wav2Vec2 worker failed (code {proc.returncode}). "
                    f"Last output: {' | '.join(last)}")
            import os; os.unlink(result_path)
            return None

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                cues = json.load(f)
            import os; os.unlink(result_path)
        except Exception as e:
            if logger:
                logger.error(f"Wav2Vec2: failed to read result file: {e}")
            return None

        if logger:
            logger.info(f"Wav2Vec2: {len(cues)} phoneme cues generated.")
        return cues

    except subprocess.TimeoutExpired:
        proc.kill()
        if logger:
            logger.error("Wav2Vec2 worker timed out (10 min limit).")
        return None
    except Exception as e:
        if logger:
            logger.error(f"Wav2Vec2 subprocess error: {e}")
        return None


def run_rhubarb(wav_path: str, rhubarb_exe: str, logger=None) -> "list[dict] | None":
    """Run rhubarb.exe on a WAV file and return the mouthCues list.

    Each cue: {"start": float, "end": float, "value": str}
    Returns None on failure. Auto-converts stereo/24-bit WAV to mono 16-bit.
    """
    import json as _json
    import subprocess as _sub
    import os as _os

    tmp_path = None
    try:
        prepared = _prepare_wav(wav_path, logger=logger)
        if prepared != wav_path:
            tmp_path = prepared  # track for cleanup

        cmd = [rhubarb_exe, "-r", "phonetic", "-f", "json", prepared]
        if logger:
            logger.info(f"Rhubarb cmd: {' '.join(cmd)}")
        result = _sub.run(cmd, capture_output=True, text=True, timeout=300)
        if logger:
            logger.info(f"Rhubarb returncode: {result.returncode}")
            if result.stderr.strip():
                logger.info(f"Rhubarb stderr: {result.stderr.strip()}")
        if result.returncode != 0:
            if logger:
                logger.error(f"Rhubarb failed (see stderr above)")
            return None
        data = _json.loads(result.stdout)
        cues = data.get("mouthCues", [])
        if logger:
            logger.info(f"Rhubarb: {len(cues)} phoneme cues from {_os.path.basename(wav_path)}")
        return cues
    except Exception as e:
        if logger:
            logger.error(f"run_rhubarb failed: {e}")
        return None
    finally:
        if tmp_path and _os.path.exists(tmp_path):
            try:
                _os.unlink(tmp_path)
            except Exception:
                pass


def decode_phoneme_expressions(logger=None) -> "dict | None":
    """Generate phoneme→expression mapping using the GNM expression decoder.

    Uses expression_decoder_model.h5 (pure numpy, no TensorFlow).
    Returns dict of phoneme_code → list[float] (150 values, dims 200-349).
    Returns None if decoder unavailable.

    6-dim label meaning (inferred from activation analysis):
      [0] brow/upper face
      [1] lip spread / smile
      [2] lip round / pucker
      [3] jaw open + tongue
      [4] lips compress / close
      [5] eyes + tongue
    """
    import numpy as np
    import h5py
    import json
    from pathlib import Path

    cfg_path = Path(__file__).parent / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    gnm_shape_path = cfg.get("gnm_path", "")
    if not gnm_shape_path:
        return None

    h5_path = (Path(gnm_shape_path).parent.parent
               / "gnm" / "shape" / "data" / "semantic_sampler"
               / "expression_decoder_model.h5")
    if not h5_path.exists():
        return None

    try:
        layers = {}
        with h5py.File(str(h5_path), "r") as f:
            mw = f["model_weights"]
            for ln in sorted(mw.keys()):
                try:
                    inner = mw[ln][ln]
                    if "kernel:0" in inner:
                        layers[ln] = (
                            inner["kernel:0"][()].astype(np.float32),
                            inner["bias:0"][()].astype(np.float32),
                        )
                except Exception:
                    pass

        def relu(x): return np.maximum(0.0, x)

        def decode(latent_78, label_6):
            x = np.concatenate([latent_78, label_6])
            x = relu(x @ layers["dense_13"][0] + layers["dense_13"][1])
            x = relu(x @ layers["dense_14"][0] + layers["dense_14"][1])
            x = relu(x @ layers["dense_15"][0] + layers["dense_15"][1])
            x = relu(x @ layers["dense_16"][0] + layers["dense_16"][1])
            x =      x @ layers["dense_17"][0] + layers["dense_17"][1]
            return x  # (383,)

        z = np.zeros(78, dtype=np.float32)

        # Phoneme → 6-dim label weights (multi-dim combinations for richer shapes)
        # label: [brow, lip_spread, lip_round, jaw_open, lip_close, eye_tongue]
        phoneme_labels = {
            "X": [0.0, 0.0,  0.0,  0.0,  0.5, 0.0],   # rest — lips lightly closed
            "A": [0.0, 0.2,  0.0,  1.8,  0.0, 0.0],   # ah — jaw open + slight spread
            "B": [0.0, 0.0,  0.0, -0.3,  2.0, 0.0],   # m/b/p — lips pressed, jaw up
            "C": [0.0, 2.0,  0.0,  0.6,  0.0, 0.0],   # ee — spread + slight open
            "D": [0.0, 0.8,  0.0,  1.4,  0.0, 0.0],   # eh — moderate spread + open
            "E": [0.0,-0.3,  2.0,  1.2,  0.0, 0.0],   # oh — round + moderate open
            "F": [0.0, 0.0,  0.0,  0.4,  1.0, 0.0],   # f/v — slight open + close
            "G": [0.0, 0.3,  0.0,  0.8,  0.0, 0.8],   # th — slight open + tongue
            "H": [0.0, 0.3,  0.0,  0.6,  0.0, 0.6],   # l/n/d — moderate open + tongue
        }

        result = {}
        for code, lw in phoneme_labels.items():
            label = np.array(lw, dtype=np.float32)
            expr = decode(z, label)   # (383,)
            result[code] = expr[200:350].tolist()  # 150 lower_face dims

        if logger:
            logger.info(f"Expression decoder: generated {len(result)} phoneme vectors.")
        return result

    except Exception as e:
        if logger:
            logger.warning(f"Expression decoder failed: {e}")
        return None


def regenerate_phoneme_calibration(output_path=None, logger=None) -> bool:
    """Run expression decoder and write full 150-dim phoneme vectors to JSON.

    Returns True on success, False if decoder unavailable.
    """
    import json
    from pathlib import Path

    result = decode_phoneme_expressions(logger=logger)
    if not result:
        return False

    path = output_path or (Path(__file__).parent / "phoneme_calibration.json")

    analysis = {}
    for code, vals in result.items():
        top = sorted(enumerate(vals), key=lambda x: abs(x[1]), reverse=True)[:10]
        analysis[code] = {f"dim{200+i}": round(v, 3) for i, v in top if abs(v) > 0.01}

    data = {
        "_description": "GNM phoneme calibration - generated by expression decoder",
        "_analysis": analysis,
        "phonemes": result,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    if logger:
        logger.info(f"Phoneme calibration regenerated: {path}")
    return True


def reset_model():
    global _gnm_model, _gender_vector
    _gnm_model = None
    _gender_vector = None   # reset so next load retries
