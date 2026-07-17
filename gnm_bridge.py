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
        # triangle_uvs shape: (T, 3, 2) — UV per vertex per triangle
        triangle_uvs = np.array(model.triangle_uvs)

        if logger:
            logger.info(f"Mesh generated: {len(vertices)} verts, {len(triangles)} faces")

        return vertices, triangles, triangle_uvs

    except Exception as e:
        if logger:
            logger.error(f"Generate error: {e}")
        return None, None


def create_max_mesh(vertices, triangles, triangle_uvs=None, name="GNM_Head",
                    identity=None, position=None, logger=None):
    """Create an Editable Mesh object in 3ds Max with optional UV mapping."""
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

            rt.update(mesh_obj)

            if position is not None:
                mesh_obj.pos = rt.point3(float(position[0]), float(position[1]), float(position[2]))

            rt.select(mesh_obj)

        rt.redrawViews()

        if identity is not None:
            save_identity_to_mesh(mesh_obj, identity)

        # UV is applied last and errors are non-fatal — mesh is already in scene
        if triangle_uvs is not None:
            try:
                _apply_uvw(rt, mesh_obj, triangles, triangle_uvs)
            except Exception as uv_err:
                if logger:
                    logger.warning(f"UV mapping skipped: {uv_err}")

        if logger:
            logger.info(f'Mesh "{name}" created in scene.')

        return mesh_obj

    except Exception as e:
        if logger:
            logger.error(f"Failed to create mesh: {e}")
        return None


def _apply_uvw(rt, mesh_obj, triangles, triangle_uvs):
    """Apply per-triangle UV coordinates using the pymxs Python API (no MAXScript strings)."""
    import numpy as np

    n_faces = len(triangles)
    uvs_flat = triangle_uvs.reshape(-1, 2)   # (T*3, 2) — one UV per tri-corner
    n_tverts = len(uvs_flat)

    # Allocate map channel
    rt.meshop.setNumMaps(mesh_obj, 2)
    rt.meshop.setMapSupport(mesh_obj, 1, True)
    rt.meshop.setNumMapVerts(mesh_obj, 1, n_tverts)
    rt.meshop.setNumMapFaces(mesh_obj, 1, n_faces)

    # Write UV verts  (V-flip: Max uses 0=bottom, GNM/OpenGL uses 0=top)
    uv_list = uvs_flat.tolist()
    for i, (u, v) in enumerate(uv_list):
        rt.meshop.setMapVert(mesh_obj, 1, i + 1, rt.point3(float(u), float(1.0 - v), 0.0))

    # Write UV faces — each tri-face i maps to corners i*3+1 .. i*3+3
    for i in range(n_faces):
        a, b, c = i * 3 + 1, i * 3 + 2, i * 3 + 3
        rt.meshop.setMapFace(mesh_obj, 1, i + 1, rt.point3(a, b, c))

    rt.update(mesh_obj)


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

    GNM outputs vertices in object-local space and setVert works in object-local
    space, so the node transform (position/rotation/pivot) is untouched.
    """
    try:
        import pymxs
        rt = pymxs.runtime

        for i, (x, y, z) in enumerate(vertices.tolist()):
            rt.setVert(mesh_obj, i + 1, rt.point3(float(x), float(y), float(z)))

        rt.update(mesh_obj)
        rt.redrawViews()
    except Exception as e:
        if logger:
            logger.error(f"Failed to update mesh vertices: {e}")


def reset_model():
    global _gnm_model, _gender_vector
    _gnm_model = None
    _gender_vector = None   # reset so next load retries
