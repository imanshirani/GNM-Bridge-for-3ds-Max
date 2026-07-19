"""
Standalone worker: called by gnm_bridge.run_wav2vec2() as a subprocess.
Usage: python wav2vec2_worker.py <wav_path> <cache_dir> <model_id>
Prints progress lines to stderr (prefix LOG:) and final JSON cues to stdout.
"""
import sys, os, json

def log(msg):
    print(f"LOG:{msg}", file=sys.stderr, flush=True)

wav_path  = sys.argv[1]
cache_dir = sys.argv[2]
model_id  = sys.argv[3]

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"]       = "1"
os.environ["CURL_CA_BUNDLE"]                   = ""
os.environ["REQUESTS_CA_BUNDLE"]               = ""
os.environ["SSL_CERT_FILE"]                    = ""
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass
try:
    import requests
    requests.packages.urllib3.disable_warnings()
    _orig = requests.Session.send
    def _no_verify(self, req, **kw):
        kw["verify"] = False
        return _orig(self, req, **kw)
    requests.Session.send = _no_verify
except Exception:
    pass

log("importing torch…")
try:
    import torch
except ImportError as e:
    log(f"torch import failed: {e}")
    sys.exit(1)
log(f"torch {torch.__version__} OK")

log("importing transformers…")
try:
    from transformers import Wav2Vec2ForCTC, AutoProcessor
except ImportError as e:
    log(f"transformers import failed: {e}")
    sys.exit(1)
log("transformers OK")

import pathlib, wave
import numpy as np

cache_path = pathlib.Path(cache_dir)
cache_path.mkdir(parents=True, exist_ok=True)

cached = any(cache_path.glob("models--facebook--wav2vec2*"))
if not cached:
    log("downloading model (~378 MB) — first time only, please wait…")
else:
    log("loading model from cache…")

try:
    log("loading processor…")
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    log("processor ready. Loading model weights…")
    model = Wav2Vec2ForCTC.from_pretrained(model_id, cache_dir=cache_dir)
    model.eval()
    log("model loaded.")
except Exception as e:
    log(f"model load failed: {e}")
    sys.exit(1)

# Read WAV (already 16kHz mono — gnm_bridge._prepare_wav handled it)
try:
    with wave.open(wav_path, "rb") as wf:
        sr  = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
except Exception as e:
    log(f"WAV read failed: {e}")
    sys.exit(1)

log(f"running inference on {len(audio)/sr:.1f}s audio…")
try:
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits[0], dim=-1).numpy()
except Exception as e:
    log(f"inference failed: {e}")
    sys.exit(1)

vocab = processor.tokenizer.convert_ids_to_tokens(range(probs.shape[1]))
frame_dur = 1.0 / 50.0

_C2P = {
    "|": "X", " ": "X",
    "a": "A",
    "b": "B", "p": "B", "m": "B",
    "i": "C",
    "e": "D",
    "o": "E", "u": "E",
    "f": "F", "v": "F", "s": "F", "z": "F",
    "t": "H", "d": "H", "l": "H", "n": "H",
    "r": "G", "w": "E", "y": "C",
    "k": "H", "g": "H", "h": "H",
    "j": "H", "x": "F", "q": "H",
}

frame_phonemes = []
for t in range(probs.shape[0]):
    best_id = int(probs[t].argmax())
    ch = vocab[best_id].lower().strip("<>[]") if best_id < len(vocab) else ""
    frame_phonemes.append(_C2P.get(ch, "X"))

cues = []
if frame_phonemes:
    cur = frame_phonemes[0]
    t_start = 0.0
    for i in range(1, len(frame_phonemes)):
        if frame_phonemes[i] != cur:
            cues.append({"start": t_start, "end": i * frame_dur, "value": cur})
            cur = frame_phonemes[i]
            t_start = i * frame_dur
    cues.append({"start": t_start,
                 "end": len(frame_phonemes) * frame_dur,
                 "value": cur})

log(f"done — {len(cues)} cues from {len(frame_phonemes)} frames.")

# Write result to the output file passed as argv[4]
out_path = sys.argv[4]
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cues, f)
log(f"result written to {out_path}")
