"""Build the 8-bit quantized Parakeet used by the app.

Run once (re-run only to change bits/model):

    /Users/clouds/anaconda3/python.app/Contents/MacOS/python tools/build_quantized.py

Produces ~/.local/groq-whisper-app/parakeet-8bit/{config.json,model.safetensors}.
8-bit is byte-identical to bf16 on our dictation clips but ~734MB vs ~1.2GB in
memory, so the RAM-tight 8GB M1 reloads it faster after the OS evicts it under
swap pressure. _get_parakeet() loads this dir (falling back to the HF bf16 repo
if it's absent).
"""
import json
import sys
import types
from pathlib import Path

# numba stub — parakeet_mlx -> librosa does `from numba import jit`, and this
# env's numba is broken. Same trick the app uses.
_stub = types.ModuleType("numba")
_stub.jit = _stub.njit = lambda *a, **k: (a[0] if a and callable(a[0]) and not k
                                          else (lambda fn: fn))
sys.modules["numba"] = _stub

import mlx.core as mx
import mlx.nn as nn
import parakeet_mlx
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten

REPO = "mlx-community/parakeet-tdt-0.6b-v2"
GROUP_SIZE, BITS = 64, 8
OUT = Path.home() / ".local/groq-whisper-app/parakeet-8bit"
OUT.mkdir(parents=True, exist_ok=True)

config = json.load(open(hf_hub_download(REPO, "config.json")))
model = parakeet_mlx.from_pretrained(REPO)
nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)

mx.save_safetensors(str(OUT / "model.safetensors"),
                    dict(tree_flatten(model.parameters())))
config["quantization"] = {"group_size": GROUP_SIZE, "bits": BITS}
json.dump(config, open(OUT / "config.json", "w"))

size_mb = (OUT / "model.safetensors").stat().st_size / 1e6
print(f"Wrote {OUT} ({size_mb:.0f} MB)")
