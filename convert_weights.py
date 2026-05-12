import os
import urllib.request
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SRC = "https://raw.githubusercontent.com/Luthiraa/TALOS-V2/main/rtl/microgpt/weights_only.npy"
ORDER = (
    "wte", "wpe",
    "layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo",
    "layer0.mlp_fc1", "layer0.mlp_fc2",
    "lm_head",
)

os.makedirs(ASSETS, exist_ok=True)
npy = os.path.join(ASSETS, "weights_only.npy")
if not os.path.exists(npy):
    urllib.request.urlretrieve(SRC, npy)
w = np.load(npy, allow_pickle=True).item()
flat = np.concatenate([np.asarray(w[k], dtype=np.float32).reshape(-1) for k in ORDER])
flat.astype(np.float32).tofile(os.path.join(ASSETS, "weights_fp32.bin"))
print("ok")
