from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pathlib import Path

# ✅ 1. Always use POSIX-style forward slashes for HF local paths
src = Path(r"C:/Users/Naila/PycharmProjects/cyberbully-detector/outputs/final_cyberbully_model")
dst = Path(r"C:/Users/Naila/PycharmProjects/cyberbully-detector/outputs/final_cyberbully_model_fixed")

print("🔄 Loading model from:", src)

# ✅ 2. Convert to POSIX string
src_str = src.as_posix()
dst_str = dst.as_posix()

# ✅ 3. Load locally
model = AutoModelForSequenceClassification.from_pretrained(src_str, local_files_only=True)
tok = AutoTokenizer.from_pretrained(src_str, local_files_only=True)

# ✅ 4. Move to CPU and save properly
device = torch.device("cpu")
model.to(device)

# ✅ 5. Save fully materialized weights (creates pytorch_model.bin)
model.save_pretrained(dst_str)
tok.save_pretrained(dst_str)

print("✅ Fully materialized model saved to:", dst)
