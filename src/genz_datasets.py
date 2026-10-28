from datasets import load_dataset
import pandas as pd, random, os

# Load the slang dictionary dataset
ds = load_dataset("MLBtrio/genz-slang-dataset")
df = pd.DataFrame(ds["train"])

# Define your manual positive/negative lists
negative = ["L", "ratio", "L+ratio", "clown", "cringe", "npc", "go touch grass", "mid"]
positive = ["W", "bet", "drip", "slay", "fire", "valid", "based", "cap"]

rows = []

# Generate simple example sentences automatically
for _, r in df.iterrows():
    slang = str(r["Slang"]).lower()
    desc = str(r["Description"])
    example = str(r["Example"])

    if any(bad in slang for bad in negative):
        rows.append({"text": f"{example} ({slang})", "label": 1})
    elif any(good in slang for good in positive):
        rows.append({"text": f"{example} ({slang})", "label": 0})
    else:
        # random neutral filler (optional)
        rows.append({"text": f"{example} ({slang})", "label": 0})

# Save
out = pd.DataFrame(rows)
os.makedirs("data/raw", exist_ok=True)
out.to_csv("data/raw/genz_dataset.csv", index=False, encoding="utf-8")
print(f"✅ saved {len(out)} rows to data/raw/genZ_datasets.csv")
