import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os
import re

# ---- SAME DATA LOADING LOGIC ----
def load_dirty_csv(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    text, label = [], []
    for row in lines:
        m = re.match(r'^(.*?),(0|1)$', row.strip())
        if m:
            text.append(m.group(1).strip())
            label.append(int(m.group(2)))

    return pd.DataFrame({"text": text, "label": label})


def load_all_datasets():
    DATA_DIR = "data/raw"
    dfs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            dfs.append(load_dirty_csv(os.path.join(DATA_DIR, file)))
    return pd.concat(dfs, ignore_index=True)


# ---- LOAD DATA ----
df = load_all_datasets().sample(frac=1, random_state=42).reset_index(drop=True)
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=42)

# ---- TOKENIZER ----
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

tokenized_ds = dataset.map(tokenize, batched=True)
tokenized_ds = tokenized_ds.rename_column("label", "labels")
tokenized_ds.set_format("torch")

# ---- LOAD TRAINED MODEL (NO TRAINING) ----
model = AutoModelForSequenceClassification.from_pretrained("outputs/final_model")

trainer = Trainer(model=model, tokenizer=tokenizer)

# ---- EVALUATE ONLY ----
preds = trainer.predict(tokenized_ds["test"])
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

print("\n✅ FINAL METRICS (NO RETRAINING)")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1-score :", f1_score(y_true, y_pred))
