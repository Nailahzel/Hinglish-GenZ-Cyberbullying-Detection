import os
import re
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




def load_dirty_csv(path):
    """
    Loads ANY messy CSV and automatically cleans:
    - Removes IDs
    - Ensures only 2 columns (text, label)
    - Drops corrupted rows
    """

    print(f"\n🧹 Cleaning dataset → {path}")

    # Read line-by-line because your datasets contain broken formats
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    cleaned_text, cleaned_label = [], []

    for row in lines:
        row = row.strip()

        # Match format: SOMETEXT , LABEL
        m = re.match(r'^(.*?),(0|1)$', row)
        if m:
            text = m.group(1).strip().strip('"').strip()
            label = int(m.group(2))
            cleaned_text.append(text)
            cleaned_label.append(label)

    df = pd.DataFrame({"text": cleaned_text, "label": cleaned_label})
    print(f"   → Loaded {len(df)} clean rows.")
    return df


def load_all_datasets():
    DATA_DIR = "data/raw"
    all_dfs = []

    print("\n📂 Loading all datasets...")

    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            path = os.path.join(DATA_DIR, file)
            df = load_dirty_csv(path)
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\n📌 TOTAL CLEAN DATA: {len(combined)} rows\n")
    return combined



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }




def main():

    df = load_all_datasets()

    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert to HF dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    print("📌 Dataset Ready!")
    print(dataset)


    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_ds = dataset.map(tokenize, batched=True)
    tokenized_ds = tokenized_ds.rename_column("label", "labels")
    tokenized_ds.set_format("torch")


    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    args = TrainingArguments(
        output_dir="outputs/checkpoints",
        eval_strategy="epoch",  # Updated!
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none"
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("\n Training Started...")
    trainer.train()

    print("\n Final Evaluation:")
    print(trainer.evaluate())


    final_dir = "outputs/final_model"
    trainer.save_model(final_dir)

    print(f"\nTraining Complete! Model saved at: {final_dir}")




if __name__ == "__main__":
    main()
