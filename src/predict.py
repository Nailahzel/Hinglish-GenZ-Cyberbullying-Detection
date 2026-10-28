import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to your saved model
MODEL_PATH = "outputs/final_model"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

model.eval()

def predict(text: str):
    """Returns prediction label + confidence score."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs).logits

    probs = torch.softmax(outputs, dim=1)
    confidence = torch.max(probs).item() * 100
    label = torch.argmax(probs).item()

    if label == 1:
        return f"Cyberbullying 😡 (Confidence: {confidence:.2f}%)"
    else:
        return f"Safe / Neutral 🙂 (Confidence: {confidence:.2f}%)"


# Test manually
if __name__ == "__main__":
    print(predict("Tu pagal ho gaya hai 🤡"))
    print(predict("You are really kind and supportive ❤️"))
