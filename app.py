import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification



if "model" not in st.session_state:

    model_dir = "outputs/final_model"

    st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True
    )

    st.session_state["model"] = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    )

    st.session_state["model"].eval()


tok = st.session_state["tokenizer"]
model = st.session_state["model"]
device = torch.device("cpu")



st.set_page_config(page_title="Cyberbullying Detector", page_icon="")
st.title(" Cyberbullying Detection Tool")
st.write("Detects **Hinglish, Gen-Z slang & Emoji-based cyberbullying**.")


user_input = st.text_area("Enter text to analyze:", height=120)



def classify(text):
    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    label = probs.argmax()
    confidence = probs[label] * 100
    return label, confidence



if st.button("Analyze"):
    if not user_input.strip():
        st.warning(" Please enter some text.")
    else:
        label, conf = classify(user_input)

        if label == 1:
            st.error(f"Cyberbullying Detected— Confidence: {conf:.2f}%")
        else:
            st.success(f"Safe Message — Confidence: {conf:.2f}%")

        st.caption("Model: Fine-tuned XLM-RoBERTa • Hinglish + Gen-Z + Emoji")
