import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os

st.set_page_config(page_title="Fake News Detector", page_icon="🔍")

# ── Load BERT model ───────────────────────────────────────────────────────────
# @st.cache_resource means this runs ONCE and stays in memory.
# Without it, the model reloads on every user interaction — very slow.

@st.cache_resource
def load_bert_model():
    model_path = "models/bert_model"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # set to evaluation mode — disables dropout layers
    return tokenizer, model

tokenizer, model = load_bert_model()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    fake_prob = probs[0].item()
    real_prob = probs[1].item()

    if fake_prob > real_prob:
        return "FAKE", fake_prob * 100
    else:
        return "REAL", real_prob * 100

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.write("Paste a news headline or short statement below.")

text_input = st.text_area("", placeholder="e.g. Government secretly adding chemicals to drinking water...", height=150)

if st.button("Analyze", type="primary"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            prediction, confidence = predict(text_input)

        if prediction == "FAKE":
            st.error(f"🔴 **FAKE** — {confidence:.1f}% confidence")
        else:
            st.success(f"🟢 **REAL** — {confidence:.1f}% confidence")

        st.caption("Note: Model trained on political statements from PolitiFact. Works best on short claims.")
    else:
        st.warning("Please enter some text.")