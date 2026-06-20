import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from explain import get_explanation

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")

# ── Load BERT model (cached, runs once) ───────────────────────────────────────
@st.cache_resource
def load_bert_model():
    model_path = "models/bert_model"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
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
st.write("Paste a news headline or short statement to check if it shows patterns of misinformation.")

with st.expander("ℹ️ About this tool & its limitations"):
    st.write("""
    This model was fine-tuned on the **LIAR dataset** — ~12,800 political statements
    fact-checked by PolitiFact. It detects *language patterns* commonly associated
    with misinformation (sweeping claims, lack of specificity, certain rhetorical patterns).

    **It does not fact-check claims against real-world data.** It cannot verify
    if something is literally true — it recognizes how misinformation tends to
    *sound*, based on patterns learned from training data. Treat results as a
    signal, not a verdict.
    """)

text_input = st.text_area(
    "Enter text:",
    placeholder="e.g. Government secretly adding chemicals to drinking water to control population...",
    height=130
)

col1, col2 = st.columns([1, 1])
with col1:
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if analyze_clicked:
    cleaned_text = text_input.strip()

    # ── Edge case: empty input ──
    if not cleaned_text:
        st.warning("⚠️ Please enter some text to analyze.")

    # ── Edge case: too short to mean anything ──
    elif len(cleaned_text.split()) < 3:
        st.warning("⚠️ Please enter a full statement (at least a few words) for a meaningful result.")

    else:
        with st.spinner("Analyzing language patterns..."):
            prediction, confidence = predict(cleaned_text)

        st.divider()

        # ── Verdict display ──
        if prediction == "FAKE":
            st.error(f"### 🔴 Likely FAKE")
        else:
            st.success(f"### 🟢 Likely REAL")

        # ── Confidence bar ──
        st.write(f"**Confidence: {confidence:.1f}%**")
        st.progress(confidence / 100)

        # ── Explanation ──
        st.write("**Why:**")
        with st.spinner("Generating explanation..."):
            explanation = get_explanation(cleaned_text, prediction, confidence)
        st.info(explanation)

        st.caption("Model: Fine-tuned DistilBERT on LIAR dataset | Explanation: LLaMA 3.1 via Groq")

st.divider()
st.caption("Built with Streamlit, HuggingFace Transformers, and Groq | [GitHub](https://github.com/vardhanarunima-oss/fake-news-detector)")