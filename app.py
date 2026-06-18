import streamlit as st
import joblib

st.title("🔍 Fake News Detector")
st.write("Baseline model (Logistic Regression) — BERT coming soon")

@st.cache_resource
def load_model():
    model = joblib.load("models/lr_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

text_input = st.text_area("Paste a news headline or statement:", height=150)

if st.button("Analyze"):
    if text_input.strip():
        vec = vectorizer.transform([text_input])
        prediction = model.predict(vec)[0]
        probabilities = model.predict_proba(vec)[0]
        confidence = max(probabilities) * 100

        if prediction == "FAKE":
            st.error(f"🔴 FAKE NEWS — {confidence:.1f}% confidence")
        else:
            st.success(f"🟢 REAL NEWS — {confidence:.1f}% confidence")
    else:
        st.warning("Please enter some text.")