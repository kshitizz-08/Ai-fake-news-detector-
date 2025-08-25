import streamlit as st
import joblib
import os

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("### Check if a news article is **Fake or Real**")

# Paths for model and vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Load model and vectorizer with error handling
try:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        st.error("Model files are missing! Please upload `model.pkl` and `vectorizer.pkl` to your repository.")
    else:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        # User input
        user_input = st.text_area("Enter News Text:", height=150, placeholder="Type or paste a news article...")

        if st.button("🔍 Predict"):
            if user_input.strip() == "":
                st.warning("⚠ Please enter some text!")
            else:
                # Transform and predict
                transformed_input = vectorizer.transform([user_input])
                prediction = model.predict(transformed_input)[0]

                # Show result
                if prediction == 0:
                    st.error("❌ This news is **FAKE**")
                else:
                    st.success("✅ This news is **REAL**")

except Exception as e:
    st.error(f"An error occurred: {e}")
