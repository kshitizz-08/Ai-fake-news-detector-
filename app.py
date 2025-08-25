import streamlit as st
import joblib
import os

# ✅ Page Config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ✅ Title & Description
st.title("📰 Fake News Detector")
st.markdown("""
This app uses a **Machine Learning model** to predict whether the entered news is **Fake** or **Real**.
Just paste the news text below and click **Predict**.
""")

# ✅ Paths for model and vectorizer
MODEL_PATH = "backend/model.pkl"
VECTORIZER_PATH = "backend/vectorizer.pkl"

# ✅ Load Model and Vectorizer
model, vectorizer = None, None
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("⚠ Model files are missing! Please upload `model.pkl` and `vectorizer.pkl` to the `backend/` folder.")
else:
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")

# ✅ User Input Section
user_input = st.text_area(
    "📝 Enter News Text:",
    height=150,
    placeholder="Type or paste a news article here..."
)

# ✅ Prediction Logic
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter some text!")
    else:
        if model and vectorizer:
            with st.spinner("Analyzing the news..."):
                try:
                    transformed_input = vectorizer.transform([user_input])
                    prediction = model.predict(transformed_input)[0]

                    # ✅ Display Result
                    if prediction == 0:
                        st.error("❌ This news is **FAKE**")
                    else:
                        st.success("✅ This news is **REAL**")
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
        else:
            st.error("Model not loaded. Please check the files.")
