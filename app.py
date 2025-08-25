import streamlit as st
import joblib
import os

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Inject custom HTML/CSS styling
st.markdown(
    """
    <style>
        /* Global styles */
        html, body { background: #0f172a; }
        .stApp { background: radial-gradient(1200px 600px at 10% 0%, #1e293b 0%, #0f172a 40%, #0b1220 100%); }

        /* Typography */
        h1, h2, h3, h4, h5, h6 { color: #e2e8f0 !important; }
        p, span, label { color: #cbd5e1 !important; }

        /* Card container */
        .app-card {
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.7), inset 0 1px 0 rgba(148,163,184,0.1);
            border-radius: 16px;
            padding: 24px 22px;
            margin: 16px 0 24px 0;
        }

        /* Text area */
        .stTextArea textarea {
            background: rgba(2, 6, 23, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.25);
            color: #e2e8f0;
            border-radius: 12px;
        }
        .stTextArea textarea:focus {
            outline: none !important;
            border-color: #60a5fa !important;
            box-shadow: 0 0 0 1px #60a5fa !important;
        }

        /* Button */
        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: #fff;
            border: 0;
            padding: 0.6rem 1rem;
            border-radius: 10px;
            font-weight: 600;
            letter-spacing: .2px;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.35);
        }
        .stButton > button:hover {
            filter: brightness(1.05);
        }

        /* Result badges */
        .badge {
            display: inline-block;
            padding: 10px 14px;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: .3px;
        }
        .badge-real {
            background: rgba(22, 163, 74, 0.15);
            border: 1px solid rgba(22, 163, 74, 0.35);
            color: #22c55e;
        }
        .badge-fake {
            background: rgba(220, 38, 38, 0.15);
            border: 1px solid rgba(220, 38, 38, 0.35);
            color: #f87171;
        }
        .subtitle {
            color: #93c5fd !important;
            font-weight: 500;
            margin-top: -8px;
        }
    </style>
    <div class="app-card">
        <h1>📰 Fake News Detector</h1>
        <p class="subtitle">Paste a headline or article and get an instant prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='app-card'><h3>Check if a news article is <strong>Fake</strong> or <strong>Real</strong></h3></div>", unsafe_allow_html=True)

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

                # Show result with custom badges
                if prediction == 0:
                    st.markdown("<span class='badge badge-fake'>❌ This news is FAKE</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge badge-real'>✅ This news is REAL</span>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred: {e}")
