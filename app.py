import streamlit as st
import joblib
import os
import base64
import re
from pathlib import Path
from typing import Dict

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _image_to_data_uri(image_path: Path) -> str:
    try:
        mime = {
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(image_path.suffix.lower(), "application/octet-stream")
        data = image_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""


def _embed_assets(html_text: str, frontend_dir: Path) -> str:
    # Inline CSS files referenced via <link rel="stylesheet" href="...">
    def replace_css(match: re.Match) -> str:
        href = match.group(1)
        css_path = (frontend_dir / href).resolve()
        css_text = _read_text(css_path)
        return f"<style>\n{css_text}\n</style>"

    html_text = re.sub(r"<link[^>]+href=\"([^\"]+)\"[^>]*>", replace_css, html_text)

    # Inline JS files referenced via <script src="...">
    def replace_js(match: re.Match) -> str:
        src = match.group(1)
        js_path = (frontend_dir / src).resolve()
        js_text = _read_text(js_path)
        return f"<script>\n{js_text}\n</script>"

    html_text = re.sub(r"<script[^>]+src=\"([^\"]+)\"[^>]*></script>", replace_js, html_text)

    # Replace <img src="..."> with data URIs so assets load inside Streamlit
    def replace_img(match: re.Match) -> str:
        prefix = match.group(1)
        src = match.group(2)
        suffix = match.group(3)
        img_path = (frontend_dir / src).resolve()
        data_uri = _image_to_data_uri(img_path)
        if data_uri:
            return f"{prefix}{data_uri}{suffix}"
        return match.group(0)

    html_text = re.sub(r"(<img[^>]+src=\")(.*?)(\"[^>]*>)", replace_img, html_text)

    # Also handle CSS url(...) references within inline styles
    def replace_css_urls(css_text: str) -> str:
        def repl(m: re.Match) -> str:
            url = m.group(1).strip('\"\'')
            asset_path = (frontend_dir / url).resolve()
            data_uri = _image_to_data_uri(asset_path)
            return f"url('{data_uri if data_uri else url}')"

        return re.sub(r"url\(([^)]+)\)", repl, css_text)

    # Apply to any <style> blocks
    def style_repl(m: re.Match) -> str:
        before = m.group(1)
        css = m.group(2)
        after = m.group(3)
        return f"{before}{replace_css_urls(css)}{after}"

    html_text = re.sub(r"(<style[^>]*>)([\s\S]*?)(</style>)", style_repl, html_text)

    return html_text


def render_frontend(page_name: str = "index.html") -> None:
    frontend_dir = Path("frontend").resolve()
    page_path = (frontend_dir / page_name)
    if not page_path.exists():
        st.error(f"Frontend page not found: {page_name}")
        return

    raw_html = _read_text(page_path)
    html = _embed_assets(raw_html, frontend_dir)
    st.components.v1.html(html, height=900, scrolling=True)


# Shared model/vectorizer paths (search root and backend/)
MODEL_CANDIDATES = [
    "model.pkl",
    str(Path("backend") / "model.pkl"),
]
VECTORIZER_CANDIDATES = [
    "vectorizer.pkl",
    str(Path("backend") / "vectorizer.pkl"),
]

def resolve_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ---- Helpers brought from backend logic (adapted for Streamlit) ----
try:
    import requests  # optional
    from bs4 import BeautifulSoup  # optional
except Exception:
    requests = None
    BeautifulSoup = None

import string
import numpy as np


def is_url(text: str) -> bool:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(text.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def extract_text_from_url(url: str) -> str:
    if requests is None or BeautifulSoup is None:
        return url
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return " ".join(soup.stripped_strings)
    except Exception:
        return url


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_top_features_for_input(model, vectorizer, X, cleaned: str, top_k: int = 10):
    try:
        feature_names = vectorizer.get_feature_names_out()
        # Handle linear models or NB
        feature_importance = None
        if hasattr(model, "coef_"):
            feature_importance = model.coef_[0]
        elif hasattr(model, "feature_log_prob_"):
            if len(model.classes_) >= 2:
                fake_idx = 1 if 1 in model.classes_ else 0
                real_idx = 0 if fake_idx == 1 else 1
                feature_importance = model.feature_log_prob_[fake_idx] - model.feature_log_prob_[real_idx]
            else:
                feature_importance = model.feature_log_prob_[0]

        top_features = []
        if feature_importance is not None and len(feature_importance) > 0:
            input_features = X.toarray()[0]
            non_zero_indices = [i for i, val in enumerate(input_features) if val > 0]
            scores = []
            for idx in non_zero_indices:
                if idx < len(feature_importance):
                    scores.append((feature_names[idx], float(input_features[idx] * feature_importance[idx]), float(input_features[idx])))
            scores.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = scores[:top_k]
        else:
            # fallback: most frequent words
            words = cleaned.split()
            freq = {}
            for w in words:
                if len(w) > 2:
                    freq[w] = freq.get(w, 0) + 1
            top_features = [(w, 0.0, c) for w, c in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        return top_features
    except Exception:
        return []


def compute_confidence(model, X) -> float:
    try:
        proba = model.predict_proba(X)[0]
        return float(max(proba))
    except Exception:
        try:
            import math
            score = model.decision_function(X)[0]
            return 1 / (1 + math.exp(-abs(float(score))))
        except Exception:
            return 0.5


def load_model_and_vectorizer():
    model_path = resolve_existing(MODEL_CANDIDATES)
    vect_path = resolve_existing(VECTORIZER_CANDIDATES)
    if not model_path or not vect_path:
        st.error("Model files are missing! Please place `model.pkl` and `vectorizer.pkl` in project root or in `backend/`.")
        return None, None
    return joblib.load(model_path), joblib.load(vect_path)


# ---- UI ----
st.title("📰 Fake News Detector")
st.markdown("### Run the entire project inside Streamlit")

mode = st.sidebar.radio(
    "Interface",
    ("Full Streamlit App", "Full HTML/CSS (frontend)", "Streamlit Minimal UI"),
    index=0,
)

if mode == "Full HTML/CSS (frontend)":
    pages: Dict[str, str] = {
        "Home (index.html)": "index.html",
        "Login (login.html)": "login.html",
        "Features (features.html)": "features.html",
    }
    page_label = st.sidebar.selectbox("Page", list(pages.keys()), index=0)
    render_frontend(pages[page_label])

elif mode == "Full Streamlit App":
    st.sidebar.subheader("Navigation")
    section = st.sidebar.selectbox("Go to", ["Predict", "Features", "Metrics", "About"], index=0)

    model, vectorizer = load_model_and_vectorizer()

    if section == "Predict":
        st.markdown("#### Enter text or a URL")
        input_text = st.text_area("Input", height=180, placeholder="Paste news text or a URL...")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            run_btn = st.button("🔍 Predict")
        with col_b:
            treat_as_url = st.checkbox("Treat input as URL (auto-detect if starts with http)")

        if run_btn:
            if not input_text.strip():
                st.warning("Please enter text or URL.")
            elif model is None or vectorizer is None:
                pass
            else:
                as_url = treat_as_url or is_url(input_text)
                resolved = extract_text_from_url(input_text) if as_url else input_text
                cleaned = clean_text(resolved)
                X = vectorizer.transform([cleaned])
                pred = model.predict(X)[0]
                label = "FAKE" if int(pred) == 1 else "REAL"
                confidence = compute_confidence(model, X)

                if label == "FAKE":
                    st.error(f"❌ Prediction: {label}  (confidence: {round(confidence*100,2)}%)")
                else:
                    st.success(f"✅ Prediction: {label}  (confidence: {round(confidence*100,2)}%)")

                with st.expander("View cleaned text"):
                    st.write(cleaned)

                st.markdown("#### Top contributing features")
                top_feats = get_top_features_for_input(model, vectorizer, X, cleaned, top_k=10)
                if top_feats:
                    st.dataframe(
                        {
                            "term": [t[0] for t in top_feats],
                            "contribution": [round(t[1], 4) for t in top_feats],
                            "tfidf_weight": [round(t[2], 4) for t in top_feats],
                        },
                        use_container_width=True,
                    )
                else:
                    st.info("No feature contributions available.")

    elif section == "Features":
        st.markdown("#### Global feature importance (if supported by model)")
        if model is None or vectorizer is None:
            pass
        else:
            feature_names = getattr(vectorizer, "get_feature_names_out", lambda: [])()
            if hasattr(model, "coef_") and len(getattr(model, "coef_", [])) > 0:
                coefs = model.coef_[0]
                # Top positive features (towards FAKE class if class 1)
                top_idx = np.argsort(coefs)[-25:][::-1]
                bot_idx = np.argsort(coefs)[:25]
                st.write("Top positive features")
                st.dataframe(
                    {
                        "term": [feature_names[i] for i in top_idx],
                        "weight": [float(coefs[i]) for i in top_idx],
                    },
                    use_container_width=True,
                )
                st.write("Top negative features")
                st.dataframe(
                    {
                        "term": [feature_names[i] for i in bot_idx],
                        "weight": [float(coefs[i]) for i in bot_idx],
                    },
                    use_container_width=True,
                )
            elif hasattr(model, "feature_log_prob_"):
                probs = model.feature_log_prob_
                if probs.shape[0] >= 2:
                    diff = probs[1] - probs[0]
                    top_idx = np.argsort(diff)[-25:][::-1]
                    bot_idx = np.argsort(diff)[:25]
                    st.write("Top features (class 1 vs class 0)")
                    st.dataframe(
                        {
                            "term": [feature_names[i] for i in top_idx],
                            "weight": [float(diff[i]) for i in top_idx],
                        }
                    )
                    st.write("Bottom features (class 0 vs class 1)")
                    st.dataframe(
                        {
                            "term": [feature_names[i] for i in bot_idx],
                            "weight": [float(diff[i]) for i in bot_idx],
                        }
                    )
            else:
                st.info("This model type does not expose global feature importances.")

    elif section == "Metrics":
        st.markdown("#### Training/validation metrics")
        metrics_paths = [
            Path("backend") / "metrics.json",
            Path("metrics.json"),
        ]
        metrics = None
        for p in metrics_paths:
            if p.exists():
                try:
                    import json
                    metrics = json.loads(p.read_text(encoding="utf-8"))
                    break
                except Exception:
                    pass
        if metrics is None:
            st.info("No metrics.json found.")
        else:
            st.json(metrics)

    else:  # About
        st.write("This Streamlit app integrates prediction, interpretability, and optional HTML frontend rendering.")

else:  # Streamlit Minimal UI
    try:
        model_path = resolve_existing(MODEL_CANDIDATES)
        vect_path = resolve_existing(VECTORIZER_CANDIDATES)
        if not model_path or not vect_path:
            st.error("Model files are missing! Please add `model.pkl` and `vectorizer.pkl` to project root or `backend/`.")
    else:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vect_path)

        user_input = st.text_area("Enter News Text:", height=150, placeholder="Type or paste a news article...")

        if st.button("🔍 Predict"):
            if user_input.strip() == "":
                st.warning("⚠ Please enter some text!")
            else:
                transformed_input = vectorizer.transform([user_input])
                prediction = model.predict(transformed_input)[0]

                if prediction == 0:
                        st.error("❌ This news is **FAKE**")
                else:
                        st.success("✅ This news is **REAL**")
except Exception as e:
    st.error(f"An error occurred: {e}")
