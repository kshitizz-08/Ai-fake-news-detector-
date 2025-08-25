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


st.title("📰 Fake News Detector")
st.markdown("### Choose interface mode")

mode = st.sidebar.radio(
    "Interface",
    ("Streamlit Minimal UI", "Full HTML/CSS (frontend)"),
    index=1,
)

# Paths for model and vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if mode == "Full HTML/CSS (frontend)":
    # Let user pick which HTML to render
    available_pages: Dict[str, str] = {
        "Home (index.html)": "index.html",
        "Login (login.html)": "login.html",
        "Features (features.html)": "features.html",
    }
    page_label = st.sidebar.selectbox("Page", list(available_pages.keys()), index=0)
    render_frontend(available_pages[page_label])
else:
    # Minimal Streamlit-based predictor
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            st.error("Model files are missing! Please upload `model.pkl` and `vectorizer.pkl` to your repository.")
        else:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)

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
