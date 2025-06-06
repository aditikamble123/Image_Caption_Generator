import streamlit as st
from PIL import Image
from model import load_captions, recommend_captions
import os

# ---------- Pastel Palette with Contrast ----------
PASTELS = {
    "bg": "#0e1117",            # dark background to match your screenshot
    "sidebar": "#1c1c1c",       # darker sidebar
    "card_bg": "#fce4ec",       # soft pink for caption cards
    "border": "#f48fb1",        # rose border
    "text": "#f9f9fb",          # white text for dark mode
}

# ---------- Streamlit Config ----------
st.set_page_config(page_title="✨ Caption Recommender", layout="wide")

# ---------- Global CSS ----------

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {PASTELS['bg']};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {PASTELS['sidebar']};
        }}
        h1, h2, h3, h4, h5, h6, p, label {{
            color: {PASTELS['text']};
        }}
        .caption-box {{
            background-color: {PASTELS['card_bg']};
            border-left: 6px solid {PASTELS['border']};
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 12px;
            font-size: 16px;
            color: #000000;  /* BLACK text */
            font-weight: 600; /* Bolder text */
        }}
    </style>
    """,
    unsafe_allow_html=True
)


# ---------- Sidebar ----------
with st.sidebar:
    st.title("🧠 Caption Recommender")
    st.markdown("Upload a photo and get aesthetic Insta captions using AI ✨")
    st.markdown("---")
    st.markdown("👩‍💻 Built with [CLIP](https://openai.com/research/clip) + Streamlit")
    st.caption("Made by Aditi Kamble 💖")

# ---------- Main Section ----------
st.title("📸 AI-Powered Caption Recommender")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # File info: filename and size (matches your screenshot style)
    file_size_kb = round(uploaded_file.size / 1024, 1)
    st.markdown(
        f"📂 **{uploaded_file.name}** &nbsp; _{file_size_kb}KB_"
    )

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        image_path = "temp.jpg"
        image.save(image_path)
        st.image(image, caption="Your Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("✨ Finding the best captions for your vibe..."):
            try:
                captions = load_captions(r"C:\Users\ADITI\projects\Image_caption\captions.csv")
                if not captions:
                    st.error("⚠️ No captions found.")
                    st.stop()

                top_captions = recommend_captions(image_path, captions, top_k=5)

                # Filter out empty or None captions
                top_captions = [cap for cap in top_captions if cap and cap.strip() != ""]

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
                st.stop()

        st.subheader("🌸 Recommended Captions")
        for i, cap in enumerate(top_captions, 1):
            st.markdown(
                f"<div class='caption-box'>💬 <b>{cap}</b></div>",
                unsafe_allow_html=True
            )

else:
    st.info("📤 Upload an image to get started!")
