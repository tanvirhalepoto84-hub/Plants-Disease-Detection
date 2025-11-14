import streamlit as st
from PIL import Image
import edge_tts
from deep_translator import GoogleTranslator
import tempfile
import asyncio
import os
from dotenv import load_dotenv
import google.generativeai as genai

# -----------------------------------------------
# 🌿 Page Configuration
# -----------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered",
)

# -----------------------------------------------
# 🌿 Custom CSS
# -----------------------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e3f2fd, #e8f5e9);
        color: #1b5e20;
    }
    .stApp {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        border-radius: 15px;
        padding: 20px;
    }
    .main-title {
        font-size: 36px;
        font-weight: 800;
        text-align: center;
        color: #2e7d32;
        margin-bottom: 20px;
    }
    .subheader {
        background: #a5d6a7;
        color: #1b5e20;
        padding: 8px 15px;
        border-radius: 10px;
        font-weight: 600;
        margin-top: 25px;
    }
    .info-box {
        background: #ffffffcc;
        border-radius: 12px;
        padding: 12px 18px;
        margin-top: 8px;
        box-shadow: 0 0 10px #c8e6c9;
    }
    .stButton>button {
        background-color: #43a047 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
    }
    .stSelectbox label {
        font-weight: 600;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# 🌿 Title
# -----------------------------------------------
st.markdown("<div class='main-title'>🌿 Plant Disease Detection App</div>", unsafe_allow_html=True)
st.caption("Powered by Advanced AI System - Analyze plant leaf images with multilingual info & voice narration.")

# -----------------------------------------------
# 🌿 Load API Key
# -----------------------------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    st.error("❌ API key not found. Please add it to your .env file.")
    st.stop()

genai.configure(api_key=API_KEY)

# -----------------------------------------------
# 🌿 Sidebar
# -----------------------------------------------
st.sidebar.title("⚙️ Options")
st.sidebar.info("Upload or capture a leaf image to detect plant disease.")
text_lang = st.sidebar.selectbox("🌐 Display Language", ["English", "Urdu", "Sindhi"])
voice_lang = st.sidebar.selectbox("🔊 Voice Language", ["English", "Urdu", "Sindhi"])

# -----------------------------------------------
# 🌿 Image Upload or Camera Capture
# -----------------------------------------------
option = st.radio("📷 Choose Image Source", ["Upload Image", "Use Camera"])

uploaded_file = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("📸 Upload a plant leaf image...", type=["jpg", "jpeg", "png"])
elif option == "Use Camera":
    captured_image = st.camera_input("🎥 Capture leaf image from camera")
    if captured_image is not None:
        uploaded_file = captured_image

# -----------------------------------------------
# 🧩 Helper Functions
# -----------------------------------------------
async def generate_edge_tts_full(text, voice):
    """Generate full, high-quality voice output."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        output_path = tmp_file.name
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path

# -----------------------------------------------
# 🌿 Main Functionality
# -----------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Selected Image", use_container_width=True)

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing image... ⏳"):
            model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = """
            You are an expert plant pathologist. Analyze the uploaded plant image and provide the following information:
            1. Plant Name
            2. Plant Scientific Name
            3. Disease
            4. Causal Agent
            5. Cause of Disease
            6. Solution
            7. Use of Medicine

            Format it neatly using simple text. If unknown, write 'Unknown'.
            """

            try:
                response = model.generate_content([prompt, image])
                st.session_state.analysis_result = response.text
                st.success("✅ Analysis Complete!")
            except Exception as e:
                st.error(f"❌ Error analyzing image: {e}")

    if st.session_state.analysis_result:
        result_text = st.session_state.analysis_result
        st.markdown("<div class='subheader'>🪴 Plant Information</div>", unsafe_allow_html=True)

        lang_codes = {"English": "en", "Urdu": "ur", "Sindhi": "sd"}

        display_text = result_text
        if text_lang != "English":
            try:
                display_text = GoogleTranslator(source="auto", target=lang_codes[text_lang]).translate(result_text)
            except Exception:
                st.warning("⚠️ Translation failed. Showing English text.")

        st.markdown(f"<div class='info-box'>{display_text}</div>", unsafe_allow_html=True)

        # 🔊 Voice Output Section
        st.markdown("<div class='subheader'>🔊 Voice Output</div>", unsafe_allow_html=True)
        voice_map = {
            "English": "en-US-AriaNeural",
            "Urdu": "ur-PK-AsadNeural",
            "Sindhi": "ur-PK-AsadNeural"  # best available for Sindhi-like tone
        }

        if st.button("▶️ Play Voice"):
            st.success(f"🎙️ Speaking in: {voice_lang}")
            try:
                voice_name = voice_map.get(voice_lang, "en-US-AriaNeural")
                paragraph = result_text

                if voice_lang != "English":
                    try:
                        paragraph = GoogleTranslator(source="auto", target=lang_codes[voice_lang]).translate(paragraph)
                    except Exception:
                        st.warning("⚠️ Voice translation failed. Using English instead.")

                # 🎧 Full high-quality voice (no small parts)
                output_file = asyncio.run(generate_edge_tts_full(paragraph, voice_name))
                with open(output_file, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")

            except Exception as e:
                st.error(f"❌ Could not generate voice: {e}")

else:
    st.info("📤 Please upload or capture a plant leaf image to begin.")
