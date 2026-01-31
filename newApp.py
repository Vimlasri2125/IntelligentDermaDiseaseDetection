import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key)


# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
MODEL_PATH = "efficientnetb0_skin_disease_final.h5"
IMG_HEIGHT = 160
IMG_WIDTH = 160

CLASS_LABELS = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)",
    "Psoriasis / Lichen Planus",
    "Seborrheic Keratoses & Benign Tumors",
    "Tinea / Ringworm / Candidiasis (Fungal)",
    "Warts / Molluscum / Viral Infections"
]

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# IMAGE PREPROCESS
# --------------------------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
def predict_skin_disease(image):
    img = preprocess_image(image)
    preds = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx]) * 100
    return CLASS_LABELS[idx], confidence

# --------------------------------------------------
# GROQ INTEGRATION (REPLACED GEMINI)
# --------------------------------------------------
def generate_disease_info(disease_name, confidence):
    if not api_key:
        return None

    try:
        prompt = f"""
You are a dermatologist AI.

Give medical information for:
Disease: {disease_name}
Confidence: {confidence:.2f}%

STRICT RULES:
- Output ONLY valid JSON
- No explanations
- No markdown
- No extra text before or after JSON

JSON FORMAT:
{{
  "display_name": "{disease_name}",
  "symptoms": ["", "", "", ""],
  "treatments": ["", "", "", ""],
  "products": ["", "", "", ""],
  "food_to_eat": ["", "", "", ""],
  "food_to_avoid": ["", "", "", ""],
  "lifestyle_changes": ["", "", "", ""],
  "emergency_warning": "",
  "follow_up": ""
}}
"""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a dermatologist. Respond only in JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        raw_text = response.choices[0].message.content.strip()

        # ✅ SAFE JSON EXTRACTION
        json_start = raw_text.find("{")
        json_end = raw_text.rfind("}") + 1

        if json_start == -1 or json_end == -1:
            return None

        clean_json = raw_text[json_start:json_end]

        return json.loads(clean_json)

    except Exception as e:
        print("Groq JSON Error:", e)
        return None


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🩺 Skin Disease Detection using AI")
st.write("Upload a skin image to predict disease and get precautions.")

uploaded_file = st.file_uploader(
    "📤 Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and model:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            disease, confidence = predict_skin_disease(image)

        st.success("Prediction Complete")

        st.subheader("🧠 AI Prediction")
        st.write(f"**Disease:** {disease}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        with st.spinner("Generating medical insights..."):
            info = generate_disease_info(disease, confidence)

        if info:
            st.subheader("🩺 Symptoms")
            st.write(", ".join(info["symptoms"]))

            st.subheader("💊 Treatments")
            st.write(", ".join(info["treatments"]))

            st.subheader("🧴 Recommended Products")
            st.write(", ".join(info["products"]))

            st.subheader("🥗 Food to Eat")
            st.write(", ".join(info["food_to_eat"]))

            st.subheader("🚫 Food to Avoid")
            st.write(", ".join(info["food_to_avoid"]))

            st.subheader("🏃 Lifestyle Changes")
            st.write(", ".join(info["lifestyle_changes"]))

            st.warning(f"⚠️ {info['emergency_warning']}")
            st.info(f"🔁 Follow-up: {info['follow_up']}")

        else:
            st.warning("⚠️ Unable to generate precautions. Please try again.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("EfficientNetB0 + Groq LLM | Educational Use Only")
