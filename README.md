# Intelligent Derma Disease Detection 🩺

## Overview
Intelligent Derma Disease Detection is an AI-powered application designed to assist in the early detection and analysis of skin diseases. Using a Deep Learning model (EfficientNetB0) and Large Language Models (Groq), the app analyzes skin images to predict potential conditions and provides detailed medical insights, including symptoms, treatments, and lifestyle recommendations.

## Features
- **Skin Disease Classification**: accurately classifies skin conditions into 10 categories using a fine-tuned EfficientNetB0 model.
- **AI-Powered Insights**: Generates comprehensive medical information (symptoms, treatments, products, diet) using Groq LLM.
- **User-Friendly Interface**: Built with Streamlit for a clean and interactive experience.
- **Real-time Analysis**: Instant feedback on uploaded images.

## Supported Classes
The model can detect the following skin conditions:
1. Eczema
2. Melanoma
3. Atopic Dermatitis
4. Basal Cell Carcinoma (BCC)
5. Melanocytic Nevi (NV)
6. Benign Keratosis-like Lesions (BKL)
7. Psoriasis / Lichen Planus
8. Seborrheic Keratoses & Benign Tumors
9. Tinea / Ringworm / Candidiasis (Fungal)
10. Warts / Molluscum / Viral Infections

## Tech Stack
- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow, Keras (EfficientNetB0)
- **LLM Integration**: Groq API (Llama 3.1)
- **Image Processing**: Pillow, OpenCV, NumPy
- **Environment Management**: Python-dotenv

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd IntelligentDermaDiseaaseDetection
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

1. **Run the Application**
   ```bash
   streamlit run newApp.py
   ```

2. **Upload an Image**
   - Click on the "Upload Skin Image" button.
   - Select an image (JPG, JPEG, PNG).

3. **View Results**
   - Click "Predict" to see the classification result and confidence score.
   - Read the AI-generated medical insights below the prediction.

## Disclaimer
This application is for **educational purposes only** and should not replace professional medical advice. Always consult a certified dermatologist for diagnosis and treatment.
