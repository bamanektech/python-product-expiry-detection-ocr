# 🛡️ ShelfLife OCR: Intelligent Product Expiry Detector

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-ED1C24?style=for-the-badge)](https://github.com/PaddlePaddle/PaddleOCR)

**ShelfLife OCR** is a specialized AI-powered web application designed to solve the common challenge of reading manufacturing and expiry dates from product packaging. By combining high-performance OCR engines with advanced Large Language Models (LLM), it accurately extracts, classifies, and standardizes dates even from cluttered packaging text.

---

## ✨ Key Features

- 📁 **Image Upload**: Seamlessly upload product packaging images (JPG, PNG, etc.).
- 🔍 **Hybrid OCR Engine**: Utilizes **PaddleOCR** (default) for state-of-the-art text extraction, with built-in support for **EasyOCR** and **PyTesseract**.
- 🧠 **AI Date Classification**: Employs **Google Gemini 2.0 Flash** to parse extracted text and identify:
  - Manufacturing Date (MFG)
  - Expiry Date (EXP)
  - Duration-based shelf life (e.g., "Best before 24 months from manufacture").
- ⚖️ **Inference Logic**: Smart algorithms handle fuzzy matching for OCR errors and automatically infer dates based on chronological order when labels are missing.
- 🎨 **Modern UI**: A responsive, clean interface with smooth animations and instant results display.

---

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **OCR Library**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **AI/LLM**: [LangChain](https://www.langchain.com/) + [Google Gemini API](https://ai.google.dev/)
- **Frontend**: HTML5, CSS3 (Gradients & Keyframe Animations), Jinja2 Templates
- **Image Processing**: [Pillow (PIL)](https://python-pillow.org/)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (optional, if using Tesseract engine)
- [PaddlePaddle](https://www.paddlepaddle.org.cn/en/install/quick) (required for PaddleOCR)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/OCR_Demo.git
   cd OCR_Demo
   ```

2. **Set up a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Google Gemini API Key:

   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

### Running the Application

Start the development server:

```bash
python app/main.py
```

Or use uvicorn directly:

```bash
uvicorn app.main:app --reload --port 8080
```

Open your browser and navigate to `http://127.0.0.1:8080`.

---

## 🧠 How the AI Works

The project uses a sophisticated prompt engineering strategy to transform messy OCR text into structured data.

1. **Fuzzy Matching**: Recognizes variations like "MGF", "Mfg Date", "Pckd on" for manufacturing, and "Exp", "Best Before", "Use By" for expiry.
2. **Date Normalization**: Understands multiple formats: `DD/MM/YYYY`, `MM-YYYY`, `DD.MON.YY`, etc.
3. **Implicit Logic**:
    - If labels are missing, the AI assumes the **earlier date** is MFG and the **later date** is EXP.
    - If a duration (e.g., "24 Months") is found without an expiry date, it calculates it from the MFG date.

---

## 📂 Project Structure

```text
OCR_Demo/
├── app/
│   ├── main.py          # FastAPI application & OCR routes
│   ├── helper.py        # LLM logic & prompt templates
│   ├── static/          # CSS, JS, and uploaded images
│   └── templates/       # HTML (Jinja2) files
├── requirements.txt     # Python dependencies
├── .env                 # API Keys (not in git)
└── LICENSE              # MIT License
```

---
