<div align="center">

# 🔍 VeriFact AI — Real-Time Fake News Detector

**Classify news as REAL or FAKE in real-time using NLP and Machine Learning.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

</div>

---

## 📌 About the Project

**VeriFact AI** is a full-stack, production-ready application for **real-time fake news detection** using classical NLP techniques and multiple machine learning models. Users can paste any news text into the interface and instantly receive a classification result (REAL or FAKE), a confidence score, key indicator words, and — for real news — links to trusted news sources.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Multi-Model Classification** | Uses Logistic Regression, Naive Bayes, and LSTM to classify news |
| 🔤 **NLP Preprocessing** | Tokenization, stopword removal, and lemmatization via NLTK |
| 📊 **TF-IDF Feature Extraction** | Converts text to numerical features for ML inference |
| 📈 **Confidence Score** | Shows prediction certainty as a percentage |
| 🔑 **Keyword Explainability** | Highlights the most influential words in the decision |
| 🌐 **Real News Redirection** | Provides 3 trusted article links (Google News, BBC, AP News) for REAL news |
| ⚡ **Real-Time Analysis** | FastAPI backend delivers near-instant predictions |
| 💅 **Modern SaaS UI** | Dark-mode React frontend with smooth animations |

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────┐
│             React Frontend              │
│  (Vite + TypeScript + Tailwind CSS)     │
│         localhost:5173                  │
└──────────────┬──────────────────────────┘
               │  HTTP POST /api/predict
               │  (Vite proxy → port 8000)
               ▼
┌─────────────────────────────────────────┐
│           FastAPI Backend               │
│         localhost:8000                  │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │       NLP Preprocessor         │    │
│  │  Tokenize → Remove Stopwords   │    │
│  │       → Lemmatize              │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │     TF-IDF Vectorizer          │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │   Logistic Regression  ──┐      │    │
│  │   Naive Bayes          ──┼─►avg │    │
│  │   LSTM                 ──┘      │    │
│  └──────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend
- **Python 3.10+**
- **FastAPI** — REST API framework
- **Scikit-learn** — Logistic Regression, Naive Bayes, TF-IDF
- **TensorFlow / Keras** — LSTM sequence model
- **NLTK** — Tokenization, lemmatization, stopword removal
- **Uvicorn** — ASGI server

### Frontend
- **React 18** + **TypeScript**
- **Vite** — Build tool & dev server
- **Tailwind CSS** — Styling
- **Framer Motion** — Animations
- **Lucide React** — Icons

---

## 🚀 Running Locally

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Clone the repository

```bash
git clone https://github.com/Harsini-15/verifact-ai---real-time-fake-news-detector.git
cd verifact-ai---real-time-fake-news-detector
```

### 2. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Train ML models

```bash
cd backend/utils
python3 model_trainer.py
cd ../..
```

> ⚠️ This only needs to be done once. Models are saved to `backend/models/`.

### 4. Start the Backend (FastAPI)

```bash
venv/bin/python backend/main.py
```

The API will be live at: `http://localhost:8000`

### 5. Start the Frontend (React)

```bash
npm install
npm run dev
```

The app will be live at: **http://localhost:5173**

---

## 📡 API Reference

### `POST /api/predict`

Classify a news article as REAL or FAKE.

**Request Body:**
```json
{
  "text": "Your news article text goes here..."
}
```

**Response:**
```json
{
  "prediction": "REAL",
  "confidence": 0.91,
  "keywords": ["economic", "growth", "reported"],
  "explanation": "The system classified this as REAL based on linguistic patterns...",
  "importantWords": ["economic", "fiscal", "growth"],
  "articles": [
    {
      "title": "Latest updates on economic",
      "source": "Google News",
      "url": "https://news.google.com/search?q=economic+growth"
    }
  ]
}
```

### `GET /api/get_real_news?q={query}`

Fetch trusted news article links for a given query.

---

## 📁 Project Structure

```
verifact-ai/
├── backend/
│   ├── main.py               # FastAPI app & API endpoints
│   ├── models/               # Saved ML models (generated by training)
│   └── utils/
│       ├── nlp_utils.py      # NLP preprocessing pipeline
│       ├── model_trainer.py  # Train & evaluate LR, NB models
│       └── lstm_model.py     # LSTM architecture & training
├── src/
│   ├── App.tsx               # Main React component
│   ├── main.tsx              # App entry point
│   └── index.css             # Global styles
├── data/
│   └── news_dataset.csv      # Training dataset
├── requirements.txt          # Python dependencies
├── package.json              # Node.js dependencies
├── vite.config.ts            # Vite + proxy configuration
└── README.md
```

---

## 🧪 Example Test Cases

| Input Text | Expected Result |
|---|---|
| *"NASA confirms Earth is actually flat and the moon is a hologram."* | ❌ FAKE |
| *"Economic growth reported in the second quarter of the fiscal year."* | ✅ REAL |
| *"Vaccines contain tracking chips connecting to 5G towers."* | ❌ FAKE |
| *"The World Health Organization released new guidelines on physical activity."* | ✅ REAL |

---

## 📝 Evaluation Metrics

Models are evaluated using:
- **Accuracy** — Overall correctness
- **Precision** — Of predicted REAL/FAKE, how many were correct
- **Recall** — Of actual REAL/FAKE, how many were detected
- **F1 Score** — Harmonic mean of Precision & Recall

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
Built with ❤️ using FastAPI + React + Machine Learning
</div>
