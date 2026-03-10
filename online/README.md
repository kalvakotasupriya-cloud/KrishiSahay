# 🌾 Kisan Call Centre Query Assistant

An AI-Powered Agricultural Helpdesk using IBM Watsonx Granite LLM and FAISS.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## 📁 File Structure

```
kisan_assistant/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── README.md                 # This file
└── utils/
    ├── __init__.py
    ├── query_engine.py       # FAISS search + IBM Watsonx LLM
    ├── translations.py       # 7-language UI string support
    ├── tts_stt.py           # Text-to-Speech & Speech-to-Text
    ├── schemes.py            # Government scheme recommender
    └── weather.py            # Weather advisory integration
```

---

## 🌍 Supported Languages
| Language | Code |
|----------|------|
| English | en |
| Hindi (हिंदी) | hi |
| Telugu (తెలుగు) | te |
| Tamil (தமிழ்) | ta |
| Kannada (ಕನ್ನಡ) | kn |
| Marathi (मराठी) | mr |
| Punjabi (ਪੰਜਾਬੀ) | pa |

---

## ⚙️ Features

- **🤖 AI Chatbot** - FAISS semantic search + IBM Watsonx Granite LLM
- **🎙️ Speech-to-Text** - Browser Web Speech API (supports regional languages)
- **🔊 Text-to-Speech** - gTTS answer playback in regional languages
- **🌍 Multilingual** - 7 Indian languages with real-time translation
- **🌦️ Weather Advisory** - Location-based weather farming tips
- **📋 Scheme Recommender** - Personalized government scheme suggestions
- **🌱 Crop Advisor** - State + season based crop recommendations
- **📶 Offline Mode** - FAISS-only mode for low connectivity areas
- **💡 Daily Tips** - Rotating farmer tips panel

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `WATSONX_API_KEY` | IBM Cloud API Key |
| `WATSONX_PROJECT_ID` | Watsonx Project ID |
| `WATSONX_URL` | Watsonx endpoint URL |
| `WEATHER_API_KEY` | OpenWeatherMap API key |

---

## 📦 Tech Stack

- **Frontend**: Streamlit + Custom CSS
- **Vector Search**: FAISS + sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: IBM Watsonx Granite 13B Instruct
- **Translation**: deep-translator (Google Translate API)
- **TTS**: gTTS (Google Text-to-Speech)
- **STT**: Browser Web Speech API
- **Weather**: OpenWeatherMap API

---

## 🌐 Offline Mode

When `WATSONX_API_KEY` is not set or mode is "Offline":
- Uses FAISS semantic similarity search only
- Pre-loaded KCC dataset with 15+ agricultural Q&A pairs
- No internet required for core functionality

---

## 📝 Adding More Q&A Data

Edit `KCC_DATA` in `utils/query_engine.py`:
```python
KCC_DATA = [
    {"query": "Your question here", "answer": "Detailed answer here"},
    ...
]
```

The FAISS index rebuilds automatically on next startup.