# 📊 Social Media Analysis Dashboard

A powerful, interactive web-based dashboard for analyzing social media data. Built with **Streamlit**, it allows users to explore trends, sentiment, key topics, and author interactions using advanced NLP and visualization tools.

---

## ✅ Features

- 📅 Filter posts by time range, keywords, or subreddit  
- 📈 Activity trends and time series analysis  
- 😃 Sentiment analysis using VADER  
- 🧠 Topic modeling (TF-IDF + NMF) with custom topic naming  
- 🌐 Author interaction network visualization (NetworkX + Plotly)  
- ☁️ Word clouds and frequent word bar charts  
- 🕵️ Named Entity Recognition (NER) with spaCy  
- 🤖 AI-generated insights using OpenAI (optional API integration)

---

## 📦 Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/social-media-dashboard.git
cd social-media-dashboard
```

2. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your dataset file named `data.jsonl` in the project root.

4. (Optional) Set your OpenAI API key in `.streamlit/secrets.toml` for AI features:

```toml
[general]
OPENAI_API_KEY = "your-openai-api-key"
```

---

## 🚀 Running the App

```bash
streamlit run main.py
```

---

## 📁 File Structure

```
📦 social-media-dashboard
├── main.py                # Main Streamlit dashboard logic
├── requirements.txt       # Python dependencies
├── data.jsonl             # Your social media dataset
├── ai_insights.py         # Optional AI insights generator
└── spacy_models/          # spaCy model cache directory
```

---

## 📝 Data Format

Your input file should be in `.jsonl` format (one JSON per line). Each JSON object should contain at least:

- `created_at`, `timestamp`, or equivalent datetime field
- `content`, `text`, `body`, etc. with post content
- Optional: `author`, `subreddit`, `title`

Example:
```json
{"created_at": "2024-10-21T14:30:00", "content": "This is a sample post", "author": "user123", "subreddit": "technology"}
```

---

## ⚙️ Technologies Used

- **Python 3.10+**
- Streamlit
- pandas, numpy
- NLTK, spaCy
- scikit-learn, TF-IDF, NMF
- matplotlib, seaborn, plotly, altair
- networkx, pyvis
- OpenAI GPT (optional)

---

## 📌 Notes

- Some advanced features (e.g., spaCy NER) will download models on first run.
- AI Insights tab will be disabled if OpenAI API key is not provided.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
