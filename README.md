# 📈 Financial News Sentiment Analyzer

An AI-powered web app that classifies financial news headlines as **Positive**, **Negative**, or **Neutral** in real-time using Machine Learning.

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)

---

## 🚀 Features

- 🔍 **Real-time sentiment classification** of any financial headline
- 📊 **Confidence score** with probability breakdown pie chart
- 📰 **Live news ticker** showing latest business headlines (via NewsAPI)
- 🧪 **Model evaluation dashboard** — Accuracy, Precision, Recall, F1-Score per class
- 💡 **Example headlines** to try instantly
- 🎨 Clean, modern dark UI built with Streamlit

---

## 🧠 Machine Learning Pipeline

| Step | Method |
|------|--------|
| Text Cleaning | Lowercase, remove punctuation, NLTK stopword removal |
| Feature Extraction | TF-IDF Vectorizer (10,000 features, 1-3 ngrams) |
| Model | LinearSVC wrapped with CalibratedClassifierCV |
| Dataset | Financial PhraseBank — Sentences_75Agree (3,453 rows) |
| Accuracy | **82.8%** on 20% held-out test set |

---

## 📁 Project Structure

```
financial_sentiment/
├── app.py              # Streamlit web app
├── train.py            # Model training script
├── preprocess.py       # Text cleaning utilities
├── model.pkl           # Trained LinearSVC model
├── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── all-data.csv        # Kaggle Financial PhraseBank dataset
└── requirements.txt    # Python dependencies
```

---

## 🛠️ Tech Stack

- **Python 3.9**
- **Scikit-learn** — LinearSVC, TF-IDF, CalibratedClassifierCV
- **Streamlit** — Web UI
- **Plotly** — Pie chart visualization
- **NLTK** — Text preprocessing
- **NewsAPI** — Live business headlines

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/nxrmal03/financial-sentiment.git
cd financial-sentiment

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

> Model is pre-trained. To retrain: `python train.py`

---

## 📊 Model Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Positive | ~85% | ~83% | ~84% |
| Negative | ~82% | ~79% | ~80% |
| Neutral | ~83% | ~86% | ~84% |
| **Overall Accuracy** | | | **82.8%** |

---

## 📌 Dataset

[Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) — Sentences with 75% annotator agreement for cleaner labels.

---

## 👨‍💻 Author

Made by **Nirmal** as part of an AI-Based Financial Risk & Investment Research Assistant project.
