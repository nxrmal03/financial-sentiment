import streamlit as st
import pickle
import re
import nltk
import requests
import pandas as pd
import plotly.graph_objects as go
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

st.set_page_config(page_title="Financial News Sentiment Analyzer", page_icon="📈", layout="centered")

NEWS_API_KEY = "c160240e251b46228c6606d3a942594b"

FALLBACK_HEADLINES = [
    "📈 Apple reports record breaking quarterly revenue, shares surge 8%",
    "📉 Oil prices fall sharply amid global demand concerns",
    "📊 Federal Reserve holds interest rates steady for third consecutive meeting",
    "💹 Tesla shares jump 12% after strong delivery numbers beat expectations",
    "⚠️ Inflation rises to 4.2%, higher than analyst forecasts",
]

@st.cache_data(ttl=300)
def fetch_live_news():
    try:
        url = (
            "https://newsapi.org/v2/top-headlines"
            "?category=business&language=en&pageSize=20"
            f"&apiKey={NEWS_API_KEY}"
        )
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            headlines = [a["title"] for a in articles if a.get("title") and "[Removed]" not in a["title"]]
            if headlines:
                icons = ["📈", "📉", "💹", "📊", "💰", "⚠️"]
                return [f"{icons[i % len(icons)]} {h}" for i, h in enumerate(headlines)]
    except Exception:
        pass
    return FALLBACK_HEADLINES

NEWS_HEADLINES = fetch_live_news()
ticker_text = "   ●   ".join(NEWS_HEADLINES)

st.markdown(f"""
<style>
.ticker-wrap {{
    width: 100%;
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    padding: 10px 0;
    overflow: hidden;
    border-radius: 8px;
    margin-bottom: 20px;
    border: 1px solid #2c5364;
}}
.ticker {{
    display: inline-block;
    white-space: nowrap;
    animation: ticker-scroll 200s linear infinite;
    color: #00d4aa;
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 0.3px;
}}
.ticker:hover {{
    animation-play-state: paused;
}}
@keyframes ticker-scroll {{
    0%   {{ transform: translateX(100vw); }}
    100% {{ transform: translateX(-100%); }}
}}
.ticker-label {{
    display: inline-block;
    background: #00d4aa;
    color: #0f2027;
    font-size: 12px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 12px;
    letter-spacing: 1px;
}}
</style>
<div class="ticker-wrap">
    <div class="ticker">
        <span class="ticker-label">LIVE</span>
        {ticker_text}
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <span class="ticker-label">LIVE</span>
        {ticker_text}
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@st.cache_data
def get_evaluation_metrics():
    rows = []
    try:
        with open('FinancialPhraseBank/Sentences_75Agree.txt', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if '@' in line:
                    parts = line.rsplit('@', 1)
                    if len(parts) == 2:
                        sentence, lbl = parts[0].strip(), parts[1].strip()
                        if lbl in ('positive', 'negative', 'neutral'):
                            rows.append({'sentence': sentence, 'sentiment': lbl})
    except FileNotFoundError:
        try:
            df = pd.read_csv('all-data.csv', encoding='latin-1', header=None,
                             names=['sentiment', 'sentence'])
            rows = df[df['sentiment'].isin(['positive','negative','neutral'])].to_dict('records')
        except Exception:
            return None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    model, vectorizer = load_model()
    df['clean'] = df['sentence'].apply(clean_text)
    _, X_test, _, y_test = train_test_split(
        df['clean'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, report


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return ' '.join(tokens)


def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba) * 100
    return prediction, confidence


LABEL_COLOR = {
    'positive': ('#2ecc71', 'rgba(46,204,113,0.15)', '✅ Positive'),
    'negative': ('#e74c3c', 'rgba(231,76,60,0.15)',  '❌ Negative'),
    'neutral':  ('#adb5bd', 'rgba(173,181,189,0.15)','➖ Neutral'),
}

EXAMPLES = [
    {"text": "Company reports record profits for Q3",                     "icon": "📰"},
    {"text": "Stock market crashes amid rising inflation fears",          "icon": "📰"},
    {"text": "Central bank keeps interest rates unchanged",               "icon": "📰"},
    {"text": "CEO resigns amid fraud investigation",                      "icon": "📰"},
    {"text": "Firm announces dividend increase for shareholders",         "icon": "📰"},
    {"text": "Annual general meeting scheduled for next month",           "icon": "📰"},
]

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Page header */
.app-title {
    font-size: 2.2rem; font-weight: 900; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #00d4aa, #0099ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.app-sub {
    color: rgba(255,255,255,0.45); font-size: 0.95rem; margin-bottom: 28px;
}
/* Section labels */
.section-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 1.4px;
    text-transform: uppercase; color: rgba(255,255,255,0.35); margin-bottom: 10px;
}
/* Example buttons — strict equal height */
div[data-testid="column"] button {
    width: 100% !important;
    height: 76px !important;
    min-height: 76px !important;
    max-height: 76px !important;
    white-space: normal !important;
    text-align: left !important;
    padding: 10px 13px !important;
    border-radius: 12px !important;
    font-size: 12.5px !important;
    line-height: 1.4 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.04) !important;
    transition: all 0.2s ease !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    align-items: flex-start !important;
}
div[data-testid="column"] button p {
    overflow: hidden !important;
    display: -webkit-box !important;
    -webkit-line-clamp: 3 !important;
    -webkit-box-orient: vertical !important;
    margin: 0 !important;
}
div[data-testid="column"] button:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(0,212,170,0.4) !important;
    background: rgba(0,212,170,0.07) !important;
    box-shadow: 0 4px 16px rgba(0,212,170,0.12) !important;
}
/* Stat cards — equal height */
.stat-card {
    height: 110px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    box-sizing: border-box !important;
}
/* Class metric cards — equal height */
.class-card {
    min-height: 220px !important;
    box-sizing: border-box !important;
}
/* Definition cards — strict equal height */
.def-card {
    height: 110px !important;
    min-height: 110px !important;
    max-height: 110px !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
}
.def-body {
    overflow: hidden !important;
    display: -webkit-box !important;
    -webkit-line-clamp: 3 !important;
    -webkit-box-orient: vertical !important;
}
/* Result card */
.result-card {
    border-radius: 16px; padding: 22px 26px;
    display: flex; flex-direction: column; justify-content: center;
    height: 100%;
}
.result-label {
    font-size: 1.6rem; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 14px;
}
.conf-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: rgba(255,255,255,0.4); margin-bottom: 6px;
}
.conf-value {
    font-size: 2rem; font-weight: 800; margin-bottom: 10px;
}
/* How it works cards */
.how-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1); border-radius: 14px;
    padding: 20px 18px; text-align: center;
    height: 140px;
    display: flex; flex-direction: column; justify-content: center;
    box-sizing: border-box;
}
.how-num {
    font-size: 1.5rem; font-weight: 800; margin-bottom: 8px;
}
.how-title { font-size: 0.95rem; font-weight: 700; margin-bottom: 6px; }
.how-body  { font-size: 0.8rem; color: rgba(255,255,255,0.5); line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── App header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">📈 Financial Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Classify financial headlines as Positive, Negative, or Neutral using ML</div>', unsafe_allow_html=True)

try:
    model, vectorizer = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("⚠️ Model not found. Please run `python train.py` first.")

# ── Example cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">💡 Try an example headline</div>', unsafe_allow_html=True)

if "example_text" not in st.session_state:
    st.session_state.example_text = ""

row1 = st.columns(3)
row2 = st.columns(3)
rows = row1 + row2

for i, ex in enumerate(EXAMPLES):
    btn_label = f"{ex['icon']} {ex['text']}"
    if rows[i].button(btn_label, key=f"ex_{i}"):
        st.session_state.example_text = ex["text"]

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">✍️ Enter your headline</div>', unsafe_allow_html=True)

with st.form("sentiment_form"):
    user_input = st.text_area(
        label="",
        value=st.session_state.example_text,
        placeholder="e.g. Apple reports record quarterly revenue, shares surge 8%...",
        height=95,
        label_visibility="collapsed"
    )
    submitted = st.form_submit_button("🔍 Analyze Sentiment", type="primary", use_container_width=True, disabled=not model_loaded)

# ── Result ─────────────────────────────────────────────────────────────────────
if submitted:
    if user_input.strip():
        prediction, confidence = predict(user_input, model, vectorizer)
        accent, bg, emoji_label = LABEL_COLOR[prediction]

        vec = vectorizer.transform([clean_text(user_input)])
        proba = model.predict_proba(vec)[0]
        classes = model.classes_
        color_map = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#adb5bd"}

        fig = go.Figure(go.Pie(
            labels=[c.capitalize() for c in classes],
            values=[round(p * 100, 1) for p in proba],
            marker_colors=[color_map[c] for c in classes],
            hole=0.45,
            textinfo="label+percent",
            textfont_size=13,
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), height=260,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
        )

        st.markdown('<div class="section-label" style="margin-top:20px">📊 Analysis Result</div>', unsafe_allow_html=True)
        res_col, chart_col = st.columns([1, 1], gap="medium")

        with res_col:
            st.markdown(
                f'<div class="result-card" style="background:{bg};border:1px solid {accent}55">'
                f'<div class="result-label" style="color:{accent}">{emoji_label}</div>'
                f'<div class="conf-label">Confidence Score</div>'
                f'<div class="conf-value" style="color:{accent}">{confidence:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.progress(int(confidence))

        with chart_col:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter a headline first.")

# ── How it works ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label" style="text-align:center;margin-bottom:16px">⚙️ How it works</div>', unsafe_allow_html=True)
h1, h2, h3 = st.columns(3, gap="medium")
for col, num, color, title, body in [
    (h1, "01", "#00d4aa", "Preprocess",  "Lowercase, remove punctuation & stopwords using NLTK"),
    (h2, "02", "#0099ff", "TF-IDF",      "Convert cleaned text into numerical feature vectors"),
    (h3, "03", "#a855f7", "LinearSVC",   "Calibrated classifier predicts sentiment with 82.8% accuracy"),
]:
    col.markdown(
        f'<div class="how-card">'
        f'<div class="how-num" style="color:{color}">{num}</div>'
        f'<div class="how-title">{title}</div>'
        f'<div class="how-body">{body}</div>'
        f'</div>', unsafe_allow_html=True)

st.markdown("---")

eval_result = get_evaluation_metrics()
if eval_result:
    acc, report = eval_result
    classes_order = ["positive", "negative", "neutral"]
    color_map = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#adb5bd"}
    icons = {"positive": "✅", "negative": "❌", "neutral": "➖"}

    st.markdown("""
    <style>
    .eval-header {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 4px;
        background: linear-gradient(90deg, #00d4aa, #0099ff, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .eval-subheader {
        text-align: center;
        color: rgba(255,255,255,0.45);
        font-size: 0.85rem;
        margin-bottom: 28px;
        letter-spacing: 0.5px;
    }
    .stat-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 20px 18px;
        text-align: center;
        transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-3px); }
    .stat-label {
        font-size: 0.75rem;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.45);
        margin-bottom: 8px;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
    }
    .stat-sub { font-size: 0.75rem; color: rgba(255,255,255,0.35); margin-top: 4px; }
    .class-card {
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 4px;
    }
    .class-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 14px;
        letter-spacing: 0.5px;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .metric-name { font-size: 0.78rem; color: rgba(255,255,255,0.55); text-transform: uppercase; letter-spacing: 0.8px; }
    .metric-val { font-size: 1rem; font-weight: 700; }
    .bar-track {
        height: 5px;
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
        margin-top: 3px;
        overflow: hidden;
    }
    .def-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
        padding: 16px;
    }
    .def-title { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.8px; margin-bottom: 6px; }
    .def-body { font-size: 0.78rem; color: rgba(255,255,255,0.5); line-height: 1.5; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="eval-header">📊 Model Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="eval-subheader">LinearSVC · TF-IDF · CalibratedClassifierCV · Financial PhraseBank 75% Agreement</div>', unsafe_allow_html=True)

    # Top stats row
    s1, s2, s3, s4 = st.columns(4)
    macro = report.get("macro avg", {})
    for col, label, val, sub, color in [
        (s1, "Accuracy",   f"{acc*100:.1f}%",                        "Overall correct",          "#00d4aa"),
        (s2, "Macro F1",   f"{macro.get('f1-score',0)*100:.1f}%",    "Avg across classes",        "#0099ff"),
        (s3, "Precision",  f"{macro.get('precision',0)*100:.1f}%",   "Macro avg",                 "#a855f7"),
        (s4, "Recall",     f"{macro.get('recall',0)*100:.1f}%",      "Macro avg",                 "#f59e0b"),
    ]:
        col.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-label">{label}</div>'
            f'<div class="stat-value" style="color:{color}">{val}</div>'
            f'<div class="stat-sub">{sub}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Per-class cards
    cc1, cc2, cc3 = st.columns(3)
    for col, cls in zip([cc1, cc2, cc3], classes_order):
        r = report.get(cls, {})
        c = color_map[cls]
        prec = r.get("precision", 0) * 100
        rec  = r.get("recall",    0) * 100
        f1   = r.get("f1-score",  0) * 100
        sup  = int(r.get("support", 0))
        col.markdown(
            f'<div class="class-card" style="background:linear-gradient(135deg,rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.18),rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.05));border:1px solid {c}44">'
            f'<div class="class-title" style="color:{c}">{icons[cls]} {cls.capitalize()} &nbsp;<span style="font-size:0.72rem;color:rgba(255,255,255,0.35);font-weight:400">{sup} samples</span></div>'
            f'<div class="metric-row"><span class="metric-name">Precision</span><span class="metric-val" style="color:{c}">{prec:.1f}%</span></div>'
            f'<div class="bar-track"><div style="height:5px;width:{prec:.0f}%;background:{c};border-radius:4px"></div></div>'
            f'<div style="margin-bottom:10px"></div>'
            f'<div class="metric-row"><span class="metric-name">Recall</span><span class="metric-val" style="color:{c}">{rec:.1f}%</span></div>'
            f'<div class="bar-track"><div style="height:5px;width:{rec:.0f}%;background:{c};border-radius:4px"></div></div>'
            f'<div style="margin-bottom:10px"></div>'
            f'<div class="metric-row"><span class="metric-name">F1-Score</span><span class="metric-val" style="color:{c}">{f1:.1f}%</span></div>'
            f'<div class="bar-track"><div style="height:5px;width:{f1:.0f}%;background:{c};border-radius:4px"></div></div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Definitions row
    st.markdown('<div style="font-size:0.8rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,0.4);margin-bottom:10px">Metric Definitions</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    for col, title, body in [
        (d1, "🎯 Accuracy",   "Correct predictions ÷ total predictions. The simplest overall score."),
        (d2, "🔵 Precision",  "Of all headlines predicted as Positive, how many were actually Positive?"),
        (d3, "🟢 Recall",     "Of all actual Positive headlines, how many did the model correctly find?"),
        (d4, "⚖️ F1-Score",   "Harmonic mean of Precision & Recall. Best single metric for imbalanced classes."),
    ]:
        col.markdown(
            f'<div class="def-card"><div class="def-title">{title}</div><div class="def-body">{body}</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:center;font-size:0.75rem;color:rgba(255,255,255,0.25);letter-spacing:0.5px">'
        f'Evaluated on 20% held-out test set · 80/20 stratified split · random_state=42 · 3,453 total samples'
        f'</div>', unsafe_allow_html=True)
else:
    st.warning("Could not compute evaluation metrics — model or data not found.")
