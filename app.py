import os
import joblib
import numpy as np
import pandas as pd
import torch
import streamlit as st
import sqlite3
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import nltk

st.set_page_config(page_title="iPhone Review Analysis", layout="wide")
st.title("📱 iPhone Review Analysis using ELECTRA + ETC")

NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

def download_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet")
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, download_dir=NLTK_DATA_DIR, quiet=True)

download_nltk_resources()

MODEL_DIR = "model"
DB_PATH = "predictions.db"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            raw_text TEXT,
            predicted_title TEXT,
            predicted_rating TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_predictions(df, text_col):
    conn = get_db_connection()
    cur = conn.cursor()
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO predictions (timestamp, raw_text, predicted_title, predicted_rating)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            str(row[text_col]),
            str(row["Predicted_title"]),
            str(row["Predicted_rating"])
        ))
    conn.commit()
    conn.close()

def load_predictions(limit=100):
    conn = get_db_connection()
    df = pd.read_sql(
        f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df

init_db()

def preprocess_data(df):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    def clean_text(text):
        text = str(text).lower()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
        return " ".join(tokens)
    for col in df.columns:
        df[col] = df[col].apply(clean_text)
    return df.astype(str).agg(" ".join, axis=1).tolist()

@st.cache_resource
def load_electra():
    tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    model = AutoModel.from_pretrained("google/electra-base-discriminator")
    model.eval()
    return tokenizer, model

def electra_feature_extraction(texts, batch_size=16):
    tokenizer, model = load_electra()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = mask.sum(dim=1)
        mean_pooled = summed / counts
        all_embeddings.append(mean_pooled.cpu().numpy())
    return np.vstack(all_embeddings)

title_model = joblib.load(os.path.join(MODEL_DIR, "ELECTRA_word_embeddings_title_ETC_model.pkl"))
rating_model = joblib.load(os.path.join(MODEL_DIR, "ELECTRA_word_embeddings_rating_ETC_model.pkl"))

labels1 = [
    'Awesome','Bad quality','Best in the market!','Brilliant','Classy product',
    'Decent product','Delightful','Does the job','Excellent','Fabulous!','Fair',
    'Good','Good choice','Good quality product','Great product',
    'Highly recommended','Just okay','Just wow!','Mind-blowing purchase',
    'Must buy!','Nice','Nice product','Perfect product!','Pretty good',
    'Really Nice','Simply awesome',
    "Simply awesome. Go for it. it's an iPhone after all.",
    'Super!','Terrific','Terrific purchase','Value-for-money','Very Good',
    'Wonderful','Worth every penny','Worth the money'
]

labels2 = [3.0, 4.0, 5.0]

le_title = LabelEncoder().fit(labels1)
le_rating = LabelEncoder().fit([str(x) for x in labels2])

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    text_column = st.selectbox("Select text column", df.select_dtypes(include="object").columns)
    if st.button("Run Prediction"):
        with st.spinner("Running prediction..."):
            X_text = preprocess_data(df[[text_column]])
            features = electra_feature_extraction(X_text)
            df_out = df.copy()
            df_out["Predicted_title"] = le_title.inverse_transform(title_model.predict(features))
            df_out["Predicted_rating"] = le_rating.inverse_transform(rating_model.predict(features))
            save_predictions(df_out, text_column)
        st.dataframe(df_out)
        st.download_button(
            "Download Results",
            df_out.to_csv(index=False).encode("utf-8"),
            "predicted_output.csv",
            "text/csv"
        )

if st.button("View Stored Predictions"):
    st.dataframe(load_predictions())
