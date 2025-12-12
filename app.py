import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import torch
import streamlit as st
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import nltk

st.set_page_config(page_title="iPhone Review Analysis", layout="wide")
st.title("📱 iPhone Review Analysis System")

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
os.makedirs(MODEL_DIR, exist_ok=True)

USER_DB = "users.db"

def get_user_db():
    return sqlite3.connect(USER_DB, check_same_thread=False)

def init_user_db():
    conn = get_user_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def register_user(username, email, password):
    conn = get_user_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)",
            (username, email, password, datetime.now().isoformat())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = get_user_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )
    user = cur.fetchone()
    conn.close()
    return user is not None

init_user_db()

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

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

menu = st.sidebar.selectbox("Menu", ["Login", "Signup", "Predict", "Logout"])

if menu == "Signup":
    st.subheader("Create Account")
    u = st.text_input("Username")
    e = st.text_input("Email")
    p = st.text_input("Password", type="password")
    if st.button("Signup"):
        if u and e and p:
            if register_user(u, e, p):
                st.success("User registered successfully")
            else:
                st.error("Username or email already exists")
        else:
            st.warning("Fill all fields")

elif menu == "Login":
    st.subheader("Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(u, p):
            st.session_state.logged_in = True
            st.session_state.user = u
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

elif menu == "Predict":
    if not st.session_state.logged_in:
        st.warning("Please login first")
    else:
        st.subheader(f"Welcome, {st.session_state.user}")
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.dataframe(df.head())
            text_col = st.selectbox("Select text column", df.select_dtypes(include="object").columns)
            if st.button("Run Prediction"):
                with st.spinner("Running prediction..."):
                    X_text = preprocess_data(df[[text_col]])
                    features = electra_feature_extraction(X_text)
                    df_out = df.copy()
                    df_out["Predicted_title"] = le_title.inverse_transform(title_model.predict(features))
                    df_out["Predicted_rating"] = le_rating.inverse_transform(rating_model.predict(features))
                st.dataframe(df_out)
                st.download_button(
                    "Download Results",
                    df_out.to_csv(index=False).encode("utf-8"),
                    "predicted_output.csv",
                    "text/csv"
                )

elif menu == "Logout":
    st.session_state.logged_in = False
    st.session_state.user = None
    st.success("Logged out successfully")
