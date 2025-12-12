import os
import joblib
import numpy as np
import pandas as pd
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import nltk

# --------------------------------------------------
# INITIAL SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="iPhone Review Analysis",
    layout="wide"
)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

MODEL_DIR = "model"
DATASET_DIR = "Dataset"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
def preprocess_data(df, target_cols=None):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = str(text).lower()
        tokens = word_tokenize(text)
        tokens = [
            lemmatizer.lemmatize(t)
            for t in tokens if t.isalnum() and t not in stop_words
        ]
        return ' '.join(tokens)

    if target_cols:
        df = df.drop(
            columns=[c for c in target_cols if c in df.columns],
            errors='ignore'
        )

    text_cols = df.select_dtypes(include='object').columns
    for col in text_cols:
        df[f'processed_{col}'] = df[col].apply(clean_text)

    processed_cols = [c for c in df.columns if c.startswith("processed_")]
    X_text = df[processed_cols].astype(str).agg(" ".join, axis=1)

    return X_text.tolist()

# --------------------------------------------------
# ELECTRA FEATURE EXTRACTION (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_electra():
    tokenizer = AutoTokenizer.from_pretrained(
        "google/electra-base-discriminator"
    )
    model = AutoModel.from_pretrained(
        "google/electra-base-discriminator"
    )
    model.eval()
    return tokenizer, model

def electra_feature_extraction(texts, batch_size=16):
    tokenizer, model = load_electra()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**encoded)

        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        mask = mask.expand(token_embeddings.size()).float()

        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = mask.sum(dim=1)
        mean_pooled = summed / counts

        all_embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(all_embeddings)

# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------
title_model_path = os.path.join(
    MODEL_DIR, "ELECTRA_word_embeddings_title_ETC_model.pkl"
)
rating_model_path = os.path.join(
    MODEL_DIR, "ELECTRA_word_embeddings_rating_ETC_model.pkl"
)

if not os.path.exists(title_model_path) or not os.path.exists(rating_model_path):
    st.error("Trained model files not found in model directory")
    st.stop()

final_model = {
    "title": joblib.load(title_model_path),
    "rating": joblib.load(rating_model_path)
}

# --------------------------------------------------
# LABEL ENCODERS
# --------------------------------------------------
labels1 = [
    'Awesome','Bad quality','Best in the market!','Brilliant',
    'Classy product','Decent product','Delightful','Does the job',
    'Excellent','Fabulous!','Fair','Good','Good choice',
    'Good quality product','Great product','Highly recommended',
    'Just okay','Just wow!','Mind-blowing purchase','Must buy!',
    'Nice','Nice product','Perfect product!','Pretty good',
    'Really Nice','Simply awesome',
    "Simply awesome. Go for it. it's an iPhone after all.",
    'Super!','Terrific','Terrific purchase','Value-for-money',
    'Very Good','Wonderful','Worth every penny','Worth the money'
]

labels2 = [3.0, 4.0, 5.0]

le_title = LabelEncoder().fit(labels1)
le_rating = LabelEncoder().fit([str(x) for x in labels2])

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📱 iPhone Review Analysis using ELECTRA + ETC")
st.write("Upload a CSV file to predict **title sentiment** and **rating**.")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        with st.spinner("Processing reviews..."):
            X_text = preprocess_data(df, target_cols=["title", "rating"])
            features = electra_feature_extraction(X_text)

            df_out = df.copy()
            df_out["Predicted_title"] = le_title.inverse_transform(
                final_model["title"].predict(features)
            )
            df_out["Predicted_rating"] = le_rating.inverse_transform(
                final_model["rating"].predict(features)
            )

        st.success("Prediction completed successfully")
        st.subheader("✅ Prediction Results")
        st.dataframe(df_out)

        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="predicted_output.csv",
            mime="text/csv"
        )
