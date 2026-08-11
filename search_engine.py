"""
search_engine.py
=================
KES Semantic Similarity Search - core engine.

This ports the pipeline built in the Colab notebook (preprocessing ->
Word2Vec training -> document embeddings -> cosine similarity search)
into a reusable module with no Colab / notebook dependencies, so it can
run inside a normal Flask web app.

Two entry points matter to the rest of the app:

  build_index(excel_path, data_dir)   -> trains a fresh index from an
                                          uploaded .xlsx and saves it
  load_index(data_dir)                -> loads a previously built index
  search(query, index, top_k, threshold) -> ranked results

Expected input Excel columns (same as the notebook): document_id,
title, raw_text, keywords, category. Rows missing raw_text/title are
skipped rather than crashing the whole build.
"""

import os
import re
import string
import pickle

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_COLUMNS = ["document_id", "title", "raw_text", "keywords", "category"]

MODEL_FILENAME = "kes_word2vec.model"
VECTORS_FILENAME = "kes_document_vectors.npy"
METADATA_FILENAME = "kes_document_metadata.csv"
INFO_FILENAME = "kes_search_metadata.pkl"


def ensure_nltk_resources():
    """Download the NLTK resources the pipeline needs, once."""
    resources = [
        ("stopwords", "corpora/stopwords"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]
    for name, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_stop_words = None
_lemmatizer = None


def _get_nlp_tools():
    global _stop_words, _lemmatizer
    if _stop_words is None:
        ensure_nltk_resources()
        _stop_words = set(stopwords.words("english"))
        _lemmatizer = WordNetLemmatizer()
    return _stop_words, _lemmatizer


def preprocess_text(text):
    """Clean, tokenize, de-stopword, and lemmatize a piece of text.
    Same steps as the notebook's preprocess_text/preprocess_query."""
    stop_words, lemmatizer = _get_nlp_tools()

    text = str(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens


def preprocess_text_to_string(text):
    return " ".join(preprocess_text(text))


def document_vector(model, tokens):
    """Average the Word2Vec vectors of tokens present in the vocabulary."""
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


# ---------------------------------------------------------------------
# Building a new index from an uploaded spreadsheet
# ---------------------------------------------------------------------

def build_index(excel_path, data_dir, sheet_name=0, progress_cb=None):
    """
    Build a full KES search index from a raw manuscript spreadsheet and
    save it into data_dir. Returns a summary dict.

    progress_cb, if given, is called with short status strings so a
    caller (e.g. a web request) can report progress.
    """
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    os.makedirs(data_dir, exist_ok=True)

    report("Reading spreadsheet...")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}"
        )

    for col in ["title", "raw_text", "keywords", "category"]:
        df[col] = df[col].fillna("").astype(str)

    df = df[df["raw_text"].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows with non-empty 'raw_text' were found.")

    report("Preprocessing text (this can take a minute)...")
    df["preprocessed_text"] = df["raw_text"].apply(preprocess_text_to_string)
    df["processed_title"] = df["title"].apply(preprocess_text_to_string)
    df["processed_keywords"] = df["keywords"].apply(preprocess_text_to_string)
    df["search_text"] = (
        df["processed_title"] + " " + df["preprocessed_text"] + " " + df["processed_keywords"]
    ).str.strip()

    sentences = [t.split() for t in df["search_text"] if len(t.split()) > 0]
    if not sentences:
        raise ValueError("No usable text remained after preprocessing.")

    report(f"Training Word2Vec on {len(sentences)} documents...")
    word2vec_model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=100,
        seed=42,
    )

    report("Building document embeddings...")
    doc_vectors = np.array(
        [document_vector(word2vec_model, t.split()) for t in df["search_text"]]
    )

    report("Saving index files...")
    word2vec_model.save(os.path.join(data_dir, MODEL_FILENAME))
    np.save(os.path.join(data_dir, VECTORS_FILENAME), doc_vectors)

    metadata = df[["document_id", "title", "category", "search_text"]].copy()
    metadata.to_csv(os.path.join(data_dir, METADATA_FILENAME), index=False)

    info = {
        "vector_size": word2vec_model.vector_size,
        "num_documents": len(doc_vectors),
        "vocabulary_size": len(word2vec_model.wv),
    }
    with open(os.path.join(data_dir, INFO_FILENAME), "wb") as f:
        pickle.dump(info, f)

    report("Done.")
    return info


# ---------------------------------------------------------------------
# Loading a previously built index
# ---------------------------------------------------------------------

class SearchIndex:
    def __init__(self, model, document_vectors, metadata, info):
        self.model = model
        self.document_vectors = document_vectors
        self.metadata = metadata
        self.info = info


def index_exists(data_dir):
    return all(
        os.path.exists(os.path.join(data_dir, fn))
        for fn in (MODEL_FILENAME, VECTORS_FILENAME, METADATA_FILENAME)
    )


def load_index(data_dir):
    if not index_exists(data_dir):
        return None

    ensure_nltk_resources()
    model = Word2Vec.load(os.path.join(data_dir, MODEL_FILENAME))
    document_vectors = np.load(os.path.join(data_dir, VECTORS_FILENAME))
    metadata = pd.read_csv(os.path.join(data_dir, METADATA_FILENAME))

    info_path = os.path.join(data_dir, INFO_FILENAME)
    if os.path.exists(info_path):
        with open(info_path, "rb") as f:
            info = pickle.load(f)
    else:
        info = {
            "vector_size": model.vector_size,
            "num_documents": len(document_vectors),
            "vocabulary_size": len(model.wv),
        }

    return SearchIndex(model, document_vectors, metadata, info)


# ---------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------

def query_to_vector(query, model):
    tokens = preprocess_text(query)
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return None, tokens
    return np.mean(vectors, axis=0), tokens


def search(query, index: SearchIndex, top_k=5, threshold=0.0):
    """
    Returns (results_list, tokens). results_list is a list of dicts,
    highest similarity first, filtered by threshold (0-1 scale).
    """
    query_vector, tokens = query_to_vector(query, index.model)

    if query_vector is None:
        return [], tokens

    query_vector = query_vector.reshape(1, -1)
    similarities = cosine_similarity(query_vector, index.document_vectors)[0]

    results = index.metadata.copy()
    results["similarity_score"] = similarities
    results = results.sort_values(by="similarity_score", ascending=False)
    results = results[results["similarity_score"] >= threshold]
    results = results.head(top_k)

    records = []
    for _, row in results.iterrows():
        records.append(
            {
                "document_id": row["document_id"],
                "title": row["title"],
                "category": row["category"],
                "similarity_score": round(float(row["similarity_score"]), 4),
                "similarity_percentage": round(float(row["similarity_score"]) * 100, 2),
            }
        )
    return records, tokens
