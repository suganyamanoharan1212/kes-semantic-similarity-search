# KES Semantic Similarity Search — Web App

This is a local Flask web app version of the semantic search pipeline
from your Colab notebook (NLTK preprocessing → Word2Vec → cosine
similarity search). ## Project Overview

KES Semantic Similarity Search is an NLP-based document retrieval web application built with Python and Flask.

The system converts document text into semantic representations using Word2Vec and compares a user's query with document vectors using cosine similarity. Results are ranked from the most semantically relevant document to the least relevant.

## NLP Pipeline

Raw Excel Dataset
→ Text Preprocessing
→ Tokenization & Lemmatization
→ Word2Vec Training
→ Document Embeddings
→ Cosine Similarity
→ Ranked Search Results

## Technologies Used

- Python
- Flask
- NLTK
- Gensim Word2Vec
- NumPy
- Pandas
- Scikit-learn
- HTML/CSS

## Key Features

- Upload raw Excel manuscript data
- NLP text preprocessing
- Word2Vec model training
- Document-level emb- eddings
- Semantic similarity search
- Cosine similarity scoring
- Ranked search results
- Flask-based web interface

## Example

User Query:

"How are manuscripts reviewed by editors?"

The system compares the query with indexed documents and returns the most semantically similar documents with similarity scores.

## Project Result

The application successfully builds a searchable semantic index and returns ranked results through a local Flask web application.No more manual `files.upload()` steps — you upload
your spreadsheet once through the browser, and it stays indexed on disk.

## 1. Open in VS Code

Unzip/copy the `kes-search-app` folder, then in VS Code: **File → Open
Folder…** and select it.

## 2. Create a virtual environment (recommended)

Open a terminal in VS Code (`` Ctrl+` ``) and run:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

VS Code may prompt "Select interpreter" — pick the `venv` one.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first time you run the app it also downloads a few small NLTK data
packages (stopwords, punkt, wordnet) automatically — that needs an
internet connection once.

## 4. Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Leave the terminal
running — that's your local web server.

## 5. Build the index

On first launch there's no model yet. In the **Intake** panel, upload
the same kind of spreadsheet the notebook used, with these columns:

| document_id | title | raw_text | keywords | category |
|---|---|---|---|---|

Click **Build index**. This runs the exact same steps as your notebook
(clean text → lemmatize → train Word2Vec → average word vectors into
document vectors) and saves the results under `./data/`:

- `kes_word2vec.model`
- `kes_document_vectors.npy`
- `kes_document_metadata.csv`
- `kes_search_metadata.pkl`

Next time you start `python app.py`, it loads these automatically — no
need to rebuild unless your dataset changes.

## 6. Search

Type a query in the **Reading room** panel, set how many results and
what minimum similarity % you want, and hit **Search**. Each result
card shows the matched title, category, and a similarity score.

## Project layout

```
kes-search-app/
├── app.py              Flask routes (/, /api/status, /api/build, /api/search)
├── search_engine.py     Preprocessing, Word2Vec training, cosine search
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
├── data/                 Saved index files land here (created at runtime)
└── uploads/              Raw uploaded spreadsheets land here (created at runtime)
```

## Notes / things you may want to change

- **Reusing your existing notebook outputs**: if you already have
  `kes_word2vec.model`, `kes_document_vectors.npy`, and
  `kes_document_metadata.csv` from the notebook, just drop them into
  `./data/` before starting the app — it will load them directly and
  skip the build step entirely.
- **Port already in use**: change `port=5000` in `app.py` (last line).
- **Bigger datasets**: Word2Vec training on a large `raw_text` corpus
  can take a while — the UI shows a "Building…" state while it works,
  and the server logs progress to the terminal.
- **Deploying beyond localhost**: `app.run(debug=True, ...)` is for
  local development only. For anything beyond your own machine, run it
  behind a production server (e.g. `waitress` on Windows or `gunicorn`
  on macOS/Linux) and turn `debug` off.
