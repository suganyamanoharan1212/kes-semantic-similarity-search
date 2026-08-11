"""
KES Semantic Similarity Search - Flask web app
================================================
Run locally:

    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000 in your browser.

On first run there is no index yet - use the "Build index" panel in the
web UI to upload the same kind of manuscript spreadsheet the Colab
notebook expected (columns: document_id, title, raw_text, keywords,
category). That trains a Word2Vec model and saves it under ./data, so
next time you start the app it loads instantly.
"""

import os
import traceback

from flask import Flask, request, jsonify, render_template

import search_engine as se

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".xlsx", ".xls"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload cap

# In-memory index, loaded at startup if data files already exist.
_index = se.load_index(DATA_DIR)


def index_status():
    if _index is None:
        return {"ready": False}
    return {
        "ready": True,
        "num_documents": _index.info.get("num_documents"),
        "vocabulary_size": _index.info.get("vocabulary_size"),
        "vector_size": _index.info.get("vector_size"),
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify(index_status())


@app.route("/api/build", methods=["POST"])
def api_build():
    global _index

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Attach an .xlsx file under 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Please upload a .xlsx or .xls file."}), 400

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        info = se.build_index(save_path, DATA_DIR)
        _index = se.load_index(DATA_DIR)
        return jsonify({"ok": True, "info": info})
    except Exception as exc:  # noqa: BLE001 - surface a readable error to the UI
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400


@app.route("/api/search", methods=["POST"])
def api_search():
    if _index is None:
        return jsonify({"error": "No index built yet. Upload a spreadsheet first."}), 400

    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k", 5))
    threshold = float(payload.get("threshold", 0.0))

    if not query:
        return jsonify({"error": "Query is empty."}), 400

    results, tokens = se.search(query, _index, top_k=top_k, threshold=threshold)

    if not results and not tokens:
        return jsonify({"results": [], "tokens": [], "message": "No recognizable words in query."})

    return jsonify({"results": results, "tokens": tokens})


if __name__ == "__main__":
    # debug=True gives auto-reload while you edit templates/static files.
    app.run(debug=True, host="127.0.0.1", port=5000)
