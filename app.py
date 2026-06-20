"""
app.py
------
Flask backend for ProText Memory Quiz.

Routes:
  GET    /              — serve the app
  GET    /api/config    — check whether a session password is required
  POST   /api/unlock    — submit the session password
  POST   /api/login     — start a named session
  POST   /api/logout    — clear session
  GET    /api/profile   — return current user profile (history + favorites)
  POST   /api/extract   — extract scored chunks from raw text
  POST   /api/score     — diff answer vs target, return annotated result
  POST   /api/history   — save/update a completed attempt
  DELETE /api/history   — clear history (body: {"mode": "last10"|"all"})
  POST   /api/favorites — toggle a chunk as a favorite
"""

import json
import random
import re
import string
import uuid
from difflib import SequenceMatcher
from pathlib import Path

import nltk

for _pkg, _kind in [("punkt", "tokenizers"), ("punkt_tab", "tokenizers"), ("stopwords", "corpora")]:
    try:
        nltk.data.find(f"{_kind}/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from flask import Flask, jsonify, request, send_from_directory, session

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "protext-memory-quiz-local-secret"

STOP_WORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Password gate — None means disabled (local mode); set via set_password()
# ---------------------------------------------------------------------------

_HOST_PASSWORD: str | None = None


def set_password(pw: str | None) -> None:
    global _HOST_PASSWORD
    _HOST_PASSWORD = pw


# ---------------------------------------------------------------------------
# Difficulty presets
#
# target_words is an approximate "notable word" budget mapping to human
# working-memory thresholds (stop words excluded from the count).
# ---------------------------------------------------------------------------

DIFFICULTY = {
    "easy":       {"max_sentences": 2,    "target_words": 8},
    "medium":     {"max_sentences": 4,    "target_words": 16},
    "hard":       {"max_sentences": 8,    "target_words": 32},
    "extra_hard": {"max_sentences": 12,   "target_words": 64},
    "recite":     {"max_sentences": 9999, "target_words": 9999},
}


# ---------------------------------------------------------------------------
# Profile helpers — one JSON file per user in data/
# ---------------------------------------------------------------------------

def _safe_name(username: str) -> str:
    return re.sub(r"[^\w\-]", "_", username.strip().lower())


def profile_path(username: str) -> Path:
    return DATA_DIR / f"{_safe_name(username)}.json"


def load_profile(username: str) -> dict:
    p = profile_path(username)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        data.setdefault("favorites", [])   # backfill older profiles
        return data
    return {"username": username, "history": [], "favorites": []}


def save_profile(data: dict) -> None:
    with open(profile_path(data["username"]), "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Text scoring — heuristic jargon-density score per sentence
# ---------------------------------------------------------------------------

def _notable_count(sentence: str) -> int:
    """Count words that are not common stop words and have length > 2."""
    return sum(
        1 for w in sentence.split()
        if w.lower().strip(string.punctuation) not in STOP_WORDS and len(w) > 2
    )


def score_sentence(sentence: str) -> float:
    """Returns a 0-1 jargon-density score for a sentence."""
    words = sentence.split()
    if not words:
        return 0.0

    notable_ratio = _notable_count(sentence) / len(words)

    jargon_count = sum(
        1 for w in words if (
            re.search(r"[A-Z][a-z]+[A-Z]", w)   # CamelCase / PascalCase
            or re.fullmatch(r"[A-Z]{2,}", w)      # acronym (TCP, API …)
            or "-" in w                            # hyphenated term
            or re.search(r"\d", w)                 # contains digit
            or re.search(r"[%/\\()\[\]{}]", w)   # technical punctuation
        )
    )

    length_score = min(len(words) / 20, 1.0)   # peaks at ~20-word sentences
    jargon_score = min(jargon_count / 3, 1.0)

    return (notable_ratio * 0.40) + (length_score * 0.35) + (jargon_score * 0.25)


# ---------------------------------------------------------------------------
# Chunk extraction — groups sentences into word-budget chunks
# ---------------------------------------------------------------------------

def extract_chunks(text: str, difficulty: str, shuffle: bool) -> list[dict]:
    """
    Tokenizes text into sentences and greedily groups them into chunks
    that fit within the notable-word budget for the given difficulty.

    Returns a list of chunk dicts: {id, text, sentence_indices, score}
    """
    cfg       = DIFFICULTY.get(difficulty, DIFFICULTY["easy"])
    sentences = sent_tokenize(text.strip())
    if not sentences:
        return []

    if difficulty == "recite":
        return [{
            "id":               str(uuid.uuid4()),
            "text":             text.strip(),
            "sentence_indices": list(range(len(sentences))),
            "score":            1.0,
        }]

    scored = [
        {"index": i, "text": s, "score": score_sentence(s), "notable": _notable_count(s)}
        for i, s in enumerate(sentences)
    ]

    max_s    = cfg["max_sentences"]
    target_w = cfg["target_words"]
    chunks, i = [], 0

    while i < len(scored):
        bucket, word_budget = [], 0
        j = i
        while j < len(scored) and len(bucket) < max_s:
            s = scored[j]
            word_budget += s["notable"]
            bucket.append(s)
            j += 1
            if word_budget >= target_w:
                break

        if bucket:
            chunks.append({
                "id":               str(uuid.uuid4()),
                "text":             " ".join(b["text"] for b in bucket),
                "sentence_indices": [b["index"] for b in bucket],
                "score":            sum(b["score"] for b in bucket) / len(bucket),
            })
        i = j

    if shuffle:
        random.shuffle(chunks)

    return chunks


# ---------------------------------------------------------------------------
# Answer scoring — word-level diff between answer and target
# ---------------------------------------------------------------------------

def _split_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _normalize_lenient(text: str) -> str:
    """Lowercase, expand hyphens to spaces, strip remaining punctuation."""
    return text.lower().replace("-", " ").translate(str.maketrans("", "", string.punctuation))


def score_answer(target: str, answer: str, lenient: bool = False) -> dict:
    """
    Word-level SequenceMatcher diff between answer and target.

    lenient=True: lowercases, strips punctuation, and expands hyphenated
    words before comparing, so capitalization and punctuation are forgiven.

    Returns:
      accuracy         — float 0-100 (matching words / target words)
      annotated_target — list of {text, correct}
      annotated_answer — list of {text, correct}
    """
    if lenient:
        t_tokens = _split_tokens(_normalize_lenient(target))
        a_tokens = _split_tokens(_normalize_lenient(answer))
    else:
        t_tokens = _split_tokens(target)
        a_tokens = _split_tokens(answer)

    matcher = SequenceMatcher(None, a_tokens, t_tokens, autojunk=False)
    opcodes = matcher.get_opcodes()

    annotated_target, annotated_answer = [], []
    for tag, i1, i2, j1, j2 in opcodes:
        correct = (tag == "equal")
        for tok in t_tokens[j1:j2]:
            annotated_target.append({"text": tok, "correct": correct})
        for tok in a_tokens[i1:i2]:
            annotated_answer.append({"text": tok, "correct": correct})

    matching = sum(j2 - j1 for tag, i1, i2, j1, j2 in opcodes if tag == "equal")
    accuracy = round(100 * matching / max(len(t_tokens), 1), 1)

    return {
        "accuracy":         accuracy,
        "annotated_target": annotated_target,
        "annotated_answer": annotated_answer,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory("templates", "index.html")


@app.get("/api/config")
def get_config():
    needs_unlock = bool(_HOST_PASSWORD) and not session.get("unlocked")
    return jsonify({"password_required": needs_unlock})


@app.post("/api/unlock")
def unlock():
    data = request.get_json(force=True)
    if data.get("password") == _HOST_PASSWORD:
        session["unlocked"] = True
        return jsonify({"ok": True})
    return jsonify({"error": "Incorrect password."}), 403


@app.post("/api/login")
def login():
    if _HOST_PASSWORD and not session.get("unlocked"):
        return jsonify({"error": "Not unlocked."}), 403

    data     = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    if not username:
        return jsonify({"error": "Username cannot be empty."}), 400

    session["username"] = username
    profile = load_profile(username)
    save_profile(profile)
    return jsonify({
        "username":  username,
        "history":   profile["history"],
        "favorites": profile["favorites"],
    })


@app.post("/api/logout")
def logout():
    was_unlocked = session.get("unlocked", False)
    session.clear()
    if was_unlocked:
        session["unlocked"] = True   # keep unlock across username switches
    return jsonify({"ok": True})


@app.get("/api/profile")
def get_profile():
    username = session.get("username")
    if not username:
        return jsonify({"error": "Not logged in."}), 401
    return jsonify(load_profile(username))


@app.post("/api/extract")
def extract():
    if not session.get("username"):
        return jsonify({"error": "Not logged in."}), 401

    data       = request.get_json(force=True)
    text       = (data.get("text") or "").strip()
    difficulty = data.get("difficulty", "easy")
    shuffle    = bool(data.get("shuffle", True))

    if not text:
        return jsonify({"error": "No text supplied."}), 400
    if difficulty not in DIFFICULTY:
        return jsonify({"error": f"Unknown difficulty '{difficulty}'."}), 400

    chunks = extract_chunks(text, difficulty, shuffle)
    return jsonify({"chunks": chunks, "total": len(chunks)})


@app.post("/api/score")
def score():
    if not session.get("username"):
        return jsonify({"error": "Not logged in."}), 401

    data    = request.get_json(force=True)
    target  = (data.get("target") or "").strip()
    answer  = (data.get("answer") or "").strip()
    lenient = bool(data.get("lenient", False))

    if not target:
        return jsonify({"error": "No target text provided."}), 400

    return jsonify(score_answer(target, answer, lenient=lenient))


@app.post("/api/history")
def add_history():
    username = session.get("username")
    if not username:
        return jsonify({"error": "Not logged in."}), 401

    data    = request.get_json(force=True)
    profile = load_profile(username)

    # Keep only the best score per unique chunk
    existing = next(
        (h for h in profile["history"] if h.get("chunk_text") == data.get("chunk_text")),
        None,
    )
    if existing:
        if data.get("accuracy", 0) > existing["accuracy"]:
            existing["accuracy"]  = data["accuracy"]
            existing["timestamp"] = data.get("timestamp", "")
    else:
        profile["history"].insert(0, {   # newest first
            "id":         str(uuid.uuid4()),
            "chunk_text": data.get("chunk_text", ""),
            "difficulty": data.get("difficulty", "easy"),
            "accuracy":   data.get("accuracy", 0),
            "timestamp":  data.get("timestamp", ""),
        })

    save_profile(profile)
    return jsonify({"ok": True, "history": profile["history"]})


@app.delete("/api/history")
def clear_history():
    username = session.get("username")
    if not username:
        return jsonify({"error": "Not logged in."}), 401

    data    = request.get_json(force=True)
    mode    = data.get("mode", "all")
    profile = load_profile(username)

    # "last10" drops the 10 most-recent entries; history is stored newest-first
    profile["history"] = profile["history"][10:] if mode == "last10" else []

    save_profile(profile)
    return jsonify({"ok": True, "history": profile["history"]})


@app.post("/api/favorites")
def toggle_favorite():
    username = session.get("username")
    if not username:
        return jsonify({"error": "Not logged in."}), 401

    data       = request.get_json(force=True)
    chunk_text = (data.get("chunk_text") or "").strip()
    if not chunk_text:
        return jsonify({"error": "chunk_text cannot be empty."}), 400

    profile   = load_profile(username)
    favorites = profile.setdefault("favorites", [])

    if chunk_text in favorites:
        favorites.remove(chunk_text)
        is_favorite = False
    else:
        favorites.append(chunk_text)
        is_favorite = True

    save_profile(profile)
    return jsonify({"is_favorite": is_favorite, "favorites": favorites})
