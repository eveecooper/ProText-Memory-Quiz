# ProText Memory Quiz

A local memory-training app for technical text. Paste dense documentation, research papers,
or spec sheets — the app extracts high-jargon chunks, runs you through a recall test,
and scores your answer with word-level green/red diff highlighting.

---

## Quick Start

```bash
# 1. Install dependencies (one time)
pip install flask nltk

# 2. Run the app
cd protext_memory_quiz
python run.py
```

The browser opens automatically at `http://127.0.0.1:5000`.

---

## Directory Layout

```
protext_memory_quiz/
├── run.py          ← console launcher (start here)
├── app.py          ← Flask backend + NLP logic
├── templates/
│   └── index.html  ← single-page frontend (HTML + vanilla JS)
├── static/         ← reserved for future assets
└── data/           ← per-user JSON profiles (auto-created on first login)
```

---

## Command-Line Options

```
python run.py [--port PORT] [--no-browser] [--debug]

Options:
  --port PORT      Port to bind to (default: 5000)
  --no-browser     Don't auto-open the browser on startup
  --debug          Enable Flask debug/hot-reload mode (dev only)
```

---

## Difficulty Levels

| Level       | Max sentences | Notable-word budget |
|-------------|---------------|---------------------|
| Easy        | 2             | ~16 words           |
| Medium      | 4             | ~32 words           |
| Hard        | 8             | ~64 words           |
| Extra Hard  | 12            | ~96 words           |
| Recite      | entire doc    | unlimited           |

"Notable words" excludes common stop words (a, the, of, for …).
The budgets map to rough human working-memory thresholds.

---

## How Chunk Extraction Works

No API calls are made. Everything runs locally using NLTK.

1. The text is tokenized into sentences with NLTK's `sent_tokenize`.
2. Each sentence receives a jargon-density score (0–1) based on:
   - Ratio of non-stop-words
   - Sentence length (sweet spot ~20 words)
   - Presence of CamelCase, acronyms, digits, hyphens, technical punctuation
3. Sentences are grouped greedily in document order until the word budget
   for the chosen difficulty is reached.
4. If "Shuffle" is enabled, chunk order is randomized; otherwise document order is preserved.

---

## Scoring

Answers are compared against the target text using Python's `difflib.SequenceMatcher`
at the word level. The accuracy percentage is:

    matching words / total words in target × 100

Each word in both the answer and the target is highlighted green (correct) or
red (incorrect/missing) in the result view.

---

## User Profiles

- Profiles are plain JSON files stored in `data/<username>.json`.
- Login is name-only — no password, no authentication. Local use only.
- History tracks best score per unique chunk. Subsequent attempts only
  update the record if the new score is higher.
- You can clear the last 10 entries or all history from the sidebar.

---

## Dependencies

| Package | Purpose                                |
|---------|----------------------------------------|
| Flask   | HTTP server and routing                |
| NLTK    | Sentence tokenization + stop words     |

NLTK data (`punkt`, `punkt_tab`, `stopwords`) is downloaded automatically
on first run if not already present.

---

## Extending the App

The codebase is intentionally minimal to stay easy to read and modify.

**Swap in a local LLM for chunk selection:**
Replace `extract_chunks()` in `app.py` with a call to Ollama, llama.cpp,
or any locally-served model. The route contract (`POST /api/extract` returning
`{chunks: [...]}`) doesn't change, so the frontend needs no edits.

**Add more difficulty levels:**
Edit the `DIFFICULTY` dict near the top of `app.py`.

**Persist session across browser restarts:**
Change `app.secret_key` to a fixed value and configure Flask-Session
with a file or Redis backend.
