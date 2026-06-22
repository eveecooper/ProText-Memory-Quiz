# ProText Memory Quiz — Developer Notes

## File Layout

```
protext_memory_quiz/
├── run.py          ← launcher (CLI entry point)
├── app.py          ← Flask backend + NLP logic
├── vars.py         ← password config
├── tests.py        ← unittest suite (no pytest required)
├── templates/
│   └── index.html  ← single-page frontend (HTML + vanilla JS)
├── static/         ← static assets (favicon, etc.)
└── data/           ← per-user JSON profiles (auto-created on first login)
```

---

## Architecture

`run.py` is the entry point. It parses CLI args, optionally enables the password gate via `set_password()`, and starts the Flask dev server.

`app.py` contains everything else: NLP logic, user profile I/O, and all API routes. The frontend (`index.html`) is a vanilla JS single-page app that talks to the Flask backend via fetch.

Request flow:
```
browser ──fetch──▶ Flask route (app.py)
                        │
                 ┌──────┴──────┐
           NLP logic        Profile I/O
        (score/extract)   (data/*.json)
```

---

## API Routes

| Method | Path              | Description                                     |
|--------|-------------------|-------------------------------------------------|
| GET    | `/`               | Serve `index.html`                              |
| GET    | `/api/config`     | Returns `{password_required: bool}`             |
| POST   | `/api/unlock`     | Submit session password `{password}`            |
| POST   | `/api/login`      | Login with `{username}`, returns profile        |
| POST   | `/api/logout`     | Clear session (unlock state preserved)          |
| GET    | `/api/profile`    | Return current user profile                     |
| POST   | `/api/extract`    | Extract chunks `{text, difficulty, shuffle}`    |
| POST   | `/api/score`      | Score answer `{target, answer, lenient}`        |
| POST   | `/api/history`    | Save attempt `{chunk_text, difficulty, accuracy, timestamp}` |
| DELETE | `/api/history`    | Clear history `{mode: "last10"\|"all"}`          |
| POST   | `/api/favorites`  | Toggle favorite `{chunk_text}`                  |

---

## How Chunk Extraction Works

1. NLTK's `sent_tokenize` splits the text into sentences.
2. Each sentence receives a jargon-density score (0–1) via `score_sentence()`:
   - **notable_ratio** (40%): fraction of non-stop-words with length > 2
   - **length_score** (35%): peaks at ~20-word sentences (`min(words/20, 1.0)`)
   - **jargon_score** (25%): CamelCase, acronyms, digits, hyphens, technical punctuation
3. Sentences are grouped greedily in document order until the notable-word budget for the chosen difficulty is reached.
4. If `shuffle=True`, chunk order is randomized.

**Difficulty presets** (edit `DIFFICULTY` dict in `app.py` to add levels):

| Level      | max_sentences | target_words (notable) |
|------------|---------------|------------------------|
| easy       | 2             | 8                      |
| medium     | 4             | 16                     |
| hard       | 8             | 32                     |
| extra_hard | 12            | 64                     |
| recite     | 9999          | 9999 (whole doc)       |

---

## How Scoring Works

`score_answer(target, answer, lenient=False)` uses `difflib.SequenceMatcher` at the word (token) level:

- `accuracy = round(100 * matching_words / max(len(target_tokens), 1), 1)`
- Returns `annotated_target` and `annotated_answer`: lists of `{text, correct}` for the frontend to highlight.

**Lenient mode** (`_normalize_lenient`): lowercases, replaces hyphens with spaces, then strips all remaining punctuation before tokenizing. This means `"Self-attention,"` and `"self attention"` score as identical.

---

## User Profiles

Profiles are plain JSON files at `data/<sanitized_username>.json`:

```json
{
  "username": "alice",
  "history": [
    {
      "id": "...",
      "chunk_text": "...",
      "difficulty": "medium",
      "accuracy": 87.5,
      "timestamp": "2025-06-01"
    }
  ],
  "favorites": ["chunk text here", "another saved chunk"]
}
```

- History is stored newest-first.
- Only the best score per unique `chunk_text` is kept.
- Old profiles without a `favorites` key are backfilled to `[]` on load.
- Login is username-only — no authentication. Intended for local/trusted-network use.

---

## Dependencies

| Package | Purpose                                |
|---------|----------------------------------------|
| Flask   | HTTP server and routing                |
| NLTK    | Sentence tokenization + stop words     |

NLTK data (`punkt`, `punkt_tab`, `stopwords`) is downloaded automatically on first run if not already present.

---

## Extending the App

**Swap in a local LLM for chunk selection:**
Replace `extract_chunks()` in `app.py` with a call to Ollama, llama.cpp, or any local model. The route contract (`POST /api/extract` → `{chunks: [...], total: N}`) is unchanged, so the frontend needs no edits.

**Add a difficulty level:**
Add an entry to the `DIFFICULTY` dict and add the corresponding `<option>` in `index.html`.

**Persist sessions across browser restarts:**
Change `app.secret_key` to a fixed value and configure Flask-Session with a file or Redis backend.

---

## Running Tests

```bash
python tests.py        # all tests
python tests.py -v     # verbose
```

The suite is pure `unittest` — no pytest or extra dependencies. Tests cover:
- `score_sentence()` — jargon scoring heuristics
- `extract_chunks()` — chunking correctness, all difficulty levels, shuffle
- `score_answer()` — accuracy math, annotated output, lenient mode
- All Flask routes — auth flow, CRUD, edge cases