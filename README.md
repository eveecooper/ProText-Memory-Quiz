# ProText Memory Quiz

A local memory-training app for technical text. Paste dense documentation, research papers, or spec sheets — the app extracts the most information-dense chunks, quizzes you on recalling them, and scores your answer word by word.

Everything runs locally. No account, no internet connection required.

---

## Quick Start

```bash
# 1. Install dependencies (one time)
pip install -r requirements.txt

# 2. Run the app
python run.py
```

The browser opens automatically at `http://127.0.0.1:5000`.

---

## Command-Line Options

```
python run.py [--host] [--port PORT] [--no-browser] [--debug]

  --host        Share the app with other devices on your local network
  --port PORT   Port to bind to (default: 5000)
  --no-browser  Don't auto-open the browser on startup
  --debug       Enable Flask debug/hot-reload mode (dev only)
```

When running with `--host`, a password is required to log in. Set the password by creating a file called `vars.py` in the project folder and assigning a `login_passcode` variable to the password of your choice as shown below.

## vars.py 

```
login_passcode = "passwording"
```

---

## How to Use

1. **Enter your name** on the login screen.
2. **Paste any text** — documentation, a paper, a spec — into the text box.
3. **Choose a difficulty** and hit **Start Quiz**.
4. The app shows you a chunk of text. Read it, then try to recall it word for word.
5. Submit your answer and see which words you got right (green) or missed (red).
6. Your best score for each chunk is saved automatically.

---

## Difficulty Levels

| Level      | Chunk size (approx.)     |
|------------|--------------------------|
| Easy       | ~1–2 sentences           |
| Medium     | ~2–4 sentences           |
| Hard       | ~4–8 sentences           |
| Extra Hard | ~8–12 sentences          |
| Recite     | The entire text at once  |

---

## Scoring

Your answer is compared to the target word by word. The accuracy percentage is:

    matching words / total words in target × 100

**Lenient mode** (toggle in the app) forgives capitalization differences, punctuation, and hyphenated words split across spaces — so `self attention` scores the same as `self-attention`.

---

## Your History

- Your best score per chunk is saved to your profile.
- Subsequent attempts only update the record if your new score is higher.
- You can clear your 10 most recent entries or all history from the sidebar.
- Chunks can be saved to **Favorites** for quick access to your practice material.

---


