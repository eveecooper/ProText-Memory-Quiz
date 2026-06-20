"""
tests.py
--------
Unit + integration tests for ProText Memory Quiz.

Run from anywhere inside the project:
    python tests.py

Verbose output:
    python tests.py -v

No pytest required — uses stdlib unittest only.
"""

import os
import sys
import unittest
import uuid

from vars import login_passcode

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from app import app, extract_chunks, score_answer, score_sentence, load_profile, save_profile
import app as app_module


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TECHNICAL_TEXT = """
The transformer architecture introduced in 'Attention Is All You Need' relies entirely
on self-attention mechanisms to draw global dependencies between input and output,
discarding recurrence and convolutions entirely. Multi-head attention allows the model
to jointly attend to information from different representation subspaces at different
positions. Positional encodings are injected into the embeddings because the architecture
contains no recurrence or convolution. The feed-forward sub-layer applies two linear
transformations with a ReLU activation in between, operating identically on each position.
Layer normalization is applied before each sub-layer via a residual connection,
which stabilizes training of very deep networks.
"""

SIMPLE_TEXT = "The cat sat on the mat. The dog ran fast."


# ===========================================================================
# 1. Unit tests — pure functions, no Flask server needed
# ===========================================================================

class TestSentenceScorer(unittest.TestCase):
    """score_sentence() should rank technical sentences higher than simple ones."""

    def test_technical_scores_higher_than_simple(self):
        technical = "Multi-head attention allows the model to attend to subspaces at different positions."
        simple    = "The cat sat on the mat."
        self.assertGreater(score_sentence(technical), score_sentence(simple))

    def test_empty_string_returns_zero(self):
        self.assertEqual(score_sentence(""), 0.0)

    def test_score_is_between_zero_and_one(self):
        for sentence in [
            "Hello.",
            "ReLU activation functions introduce non-linearity into neural network architectures.",
            "TCP/IP stack implements OSI layers 3 and 4 via IPv4/IPv6 and TCP/UDP respectively.",
        ]:
            score = score_sentence(sentence)
            self.assertGreaterEqual(score, 0.0, f"Score below 0 for: {sentence}")
            self.assertLessEqual(score,   1.0, f"Score above 1 for: {sentence}")

    def test_camelcase_boosts_score(self):
        plain = "the function returns a value"
        camel = "the sendHttpRequest function returns a ResponseObject value"
        self.assertGreater(score_sentence(camel), score_sentence(plain))

    def test_whitespace_only_returns_zero(self):
        self.assertEqual(score_sentence("   "), 0.0)

    def test_acronym_boosts_score(self):
        plain  = "the protocol transfers data"
        acronym = "the TCP protocol transfers data via UDP"
        self.assertGreater(score_sentence(acronym), score_sentence(plain))


class TestChunkExtraction(unittest.TestCase):
    """extract_chunks() should produce correctly-sized, non-empty chunks."""

    def test_returns_list(self):
        self.assertIsInstance(extract_chunks(TECHNICAL_TEXT, "easy", False), list)

    def test_non_empty_for_valid_text(self):
        self.assertGreater(len(extract_chunks(TECHNICAL_TEXT, "easy", False)), 0)

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(extract_chunks("", "easy", False), [])

    def test_whitespace_only_returns_empty_list(self):
        self.assertEqual(extract_chunks("   \n\t  ", "medium", False), [])

    def test_each_chunk_has_required_keys(self):
        for chunk in extract_chunks(TECHNICAL_TEXT, "medium", False):
            self.assertIn("id",               chunk)
            self.assertIn("text",             chunk)
            self.assertIn("sentence_indices", chunk)
            self.assertIn("score",            chunk)

    def test_chunk_text_is_nonempty_string(self):
        for chunk in extract_chunks(TECHNICAL_TEXT, "easy", False):
            self.assertIsInstance(chunk["text"], str)
            self.assertGreater(len(chunk["text"].strip()), 0)

    def test_chunk_ids_are_unique(self):
        chunks = extract_chunks(TECHNICAL_TEXT, "easy", False)
        ids = [c["id"] for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_chunk_score_between_zero_and_one(self):
        for chunk in extract_chunks(TECHNICAL_TEXT, "medium", False):
            self.assertGreaterEqual(chunk["score"], 0.0)
            self.assertLessEqual(chunk["score"],   1.0)

    def test_recite_mode_returns_single_chunk(self):
        chunks = extract_chunks(TECHNICAL_TEXT, "recite", False)
        self.assertEqual(len(chunks), 1)
        self.assertIn("self-attention", chunks[0]["text"])

    def test_all_difficulties_produce_chunks(self):
        for difficulty in ("easy", "medium", "hard", "extra_hard"):
            with self.subTest(difficulty=difficulty):
                chunks = extract_chunks(TECHNICAL_TEXT, difficulty, False)
                self.assertGreater(len(chunks), 0)

    def test_easy_chunks_shorter_than_hard(self):
        easy = extract_chunks(TECHNICAL_TEXT, "easy", False)
        hard = extract_chunks(TECHNICAL_TEXT, "hard", False)
        avg_easy = sum(len(c["text"].split()) for c in easy) / len(easy)
        avg_hard = sum(len(c["text"].split()) for c in hard) / len(hard)
        self.assertLess(avg_easy, avg_hard)

    def test_single_sentence_text(self):
        chunks = extract_chunks(
            "Self-attention draws global dependencies between input and output.", "easy", False
        )
        self.assertEqual(len(chunks), 1)

    def test_shuffle_produces_same_chunks_different_order(self):
        ordered  = extract_chunks(TECHNICAL_TEXT, "easy", False)
        shuffled = extract_chunks(TECHNICAL_TEXT, "easy", True)
        self.assertEqual(sorted(c["text"] for c in ordered), sorted(c["text"] for c in shuffled))

    def test_unknown_difficulty_falls_back_to_easy(self):
        # Unknown difficulty should not crash — falls back to "easy" config
        chunks = extract_chunks(TECHNICAL_TEXT, "unknown_level", False)
        self.assertGreater(len(chunks), 0)


class TestScoreAnswer(unittest.TestCase):
    """score_answer() should return correct accuracy and annotated token lists."""

    def test_perfect_match_is_100(self):
        self.assertEqual(score_answer("hello world", "hello world")["accuracy"], 100.0)

    def test_complete_mismatch_is_zero(self):
        self.assertEqual(score_answer("hello world", "foo bar")["accuracy"], 0.0)

    def test_partial_match(self):
        result = score_answer("the quick brown fox", "the quick brown cat")
        self.assertGreater(result["accuracy"], 0.0)
        self.assertLess(result["accuracy"], 100.0)

    def test_empty_answer_is_zero(self):
        self.assertEqual(score_answer("hello world", "")["accuracy"], 0.0)

    def test_result_has_required_keys(self):
        result = score_answer("hello world", "hello world")
        self.assertIn("accuracy",         result)
        self.assertIn("annotated_target", result)
        self.assertIn("annotated_answer", result)

    def test_annotated_tokens_have_correct_flag(self):
        result = score_answer("hello world", "hello world")
        for token in result["annotated_target"]:
            self.assertIn("text",    token)
            self.assertIn("correct", token)
            self.assertTrue(token["correct"])

    def test_wrong_word_flagged_incorrect(self):
        result = score_answer("hello world", "hello earth")
        incorrect = [t for t in result["annotated_answer"] if not t["correct"]]
        self.assertGreater(len(incorrect), 0)

    def test_missing_words_reduce_accuracy(self):
        full    = score_answer("one two three four five", "one two three four five")
        partial = score_answer("one two three four five", "one two")
        self.assertGreater(full["accuracy"], partial["accuracy"])

    def test_extra_words_in_answer_dont_inflate_score(self):
        result = score_answer("hello world", "hello world extra junk words here")
        self.assertLessEqual(result["accuracy"], 100.0)

    def test_accuracy_is_float(self):
        self.assertIsInstance(score_answer("hello world", "hello earth")["accuracy"], float)

    def test_accuracy_rounds_to_one_decimal(self):
        # 1 matching word out of 3 = 33.333... → should round to 33.3
        result = score_answer("one two three", "one foo bar")
        self.assertEqual(result["accuracy"], 33.3)

    # ---- Lenient mode ----

    def test_lenient_ignores_capitalization(self):
        result = score_answer("Hello World", "hello world", lenient=True)
        self.assertEqual(result["accuracy"], 100.0)

    def test_lenient_ignores_trailing_punctuation(self):
        result = score_answer("hello world.", "hello world", lenient=True)
        self.assertEqual(result["accuracy"], 100.0)

    def test_lenient_splits_hyphenated_words(self):
        # "self-attention" in lenient mode becomes two tokens: "self" + "attention"
        result = score_answer("self-attention mechanism", "self attention mechanism", lenient=True)
        self.assertEqual(result["accuracy"], 100.0)

    def test_lenient_strips_commas(self):
        result = score_answer("one, two, three", "one two three", lenient=True)
        self.assertEqual(result["accuracy"], 100.0)

    def test_lenient_combined_case_and_punctuation(self):
        result = score_answer("Hello, World!", "hello world", lenient=True)
        self.assertEqual(result["accuracy"], 100.0)

    def test_lenient_false_is_strict_by_default(self):
        result = score_answer("Hello World", "hello world", lenient=False)
        self.assertLess(result["accuracy"], 100.0)


# ===========================================================================
# 2. Integration tests — full HTTP round-trips via Flask test client
# ===========================================================================

class TestFlaskRoutes(unittest.TestCase):
    """Tests hit the actual Flask routes end-to-end with a test client."""

    def setUp(self):
        app.config["TESTING"]    = True
        app.config["SECRET_KEY"] = "test-secret"
        self.client = app.test_client()
        self.ctx    = app.test_request_context()
        self.ctx.push()
        app_module.set_password(login_passcode)

    def tearDown(self):
        self.ctx.pop()

    def _unlock(self):
        self.client.post("/api/unlock", json={"password": login_passcode})

    # ---- Config ----

    def test_config_password_required_when_not_unlocked(self):
        r = self.client.get("/api/config")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["password_required"])

    def test_config_no_password_required_when_no_password_set(self):
        app_module.set_password(None)
        try:
            r = self.client.get("/api/config")
            self.assertEqual(r.status_code, 200)
            self.assertFalse(r.get_json()["password_required"])
        finally:
            app_module.set_password(login_passcode)

    def test_config_no_password_required_after_unlock(self):
        self._unlock()
        r = self.client.get("/api/config")
        self.assertFalse(r.get_json()["password_required"])

    # ---- Unlock ----

    def test_unlock_correct_password(self):
        r = self.client.post("/api/unlock", json={"password": login_passcode})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])

    def test_unlock_wrong_password(self):
        r = self.client.post("/api/unlock", json={"password": "wrong"})
        self.assertEqual(r.status_code, 403)

    def test_unlock_empty_password(self):
        r = self.client.post("/api/unlock", json={"password": ""})
        self.assertEqual(r.status_code, 403)

    # ---- Login / logout ----

    def test_login_blocked_without_unlock(self):
        r = self.client.post("/api/login", json={"username": "testuser"})
        self.assertEqual(r.status_code, 403)

    def test_login_success(self):
        self._unlock()
        r = self.client.post("/api/login", json={"username": "testuser"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["username"], "testuser")
        self.assertIn("history",   data)
        self.assertIn("favorites", data)

    def test_login_empty_username_returns_400(self):
        self._unlock()
        r = self.client.post("/api/login", json={"username": ""})
        self.assertEqual(r.status_code, 400)

    def test_login_whitespace_username_returns_400(self):
        self._unlock()
        r = self.client.post("/api/login", json={"username": "   "})
        self.assertEqual(r.status_code, 400)

    def test_logout_clears_session(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        r = self.client.post("/api/logout")
        self.assertEqual(r.status_code, 200)
        r2 = self.client.get("/api/profile")
        self.assertEqual(r2.status_code, 401)

    def test_logout_preserves_unlock_state(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        self.client.post("/api/logout")
        # Should be able to log in again without re-unlocking
        r = self.client.post("/api/login", json={"username": "anotheruser"})
        self.assertEqual(r.status_code, 200)

    # ---- Profile ----

    def test_profile_requires_login(self):
        r = self.client.get("/api/profile")
        self.assertEqual(r.status_code, 401)

    def test_profile_returns_after_login(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        r = self.client.get("/api/profile")
        self.assertEqual(r.status_code, 200)
        self.assertIn("username", r.get_json())

    def test_profile_includes_favorites_field(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        r = self.client.get("/api/profile")
        data = r.get_json()
        self.assertIn("favorites", data)
        self.assertIsInstance(data["favorites"], list)

    # ---- Extract ----

    def test_extract_requires_login(self):
        r = self.client.post("/api/extract", json={"text": "hello", "difficulty": "easy"})
        self.assertEqual(r.status_code, 401)

    def test_extract_returns_chunks(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        r = self.client.post("/api/extract", json={
            "text":       TECHNICAL_TEXT,
            "difficulty": "easy",
            "shuffle":    False,
        })
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("chunks", data)
        self.assertGreater(data["total"], 0)

    def test_extract_empty_text_returns_400(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        r = self.client.post("/api/extract", json={"text": "", "difficulty": "easy"})
        self.assertEqual(r.status_code, 400)

    def test_extract_invalid_difficulty_returns_400(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        r = self.client.post("/api/extract", json={"text": TECHNICAL_TEXT, "difficulty": "godmode"})
        self.assertEqual(r.status_code, 400)

    # ---- Score ----

    def test_score_requires_login(self):
        r = self.client.post("/api/score", json={"target": "hello", "answer": "hello"})
        self.assertEqual(r.status_code, 401)

    def test_score_perfect_answer(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        r = self.client.post("/api/score", json={"target": "hello world", "answer": "hello world"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["accuracy"], 100.0)

    def test_score_empty_answer_returns_zero(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        r = self.client.post("/api/score", json={"target": "hello world", "answer": ""})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["accuracy"], 0.0)

    def test_score_lenient_mode_via_route(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        r = self.client.post("/api/score", json={
            "target":  "Self-attention, sometimes called intra-attention.",
            "answer":  "self attention sometimes called intra attention",
            "lenient": True,
        })
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["accuracy"], 100.0)

    def test_score_empty_target_returns_400(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        r = self.client.post("/api/score", json={"target": "", "answer": "something"})
        self.assertEqual(r.status_code, 400)

    # ---- History ----

    def test_history_requires_login(self):
        r = self.client.post("/api/history", json={"chunk_text": "x", "accuracy": 0})
        self.assertEqual(r.status_code, 401)

    def test_history_save_and_retrieve(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        r = self.client.post("/api/history", json={
            "chunk_text": "test chunk",
            "difficulty": "easy",
            "accuracy":   75.0,
            "timestamp":  "2025-01-01",
        })
        self.assertEqual(r.status_code, 200)
        history = r.get_json()["history"]
        self.assertTrue(any(h["chunk_text"] == "test chunk" for h in history))

    def test_history_best_score_only_updates_on_improvement(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        payload = {"chunk_text": "unique chunk abc", "difficulty": "easy", "timestamp": "2025-01-01"}

        self.client.post("/api/history", json={**payload, "accuracy": 60.0})
        self.client.post("/api/history", json={**payload, "accuracy": 40.0})  # worse — no update
        r = self.client.post("/api/history", json={**payload, "accuracy": 80.0})  # better — updates

        entry = next(h for h in r.get_json()["history"] if h["chunk_text"] == "unique chunk abc")
        self.assertEqual(entry["accuracy"], 80.0)

    def test_history_clear_all(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        self.client.post("/api/history", json={
            "chunk_text": "to be deleted", "difficulty": "easy", "accuracy": 50.0, "timestamp": ""
        })
        r = self.client.delete("/api/history", json={"mode": "all"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["history"], [])

    def test_history_clear_last10_drops_newest(self):
        user = f"last10_{uuid.uuid4().hex[:8]}"
        self._unlock()
        self.client.post("/api/login", json={"username": user})

        for i in range(15):
            self.client.post("/api/history", json={
                "chunk_text": f"l10_chunk_{i}",
                "difficulty": "easy",
                "accuracy":   float(i),
                "timestamp":  "",
            })

        r = self.client.delete("/api/history", json={"mode": "last10"})
        self.assertEqual(r.status_code, 200)
        history = r.get_json()["history"]
        # 15 entries minus the 10 newest = 5 oldest remain
        self.assertEqual(len(history), 5)
        remaining = {h["chunk_text"] for h in history}
        for i in range(5):   # chunks 0-4 (oldest) should survive
            self.assertIn(f"l10_chunk_{i}", remaining)

    # ---- Favorites ----

    def test_favorite_requires_login(self):
        r = self.client.post("/api/favorites", json={"chunk_text": "hello"})
        self.assertEqual(r.status_code, 401)

    def test_favorite_empty_chunk_returns_400(self):
        self._unlock()
        self.client.post("/api/login", json={"username": "testuser"})
        r = self.client.post("/api/favorites", json={"chunk_text": ""})
        self.assertEqual(r.status_code, 400)

    def test_favorite_toggle_adds_chunk(self):
        user = f"favtest_{uuid.uuid4().hex[:8]}"
        self._unlock()
        self.client.post("/api/login", json={"username": user})
        r = self.client.post("/api/favorites", json={"chunk_text": "unique add chunk aaa"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue(data["is_favorite"])
        self.assertIn("unique add chunk aaa", data["favorites"])

    def test_favorite_toggle_removes_on_second_call(self):
        user = f"favtoggle_{uuid.uuid4().hex[:8]}"
        self._unlock()
        self.client.post("/api/login", json={"username": user})
        chunk = "unique toggle test chunk xyz"
        self.client.post("/api/favorites", json={"chunk_text": chunk})   # add
        r = self.client.post("/api/favorites", json={"chunk_text": chunk})  # remove
        data = r.get_json()
        self.assertFalse(data["is_favorite"])
        self.assertNotIn(chunk, data["favorites"])

    def test_login_returns_favorites(self):
        user = f"favlogin_{uuid.uuid4().hex[:8]}"
        self._unlock()
        self.client.post("/api/login", json={"username": user})
        self.client.post("/api/favorites", json={"chunk_text": "unique login fav chunk bbb"})
        self.client.post("/api/logout")
        self._unlock()
        r = self.client.post("/api/login", json={"username": user})
        data = r.get_json()
        self.assertIn("favorites", data)
        self.assertIn("unique login fav chunk bbb", data["favorites"])

    def test_multiple_favorites_accumulate(self):
        user = f"favmulti_{uuid.uuid4().hex[:8]}"
        self._unlock()
        self.client.post("/api/login", json={"username": user})
        self.client.post("/api/favorites", json={"chunk_text": "chunk alpha"})
        r = self.client.post("/api/favorites", json={"chunk_text": "chunk beta"})
        data = r.get_json()
        self.assertIn("chunk alpha", data["favorites"])
        self.assertIn("chunk beta",  data["favorites"])

    # ---- Index route ----

    def test_index_returns_html(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"ProText Memory Quiz", r.data)
        self.assertIn(b"Favorites", r.data)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
