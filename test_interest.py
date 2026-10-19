"""
Ad hoc check for core/interest.py (mocked Ollama and vote endpoint, same
approach as test_v213_fixes.py). Run: venv/bin/python test_interest.py
"""
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core.interest as interest
from core.deduplication import _pack_embedding, connect_history_db


def test_filter_by_interest():
    answers = {"sport": "SCARTA.", "politica": "TIENI", "ollama down": None, "garbled": "forse"}
    interest._call_ollama_chat = lambda prompt, **kw: next(a for t, a in answers.items() if f"Titolo: {t}\n" in prompt)
    kept = interest.filter_by_interest([{"title": t, "source": "s", "category": "c"} for t in answers])
    assert [a["title"] for a in kept] == ["politica", "ollama down", "garbled"], kept


def test_assign_sections():
    answers = {"a": "Italy", "b": "Europe", "c": '"Technology and AI".', "d": "Sports", "e": None, "f": "Italy"}
    # second look, "where does it happen?" (prompts/section_country.txt)
    countries = {"a": "Israele", "b": "Germany.", "d": "INTERNAZIONALE", "e": None, "f": "Italia"}

    def fake(prompt, **kw):
        title = next(t for t in answers if f"Titolo: {t}\n" in prompt)
        return countries[title] if prompt.rstrip().endswith("Paese:") else answers[title]

    interest._call_ollama_chat = fake
    arts = [{"title": t, "source": "s", "category": "Tech & AI"} for t in answers]
    got = [a["category"] for a in interest.assign_sections(arts)]
    # Oct 19, 2026: an Israeli election filed under Italy is moved by the country answer;
    # short "Europe" is expanded; off-list and failed answers fall back from the source bucket
    assert got == [
        "Rest of the world", "European Union / Europe", "Technology and AI", "Technology and AI",
        "Technology and AI", "Italy",
    ], got
    assert arts[0]["category"] == "Tech & AI"  # input not mutated


def _vec(liked: bool, rng) -> list[float]:
    # Two well-separated blobs in 16 dimensions.
    return [(1.0 if liked else -1.0) * (i < 8) + rng.gauss(0, 0.3) for i in range(16)]


def test_votes_sync_and_ranking():
    rng = random.Random(0)
    db_path = Path(tempfile.mkstemp(suffix=".db")[1])
    conn = connect_history_db(db_path)
    remote_votes = {}
    for i in range(240):
        liked = i % 3 != 0
        url = f"https://old/{i}"
        conn.execute(
            "INSERT INTO articles (url, category, source, title, embedding, first_seen_date) VALUES (?,?,?,?,?,?)",
            (url, "c", "src", "t", _pack_embedding(_vec(liked, rng)), "2026-10-01"),
        )
        remote_votes[url] = 1 if liked else -1
    remote_votes["https://not-in-db"] = 1
    today = [{"url": f"https://new/{i}", "title": f"n{i}", "source": "src", "liked": i % 2 == 0} for i in range(10)]
    for a in today:
        conn.execute(
            "INSERT INTO articles (url, category, source, title, embedding, first_seen_date) VALUES (?,?,?,?,?,?)",
            (a["url"], "c", "src", a["title"], _pack_embedding(_vec(a["liked"], rng)), "2026-10-10"),
        )
    conn.commit()
    conn.close()

    # Not enough votes yet -> untouched, same order.
    class _Resp:
        def __init__(self, data): self._data = data
        def raise_for_status(self): pass
        def json(self): return self._data
    few = dict(list(remote_votes.items())[:50])
    interest.requests.get = lambda url, headers, timeout: _Resp(few)
    cfg = {"url": "https://site", "key": "k"}
    assert interest.apply_votes(today, cfg, db_path=db_path) == today

    # Enough votes -> liked blob ranked first, disliked blob dropped.
    interest.requests.get = lambda url, headers, timeout: _Resp(remote_votes)
    ranked = interest.apply_votes(today, cfg, db_path=db_path)
    assert ranked and all(a["liked"] for a in ranked), [a["title"] for a in ranked]
    assert len(ranked) == 5, len(ranked)

    # No config, or the endpoint down -> never raises, nothing dropped by sync itself.
    assert len(interest.apply_votes(today, {}, db_path=db_path)) == 5  # votes already in the db still apply
    def _boom(*a, **k): raise interest.requests.ConnectionError("down")
    interest.requests.get = _boom
    assert len(interest.apply_votes(today, cfg, db_path=db_path)) == 5


if __name__ == "__main__":
    test_filter_by_interest()
    test_assign_sections()
    test_votes_sync_and_ranking()
    print("ok")
