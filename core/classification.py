"""
core/classification.py

New stage (v2.1), runs between ingestion and dedup: re-derives each
article's real topic category from its content, and drops non-news junk
(sport, weather, gossip, product reviews/deals) before it ever reaches
dedup/summarization.

Why this exists (see news_pipeline_v2.md changelog, v2.1 -- replaces
nothing, it's a new stage): core/ingestion.py sets article["category"]
to whichever sources.json bucket a feed lives under (see that module's
docstring). That's fine for a source with a genuinely single-topic feed
(ANSA's per-section RSS), but several sources -- Il Fatto Quotidiano, La
Repubblica, Libero, and formerly ANSA's own "Mondo" feed sitting inside
the "Italian National/Regional" bucket -- publish (or were filed as)
mixed-topic content: a general Italian outlet's feed carries politics,
sport, gossip and world news side by side, and a "Mondo" feed is
definitionally about other countries, not Italy. Trusting the source
bucket alone meant world-politics stories and entertainment/sport/weather
filler were showing up under "Italian National/Regional", and the Tech
section (all one source, Wired) skewed almost entirely to product
reviews and deals rather than actual tech/AI news.

This stage does one classification call per article. v2.1 originally
used the fast fallback model (llama3.2:3b) here as a deliberate cost
saving over the primary qwen2.5:7b-instruct, since classification into
six fixed labels looked like an easier task than synthesis. v2.1.3
REVERSED that decision: live runs showed llama3.2:3b unreliable
specifically at retaining/assigning the two Italian-language labels at
volume (see CLASSIFIER_MODEL's comment below for the evidence), so this
now uses the primary model too, paired with a second-look confirmation
step for the specific direction that was failing (see
_confirm_departure_from_italian).

Design choices, matching the rest of the pipeline's philosophy:
* The source-bucket category is passed to the model as a *hint*, not
  discarded -- the prompt (prompts/classifier.txt) instructs the model to
  keep the hint unless the content clearly belongs elsewhere, or unless
  it's junk. This avoids a small/fast model confidently reclassifying
  borderline cases wrong more often than the source bucket already was.
* SCARTA ("discard") is one of the six possible outputs, not a separate
  pass -- sport, weather, gossip/reality-TV, horoscopes, product
  reviews/buying guides/deals, and pure anecdote-with-no-current-
  relevance content are dropped here rather than flowing through dedup
  and burning a micro-summary call on content nobody asked for.
* Degrade, don't crash: a failed Ollama call, an empty/unparseable
  response, or a response that doesn't exactly match one of the six
  known labels all fall back to KEEPING the article's original
  source-bucket category rather than dropping it or guessing -- losing
  a legitimate story to a classifier hiccup would be worse than an
  occasional miscategorized one slipping through, which is exactly the
  failure mode this stage exists to reduce, not eliminate.
* Caching: like core/summarizer.py's micro-summary cache, classifications
  are cached in history.db (same `articles` table, new
  `category_override` column) keyed by URL -- Opinion/Geopolitics articles
  in particular can resurface across more than one day's run (4-day
  lookback_days), and there's no reason to re-classify unchanged content.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from pathlib import Path

import requests

from core.deduplication import HISTORY_DB_PATH, connect_history_db

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"

# v2.1.3 CHANGE (replaces the v2.1/10.3 decision to use llama3.2:3b here --
# see news_pipeline_v2.md changelog for the full trace): two consecutive
# live runs (the confirmation run in 10.10, and again on Sept 27, 2026)
# showed llama3.2:3b draining nearly ALL Italian-hinted articles out of
# "Italian National/Regional" -- on Sept 27, zero of 60 articles from
# Adnkronos/Il Fatto/Repubblica/Libero kept that label; every one was
# reclassified elsewhere or discarded, including plainly-wrong calls like
# discarding an acid-attack story and a minor's accidental-death story as
# SCARTA. That's not an occasional borderline miss, it's the 3B model
# failing at this specific task for Italian-language input at volume --
# so the "deliberate cost/latency saving" tradeoff from 10.3 no longer
# holds for classification quality, per this project's own practice of
# revisiting a past decision when a concrete reason shows up. Switching to
# the primary model (already proven reliable at Italian text elsewhere in
# this pipeline, and already resident on this machine -- no new VRAM ask)
# is slower per call, not free, so it's paired below with a second,
# narrower safety net (CONFIRM_DEPARTURE_FROM_ITALIAN) rather than trusted
# alone.
CLASSIFIER_MODEL = "qwen3.5:9b-q4_K_M"  # was qwen2.5:7b-instruct until Oct 12, 2026
TEMPERATURE = 0.0  # deterministic: this is a label-selection task, not
                    # generation, so there's no reason for any variation.
CLASSIFY_TIMEOUT = 45  # seconds -- bumped from 30s for the larger model (v2.1.3).
CONFIRM_TIMEOUT = 45  # seconds -- same model, same-sized prompt, for the
                       # second-look confirmation call (see classify_all()).
SNIPPET_CHARS = 300  # only need enough of the article to tell the topic,
                      # not the whole body -- keeps this call cheap.

ALLOWED_CATEGORIES = {
    "Italian National/Regional",
    "Italian Economy/Finance",
    "US/World",
    "Tech & AI",
    "Opinion/Geopolitics",
}
DISCARD_LABEL = "SCARTA"

# v2.1.2 fix: the two Italian-language categories. See ITALIAN_TARGET_GUARD
# below for why this set matters -- a live run's history.db showed the
# classifier reassigning articles from EVERY non-Italian source (NYT, BBC,
# Politico, Fox News, Washington Post, Daily Beast, NPR, Foreign Policy, War
# on the Rocks, Project Syndicate, even Wired) into "Italian National/
# Regional" -- 139 cached misclassifications across virtually every source
# in sources.json, not occasional borderline cases. That's consistent with
# llama3.2:3b defaulting toward whichever category prompts/classifier.txt
# lists first when it's not confident, regardless of the hint -- the
# prompt's "keep the hint if in doubt" instruction (classifier.txt rule 2)
# clearly isn't holding in practice for a 3B model at this volume.
ITALIAN_CATEGORIES = {"Italian National/Regional", "Italian Economy/Finance"}

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(filename: str) -> str:
    with open(PROMPTS_DIR / filename, "r", encoding="utf-8") as f:
        return f.read()


def _ensure_category_override_column(conn: sqlite3.Connection) -> None:
    """Same PRAGMA-check idiom as core/summarizer.py's
    _ensure_micro_summary_column -- core/deduplication.py owns the
    `articles` table's core schema; this adds the one column
    classification.py itself needs, if missing."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(articles)").fetchall()}
    if "category_override" not in cols:
        conn.execute("ALTER TABLE articles ADD COLUMN category_override TEXT")
        conn.commit()


def _get_cached_category(conn: sqlite3.Connection, url: str) -> str | None:
    row = conn.execute("SELECT category_override FROM articles WHERE url = ?", (url,)).fetchone()
    return row[0] if row and row[0] else None


def _cache_category(conn: sqlite3.Connection, url: str, category: str) -> None:
    """Same defensive rowcount check as core/summarizer.py's
    _cache_micro_summary -- requires a row for this URL to already exist
    (written by core.deduplication's _record_seen via dedupe_articles(),
    OR, since classification now runs BEFORE dedup in v2.1's pipeline
    order, this module writes its own placeholder row first if needed --
    see classify_all())."""
    cur = conn.execute("UPDATE articles SET category_override = ? WHERE url = ?", (category, url))
    if cur.rowcount == 0:
        logger.warning(
            "Could not cache classification for %s -- no matching row in articles table.", url,
        )


def _snippet(article: dict) -> str:
    body = (article.get("body") or "").strip()
    if not body:
        return article.get("title", "")
    if len(body) <= SNIPPET_CHARS:
        return body
    return body[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"


def _call_ollama_chat(prompt: str, timeout: int = CLASSIFY_TIMEOUT) -> str | None:
    """Never raises -- one failed classification call degrades to
    "keep the original category" (see classify_all()), not a crash.
    `timeout` is parametrized (v2.1.3) so the confirmation call
    (CONFIRM_TIMEOUT) can share this same request path."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": CLASSIFIER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # qwen3.5 reasons before answering unless told not to: a
                # one-word answer took 55s with it on, 0.5s with it off.
                "think": False,
                "options": {"temperature": TEMPERATURE},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content")
        if not content or not content.strip():
            logger.warning("Ollama classification call returned empty content for model %s", CLASSIFIER_MODEL)
            return None
        return content.strip()
    except requests.RequestException as exc:
        logger.warning(
            "Ollama classification request failed -- is Ollama running with '%s' pulled? "
            "('ollama pull %s' if not): %s",
            CLASSIFIER_MODEL, CLASSIFIER_MODEL, exc,
        )
        return None


def _parse_label(raw: str) -> str | None:
    """The model is asked for an exact label; this tolerates the common
    small-model failure modes (surrounding quotes/punctuation, a trailing
    period, stray whitespace) but does NOT try to fuzzy-match a garbled
    answer to the closest known label -- if it's not an exact match after
    light cleanup, that's treated as a parse failure (fall back to the
    original category), not a guess."""
    cleaned = raw.strip().strip('"\'.` \n\t')
    if cleaned in ALLOWED_CATEGORIES or cleaned == DISCARD_LABEL:
        return cleaned
    # A model sometimes answers on the first line only, with commentary after.
    first_line = cleaned.splitlines()[0].strip().strip('"\'.` ') if cleaned else ""
    if first_line in ALLOWED_CATEGORIES or first_line == DISCARD_LABEL:
        return first_line
    return None


def _confirm_departure_from_italian(
    article: dict, original_category: str, proposed_label: str, confirm_prompt_template: str,
) -> bool:
    """v2.1.3 safety net -- the "drain" direction (see module comment
    above ITALIAN_CATEGORIES): moving an article OUT of an Italian
    category, whether to another allowed category or to SCARTA. The
    v2.1.2 guard only ever protected the opposite direction (moving IN);
    a live run on Sept 27, 2026 showed the drain direction failing
    completely -- 0 of 60 Adnkronos/Il Fatto/Repubblica/Libero articles
    kept "Italian National/Regional" that morning, including real crime
    stories wrongly discarded as SCARTA. Rather than hard-blocking this
    direction entirely (which would defeat the original, legitimate
    reason classification.py exists -- a genuinely mixed-topic Italian
    feed sometimes DOES carry a real world-affairs or junk item), this
    asks a second, narrower, independent question before trusting the
    first classifier's proposed departure. Never raises: any failure to
    get a clear answer (Ollama down, unparseable response) means "don't
    confirm" -- keep the Italian category, same "degrade to the safer
    side" policy as everywhere else in this stage."""
    prompt = confirm_prompt_template.format(
        source=article.get("source", ""),
        hint_category=original_category,
        proposed_label=proposed_label,
        title=article.get("title", ""),
        snippet=_snippet(article),
    )
    raw = _call_ollama_chat(prompt, timeout=CONFIRM_TIMEOUT)
    if raw is None:
        return False
    cleaned = raw.strip().strip('"\'.` \n\t').upper()
    first_word = cleaned.splitlines()[0].strip() if cleaned else ""
    return first_word == "CONFERMA"


def classify_article(article: dict, prompt_template: str) -> str | None:
    """Returns one of ALLOWED_CATEGORIES, DISCARD_LABEL, or None (meaning:
    couldn't classify -- caller should keep the article's existing
    category rather than treat this as a discard)."""
    prompt = prompt_template.format(
        source=article.get("source", ""),
        hint_category=article.get("category", ""),
        title=article.get("title", ""),
        snippet=_snippet(article),
    )
    raw = _call_ollama_chat(prompt)
    if raw is None:
        return None
    label = _parse_label(raw)
    if label is None:
        logger.warning(
            "Unparseable classification response for '%s': %r -- keeping original category '%s'",
            article.get("title", ""), raw, article.get("category"),
        )
    return label


def classify_all(articles: list[dict], db_path: Path = HISTORY_DB_PATH) -> list[dict]:
    """Runs classification over every ingested article, before dedup.

    Returns a new list: articles labeled SCARTA are dropped; articles
    successfully reclassified have article["category"] overwritten;
    articles that couldn't be classified (Ollama failure, unparseable
    response) keep their original source-bucket category unchanged.
    Never raises -- classification failures degrade the accuracy of
    categorization, they never drop a legitimate article or crash the run.

    Note on ordering: this runs BEFORE core.deduplication.dedupe_articles(),
    so (unlike summarizer.py's micro-summary cache) the `articles` table
    row for a given URL may not exist yet when this tries to cache a
    result. This function writes a minimal placeholder row itself in that
    case -- dedupe_articles()'s own _record_seen() upsert will fill in the
    rest (embedding, first_seen_date) right after, on the same row."""
    conn = connect_history_db(db_path)
    _ensure_category_override_column(conn)
    prompt_template = load_prompt("classifier.txt")
    confirm_prompt_template = load_prompt("classifier_confirm.txt")

    kept: list[dict] = []
    discarded = 0
    reclassified = 0
    kept_unchanged = 0
    blocked_into_italian = 0
    drain_confirmed = 0
    drain_blocked = 0
    # v2.1.3: (original_category -> final_category) counts, logged at the
    # end -- see news_pipeline_v2.md 7.1's own recommendation that this
    # class of bug (a category silently drained to zero) needs to be
    # visible in the run log itself, not just via a direct history.db
    # query after the fact.
    distribution: Counter = Counter()

    try:
        for article in articles:
            url = article["url"]
            cached = _get_cached_category(conn, url)
            if cached is not None:
                label = cached
            else:
                label = classify_article(article, prompt_template)
                if label is not None:
                    # Ensure a row exists to cache against (see docstring) --
                    # harmless no-op UPDATE below if dedup already wrote one
                    # for this URL earlier in the same run (cross-source
                    # duplicate URLs shouldn't happen, but INSERT OR IGNORE
                    # keeps this safe either way).
                    conn.execute(
                        """
                        INSERT INTO articles (url, category, source, title, first_seen_date, shown_date)
                        VALUES (?, ?, ?, ?, date('now'), NULL)
                        ON CONFLICT(url) DO NOTHING
                        """,
                        (url, article.get("category", ""), article.get("source", ""), article.get("title", "")),
                    )
                    _cache_category(conn, url, label)
                    conn.commit()

            original_category = article.get("category")

            if label is None:
                kept_unchanged += 1
                distribution[(original_category, original_category)] += 1
                kept.append(article)
                continue

            if label == original_category:
                # Classifier agreed with the hint -- nothing to guard.
                distribution[(original_category, original_category)] += 1
                kept.append(article)
                continue

            # A change IS being proposed (reclassify to another allowed
            # category, or discard as SCARTA). Two directional guards, in
            # order -- see each guard's own comment for the live-run
            # evidence behind it.
            if label in ITALIAN_CATEGORIES and original_category not in ITALIAN_CATEGORIES:
                # v2.1.2 guard (bleed: non-Italian hint -> Italian label).
                blocked_into_italian += 1
                logger.warning(
                    "Classifier tried to move '%s' (%s) from '%s' into '%s' -- blocked, "
                    "keeping original category (see v2.1.2 changelog).",
                    article.get("title", ""), article.get("source", ""), original_category, label,
                )
                distribution[(original_category, original_category)] += 1
                kept.append(article)
                continue

            if original_category in ITALIAN_CATEGORIES:
                # v2.1.3 guard (drain: Italian hint -> anything else,
                # including SCARTA). See _confirm_departure_from_italian's
                # docstring for why this direction needed its own guard.
                if _confirm_departure_from_italian(article, original_category, label, confirm_prompt_template):
                    drain_confirmed += 1
                else:
                    drain_blocked += 1
                    logger.warning(
                        "Classifier tried to move '%s' (%s) OUT of '%s' into '%s' -- not "
                        "confirmed by second-look check, keeping original category "
                        "(see v2.1.3 changelog).",
                        article.get("title", ""), article.get("source", ""), original_category, label,
                    )
                    distribution[(original_category, original_category)] += 1
                    kept.append(article)
                    continue

            # Either this wasn't an Italian-category departure, or it was
            # and the second-look check just confirmed it -- apply it.
            if label == DISCARD_LABEL:
                discarded += 1
                distribution[(original_category, "SCARTA")] += 1
                continue

            article = dict(article)
            article["category"] = label
            reclassified += 1
            distribution[(original_category, label)] += 1
            kept.append(article)

        logger.info(
            "Classification: %d article(s) in -> %d discarded (junk/off-topic), "
            "%d reclassified, %d kept as-is or unclassifiable, %d blocked reclassification(s) "
            "into an Italian category from a non-Italian hint, %d confirmed departure(s) from "
            "an Italian category, %d departure(s) from an Italian category NOT confirmed "
            "(kept Italian) -> %d out",
            len(articles), discarded, reclassified, kept_unchanged, blocked_into_italian,
            drain_confirmed, drain_blocked, len(kept),
        )
        for original_category in sorted({oc for oc, _ in distribution}):
            row = {fc: n for (oc, fc), n in distribution.items() if oc == original_category}
            logger.info(
                "Classification distribution for original category '%s': %s",
                original_category,
                ", ".join(f"{fc}={n}" for fc, n in sorted(row.items(), key=lambda kv: -kv[1])),
            )
        return kept
    finally:
        conn.close()


def _setup_logging() -> None:
    import logging as _logging
    from datetime import datetime as _datetime
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"run_{_datetime.now():%Y-%m-%d}.log"
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[_logging.FileHandler(log_file, encoding="utf-8"), _logging.StreamHandler()],
    )


if __name__ == "__main__":
    from core.ingestion import ingest_all

    _setup_logging()

    print("Running ingestion...")
    articles = ingest_all()
    print(f"Ingested {len(articles)} article(s) total.\n")

    print("Running classification (does NOT run dedup/summarize)...")
    classified = classify_all(articles)

    print(f"\n=== {len(classified)} article(s) kept after classification (from {len(articles)} ingested) ===")
    by_category: dict[str, list[dict]] = {}
    for a in classified:
        by_category.setdefault(a["category"], []).append(a)
    for cat, items in by_category.items():
        print(f"\n-- {cat} ({len(items)}) --")
        for a in items[:5]:
            print(f"  [{a['source']} / {a.get('lean', 'unknown')}] {a['title']}")
