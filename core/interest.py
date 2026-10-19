"""
core/interest.py

Personalization, in three independent pieces:

1. filter_by_interest() -- runs right after classify_all(). One yes/no
   LLM call per article against prompts/interest_filter.txt, which is a
   hand-condensed version of interest_profile.md (the y/m/n checklist).
   This is deliberately a SEPARATE call from classification rather than
   more rules in prompts/classifier.txt: the classifier's two Italian-
   category guards exist to stop it discarding real news, and they were
   also blocking legitimate junk discards (the Oct 10, 2026 run kept a
   Serie A preview, a tennis story and a car review because the
   second-look check didn't confirm SCARTA). "Is this news?" and "does
   this reader care?" are different questions; this stage only asks the
   second one. Not cached: ~0.5s per article, once a day.

2. apply_votes() -- runs right after dedupe_articles(). Pulls the 👍/👎
   votes cast on the published page (functions/api/vote.js, stored in
   Cloudflare KV) into history.db's `articles.vote` column, then, once
   there are enough of them, fits a logistic regression on the stored
   embeddings (+ one-hot source) and uses it to sort each day's articles
   and drop the ones it's confident the reader doesn't want.

3. assign_sections() -- runs last, just before summarization: files each
   surviving article under one of the digest's SECTIONS.

Same policy as every other stage: never raises, degrades to "do nothing".
"""

from __future__ import annotations

import logging
import sqlite3
import warnings
from collections import Counter
from pathlib import Path

import requests

from core.classification import _call_ollama_chat, _snippet, load_prompt
from core.deduplication import HISTORY_DB_PATH, _unpack_embedding, connect_history_db

logger = logging.getLogger(__name__)

# Digest sections: the "###" headings of interest_profile.md, in its order,
# minus "Politics and institutions" (politics is split by geography instead,
# so every article has exactly one home) and "Sport" (always dropped).
SECTIONS = [
    "Italy",
    "European Union / Europe",
    "United States",
    "Rest of the world",
    "Geopolitics and security",
    "Economy and finance",
    "Technology and AI",
    "Science, health, environment",
    "Crime and incidents",
    "Society and culture",
    "Transport and infrastructure",
]
# Where core/delivery.py shows a source's political-lean tag (L/CL/CR/R).
POLITICS_SECTIONS = set(SECTIONS[:5])
# Used when the section call fails or answers something off-list: the
# closest section for each source-bucket category from classification.py.
_SECTION_FALLBACK = {
    "Italian National/Regional": "Italy",
    "Italian Economy/Finance": "Economy and finance",
    "US/World": "Rest of the world",
    "Tech & AI": "Technology and AI",
    "Opinion/Geopolitics": "Geopolitics and security",
}

# assign_sections()'s second look (prompts/section_country.txt) answers with
# a country name, mapped to a section here. Oct 19, 2026: asked to pick one of
# ITALIA/USA/EUROPA/ALTRO instead, it answered ITALIA for St Petersburg, an
# Israeli election, Yemen and Australia (10 of 13 known cases right; reordering
# the options, 10 of 17 older ones). Country names: 30 of 33.
# INTERNAZIONALE or a failed call changes nothing.
_GEO_SECTIONS = set(SECTIONS[:4])
_EUROPE = {
    "albania", "andorra", "austria", "belgio", "bielorussia", "bosnia", "bosnia ed erzegovina", "bulgaria",
    "cipro", "croazia", "danimarca", "estonia", "finlandia", "francia", "germania", "grecia", "irlanda",
    "islanda", "kosovo", "lettonia", "liechtenstein", "lituania", "lussemburgo", "macedonia del nord", "malta",
    "moldavia", "monaco", "montenegro", "norvegia", "paesi bassi", "olanda", "polonia", "portogallo",
    "regno unito", "gran bretagna", "inghilterra", "scozia", "repubblica ceca", "romania", "russia", "serbia",
    "slovacchia", "slovenia", "spagna", "svezia", "svizzera", "ucraina", "ungheria", "unione europea", "europa",
    # the prompt asks for Italian, but an English article can get an English answer
    "belarus", "belgium", "croatia", "cyprus", "czech republic", "denmark", "finland", "france",
    "germany", "greece", "hungary", "iceland", "ireland", "latvia", "lithuania", "luxembourg", "moldova",
    "netherlands", "north macedonia", "norway", "poland", "portugal", "slovakia", "spain", "sweden",
    "switzerland", "ukraine", "united kingdom", "uk", "england", "scotland", "european union", "europe",
}


def _country_section(answer: str | None) -> str | None:
    lines = (answer or "").strip().splitlines()
    country = lines[0].strip('"\'.`* ').lower() if lines else ""
    if not country or country.startswith("internazional"):
        return None
    if country in ("italia", "italy", "vaticano", "san marino"):
        return "Italy"
    if country in ("stati uniti", "usa", "stati uniti d'america", "united states", "us"):
        return "United States"
    return "European Union / Europe" if country in _EUROPE else "Rest of the world"


VOTES_TIMEOUT = 20  # seconds
MIN_VOTES = 200  # below this the model stays off -- too few examples for 768 dimensions
MIN_PER_CLASS = 30  # ...and it needs a real share of both likes and dislikes
# ponytail: fixed cut-off on a class-balanced model's probability; lower it (or
# set to 0 to rank without dropping) if the digest starts losing things you want.
DROP_BELOW = 0.15


def filter_by_interest(articles: list[dict]) -> list[dict]:
    """Drops articles the interest-profile prompt answers SCARTA for.
    Anything else -- TIENI, an unparseable answer, an Ollama failure --
    keeps the article."""
    prompt_template = load_prompt("interest_filter.txt")
    kept = []
    for article in articles:
        raw = _call_ollama_chat(prompt_template.format(
            source=article.get("source", ""),
            category=article.get("category", ""),
            title=article.get("title", ""),
            snippet=_snippet(article),
        ))
        if raw and raw.strip().strip('"\'.` ').upper().startswith("SCARTA"):
            logger.info("Interest filter dropped: '%s' (%s)", article.get("title", ""), article.get("source", ""))
            continue
        kept.append(article)
    logger.info("Interest filter: %d article(s) in -> %d kept", len(articles), len(kept))
    return kept


def assign_sections(articles: list[dict]) -> list[dict]:
    """Replaces each article's source-bucket category with its digest
    section (one LLM call each, prompts/section.txt). Runs AFTER
    dedupe_articles() on purpose: cross-day dedup compares against the
    categories stored in history.db, which are the source-bucket ones."""
    prompt_template = load_prompt("section.txt")
    country_template = load_prompt("section_country.txt")
    out = []
    for article in articles:
        raw = _call_ollama_chat(prompt_template.format(
            source=article.get("source", ""), title=article.get("title", ""), snippet=_snippet(article),
        ))
        section = (raw or "").strip().splitlines()[0].strip('"\'.` ') if raw else ""
        # The prompt offers the short "Europe": the model kept answering that
        # instead of the full heading anyway.
        section = {"Europe": "European Union / Europe"}.get(section, section)
        # Second look, as a different question: where does the story take
        # place? Oct 12, 2026: Peru's president was filed under Italy, a
        # California tax under Rest of the world, and a piece on Germany got
        # the off-list answer "Germany". Only geographic or off-list answers
        # are checked, and Italy is trusted for a source classify_all() already
        # kept in an Italian category -- an Italian politician abroad stays there.
        italian_source = (article.get("category") or "").startswith("Italian")
        if section not in SECTIONS or (section in _GEO_SECTIONS and not (section == "Italy" and italian_source)):
            answer = _call_ollama_chat(country_template.format(
                source=article.get("source", ""), title=article.get("title", ""), snippet=_snippet(article),
            ))
            checked = _country_section(answer)
            if checked and checked != section:
                logger.info(
                    "Section second look: '%s' %r -> '%s'.", article.get("title", ""), section, checked,
                )
                section = checked
        if section not in SECTIONS:
            fallback = _SECTION_FALLBACK.get(article.get("category"), "Rest of the world")
            logger.warning(
                "No usable section for '%s' (model said %r) -- using '%s'.", article.get("title", ""), raw, fallback,
            )
            section = fallback
        out.append({**article, "category": section})
    logger.info("Sections: %s", ", ".join(f"{s}={n}" for s, n in Counter(a["category"] for a in out).most_common()))
    return out


def _ensure_vote_column(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(articles)").fetchall()}
    if "vote" not in cols:
        conn.execute("ALTER TABLE articles ADD COLUMN vote INTEGER")
        conn.commit()


def sync_votes(conn: sqlite3.Connection, votes_config: dict) -> int:
    """Copies {url: 1|-1} from the site's /api/vote into articles.vote.
    Returns how many rows were updated (0 on any failure/misconfiguration)."""
    base_url, key = votes_config.get("url"), votes_config.get("key")
    if not base_url or not key:
        logger.info("No votes.url/votes.key in config.yaml -- skipping vote sync.")
        return 0
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/vote", headers={"x-key": key}, timeout=VOTES_TIMEOUT)
        resp.raise_for_status()
        votes = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Could not fetch votes from %s: %s", base_url, exc)
        return 0
    updated = 0
    for url, v in votes.items():
        if v in (1, -1):
            updated += conn.execute("UPDATE articles SET vote = ? WHERE url = ?", (v, url)).rowcount
    conn.commit()
    logger.info("Vote sync: %d vote(s) on the site, %d matched an article in history.db.", len(votes), updated)
    return updated


def _features(embedding: list[float], source: str, sources: list[str]) -> list[float]:
    norm = sum(x * x for x in embedding) ** 0.5 or 1.0
    return [x / norm for x in embedding] + [1.0 if source == s else 0.0 for s in sources]


def rank_by_votes(articles: list[dict], conn: sqlite3.Connection) -> list[dict]:
    """Sorts `articles` by predicted P(like), most-liked first, and drops
    those under DROP_BELOW. Returns `articles` unchanged while there
    aren't enough votes yet (see MIN_VOTES/MIN_PER_CLASS)."""
    todays = {}
    for article in articles:
        row = conn.execute("SELECT embedding FROM articles WHERE url = ?", (article["url"],)).fetchone()
        if row and row[0]:
            todays[article["url"]] = _unpack_embedding(row[0])
    if not todays:
        return articles
    dim = len(next(iter(todays.values())))

    # Only votes whose embedding matches today's dimension: an embedding-model
    # change makes older vectors incomparable, not wrong.
    voted = [
        (_unpack_embedding(emb), source, vote)
        for emb, source, vote in conn.execute(
            "SELECT embedding, source, vote FROM articles WHERE vote IS NOT NULL AND embedding IS NOT NULL"
        )
        if len(emb) == dim * 4
    ]
    counts = Counter(vote for _, _, vote in voted)
    if len(voted) < MIN_VOTES or min(counts[1], counts[-1]) < MIN_PER_CLASS:
        logger.info(
            "Preference model inactive: %d usable vote(s) (%d like, %d dislike); needs %d total and %d of each.",
            len(voted), counts[1], counts[-1], MIN_VOTES, MIN_PER_CLASS,
        )
        return articles

    from sklearn.linear_model import LogisticRegressionCV

    sources = sorted({source for _, source, _ in voted})
    # neg_log_loss, not the default accuracy: when several regularisation
    # strengths classify equally well, accuracy picks the strongest one, which
    # squashes every probability toward 0.5 and makes DROP_BELOW meaningless.
    model = LogisticRegressionCV(cv=5, class_weight="balanced", max_iter=2000, scoring="neg_log_loss")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # sklearn 1.9 default-change notices
        model.fit([_features(e, s, sources) for e, s, _ in voted], [v for _, _, v in voted])
    like_col = list(model.classes_).index(1)

    scored = []
    for article in articles:
        emb = todays.get(article["url"])
        # No embedding (Ollama failure at dedup) -> can't judge; treat as neutral, never drop.
        score = model.predict_proba([_features(emb, article.get("source", ""), sources)])[0][like_col] if emb else 0.5
        scored.append((score, article))
    kept = [a for score, a in sorted(scored, key=lambda sa: -sa[0]) if score >= DROP_BELOW]
    for score, article in scored:
        if score < DROP_BELOW:
            logger.info("Preference model dropped (p=%.2f): '%s'", score, article.get("title", ""))
    logger.info(
        "Preference model (trained on %d vote(s)): %d article(s) in -> %d kept.",
        len(voted), len(articles), len(kept),
    )
    return kept


def apply_votes(articles: list[dict], votes_config: dict, db_path: Path = HISTORY_DB_PATH) -> list[dict]:
    conn = connect_history_db(db_path)
    try:
        _ensure_vote_column(conn)
        sync_votes(conn, votes_config)
        return rank_by_votes(articles, conn)
    except Exception:
        logger.exception("Vote-based ranking failed -- continuing with unranked articles.")
        return articles
    finally:
        conn.close()
