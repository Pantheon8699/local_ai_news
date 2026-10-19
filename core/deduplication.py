"""
core/deduplication.py

Stage 2 of the pipeline (see news_pipeline_v2.md, section 4).

Two distinct kinds of "don't show me the same thing twice" happen here,
solved two different ways:

1. Same-day duplicates: multiple outlets covering the same story in one
   run (e.g. ANSA and Adnkronos both reporting the same event). Solved
   via embedding similarity clustering *within each category* -- not
   keyword/Jaccard overlap, because Italian outlets often rephrase or
   translate wire stories rather than sharing headline vocabulary.

2. Cross-day repeats: with Opinion/Geopolitics now looking back up to 4
   days (see core/ingestion.py's "lookback_days"), the same piece can
   legitimately appear in more than one day's ingestion run before it
   ages out of the window. Solved via history.db: every article that has
   actually been shown in a delivered digest is recorded there (via
   mark_as_shown(), meant to be called by delivery.py once that stage
   exists), and this module filters incoming articles against that
   record *before* clustering -- by both URL and embedding similarity,
   so a story that resurfaces under a different URL/slug still gets
   caught.

Embeddings come from Ollama's local EMBEDDING_MODEL (see its comment) via its
REST API (http://localhost:11434/api/embeddings) using plain `requests`,
not curl_cffi -- this is loopback traffic with no bot detection to route
around (see core/ingestion.py's docstring for why that distinction
mattered there).

Failure handling: if Ollama is unreachable or the model isn't pulled,
that's logged once per article and that article falls back to being its
own unclustered singleton -- degraded (no dedup that run) rather than a
crashed run, consistent with the rest of the pipeline's philosophy.

IMPORTANT -- read before wiring this into anything: dedupe_articles()
never marks anything as "shown". That is a separate, explicit call
(mark_as_shown()) meant to be made by delivery.py after a digest has
actually been rendered and deployed. Calling dedupe_articles() repeatedly
(e.g. while testing this module in isolation, as the __main__ block
below does) is always safe and has no effect on future-run suppression;
only mark_as_shown() commits to "the user has actually seen this."

Schema note: news_pipeline_v2.md section 8 assigns "writes history.db"
to delivery.py. In practice the per-article dedup bookkeeping (the
`articles` table below: dedup keys, embeddings, first-seen/shown dates)
has to live here, since this is the module computing embeddings and
doing the comparisons -- delivery.py's ownership of history.db is the
separate `digests` table (the rendered-digest archive for the web app),
which doesn't exist yet since delivery.py doesn't exist yet. Both share
the same history.db file, different tables.

v2.1 addition -- comparison clusters: dedupe_articles() now returns a
second list alongside the usual deduped/merged articles: the raw
(unmerged) same-story clusters that span more than one political/
editorial "lean" (see core/ingestion.py's "lean" field, sourced from
sources.json) within the categories where that distinction matters
(COMPARISON_CATEGORIES below). This reuses the exact same embedding
clustering pass _cluster_by_similarity() already does for ordinary
same-day dedup -- no second embedding computation -- it just also keeps
a copy of the raw cluster (before _merge_cluster() collapses it to one
representative) whenever that cluster qualifies. core/comparison.py (new,
v2.1) consumes these to build the "Comparativa" digest section: a
same-story cluster that only has one lean represented has nothing to
compare, so it's not useful there even though it's still merged normally
for the main digest.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
# bge-m3 replaced nomic-embed-text on Oct 12, 2026: half the sources are
# Italian and half English, and nomic barely connected the two. On that day's
# ten known same-story pairs bge-m3 ranked every one in the top 7 of 2,080
# pairs; nomic ranked them from 1st to 1,022nd (see CHANGELOG.md).
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_TIMEOUT = 30  # seconds -- local inference, but a cold model load can be slow once
# Same-day clustering: two articles this close are the same story from two
# outlets. Set from bge-m3 scores on the Oct 8 and Oct 12, 2026 articles: real
# same-story pairs (also Italian/English) score 0.65-0.93, unrelated ones
# mostly under 0.60.
# ponytail: tuned on two days; raise it if distinct stories start merging
# (the nearest misses were two same-outlet pieces on the same person, ~0.70).
SIMILARITY_THRESHOLD = 0.65
# Cross-day "already shown": much stricter on purpose. A follow-up to a
# running story (yesterday "final vote today", today "the vote") scores
# 0.65-0.76 against the earlier piece and must NOT be suppressed; only the
# same article coming back under a new URL should be. Oct 8 vs Oct 6, 2026:
# 12 of 222 new articles matched an old one at 0.65, none at 0.80.
CROSS_DAY_THRESHOLD = 0.85
CROSS_DAY_LOOKBACK_DAYS = 10  # how far back to check "already shown" history

# v2.1: categories where cross-lean framing differences are actually
# meaningful. Italian Economy/Finance, Tech & AI and Opinion/Geopolitics
# are left out on purpose -- market data and tech-product facts don't have
# a "left/right" framing, and Opinion/Geopolitics is already explicitly
# opinion content from analysis outlets, not the domestic-politics press
# spectrum the Comparativa section is meant to surface.
COMPARISON_CATEGORIES = {"Italian National/Regional", "US/World"}
MIN_COMPARISON_LEANS = 2  # a cluster needs at least this many DISTINCT
                           # known leans (excluding "unknown") to be worth
                           # a side-by-side comparison.

# v2.1.3 fix: a live run (Sept 27, 2026) produced a "Comparativa" entry
# built from two ANSA - Mondo articles -- the SAME wire story, published
# by ANSA under two different regional sub-URLs the same day (identical
# title/body, different slugs), which the embedding clusterer correctly
# recognised as one story but which is obviously not a cross-perspective
# comparison: same outlet, same "wire" lean, twice. Root cause: the old
# qualification check only counted distinct LEAN VALUES, so a cluster
# that also happened to (falsely, or genuinely) include one differently-
# labelled article anywhere qualified, even though the two ACTUAL
# citations the model ended up using were the same non-opinion source.
# Two tightened rules, both requested directly (see news_pipeline_v2.md
# changelog, v2.1.3):
#   1. "wire" / "financial" / "analysis" / "neutral" are not political
#      sides -- only a genuine editorial stance counts toward diversity.
#   2. Comparativa is Italian-press-vs-Italian-press or
#      US/UK-press-vs-US/UK-press, never cross-market (see "market" field,
#      core/ingestion.py v2.1.3) -- comparing an Italian wire piece
#      against a US outlet's framing was never what was asked for.
OPINION_LEANS = {"left", "center-left", "center", "center-right", "right"}
MIN_COMPARISON_SOURCES = 2  # a cluster also needs at least this many
                             # DISTINCT outlets -- two URLs from the same
                             # outlet (a wire service cross-posting one
                             # story to two sections) is not "two outlets'
                             # takes," it's one outlet's take, twice.

HISTORY_DB_PATH = Path(__file__).resolve().parent.parent / "history.db"


def connect_history_db(db_path: Path = HISTORY_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            embedding BLOB,
            first_seen_date TEXT NOT NULL,
            shown_date TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_shown_date ON articles(shown_date)")
    conn.commit()
    return conn


def _pack_embedding(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def get_embedding(text: str) -> list[float] | None:
    """Call Ollama's local embeddings endpoint. Returns None on any
    failure (Ollama not running, model not pulled, timeout, ...) -- never
    raises, so one embedding failure degrades gracefully rather than
    killing the whole dedup pass."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=EMBEDDING_TIMEOUT,
        )
        resp.raise_for_status()
        embedding = resp.json().get("embedding")
        if not embedding:
            logger.warning("Ollama returned no embedding for text starting '%s...'", text[:60])
            return None
        return embedding
    except requests.RequestException as exc:
        logger.warning(
            "Embedding request failed -- is Ollama running with '%s' pulled? "
            "('ollama pull %s' if not): %s",
            EMBEDDING_MODEL, EMBEDDING_MODEL, exc,
        )
        return None


def _embedding_text(article: dict) -> str:
    """Title + first ~2 sentences of body, per news_pipeline_v2.md 4.2."""
    body = article.get("body") or ""
    # Cheap sentence split -- good enough for "first ~2 sentences"; a
    # missing/short body just means we embed the title plus whatever's there.
    sentences = body.replace("!", ".").replace("?", ".").split(".")
    snippet = ". ".join(s.strip() for s in sentences[:2] if s.strip())
    return f"{article['title']}. {snippet}".strip()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cluster_by_similarity(
    items: list[tuple[dict, "list[float] | None"]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[list[dict]]:
    """Greedy clustering: an article joins the first existing cluster
    whose representative embedding is similar enough; otherwise it starts
    a new cluster. Articles with no embedding (Ollama failure) always
    start their own singleton cluster -- can't safely judge similarity
    without one."""
    clusters: list[dict] = []  # each: {"rep_embedding": [...] | None, "articles": [...]}
    for article, embedding in items:
        placed = False
        if embedding is not None:
            for cluster in clusters:
                if cluster["rep_embedding"] is None:
                    continue
                if _cosine_similarity(embedding, cluster["rep_embedding"]) >= threshold:
                    cluster["articles"].append(article)
                    placed = True
                    break
        if not placed:
            clusters.append({"rep_embedding": embedding, "articles": [article]})
    return [c["articles"] for c in clusters]


def _merge_cluster(cluster: list[dict]) -> dict:
    """Pick the article with the longest body as the representative;
    merge the others' source names into it. Per news_pipeline_v2.md 4.2."""
    if len(cluster) == 1:
        article = dict(cluster[0])
        article["sources"] = [article["source"]]
        article["leans"] = [article.get("lean", "unknown")]
        return article
    representative = max(cluster, key=lambda a: len(a.get("body") or ""))
    merged = dict(representative)
    merged["sources"] = sorted({a["source"] for a in cluster})
    merged["leans"] = sorted({a.get("lean", "unknown") for a in cluster})  # for delivery.py's L/CL/CR/R tags
    return merged


def _load_shown_history(conn: sqlite3.Connection, today: date) -> list[dict]:
    """Articles marked as shown within CROSS_DAY_LOOKBACK_DAYS of today,
    for cross-day dedup comparison."""
    cutoff = (today - timedelta(days=CROSS_DAY_LOOKBACK_DAYS)).isoformat()
    rows = conn.execute(
        "SELECT url, category, embedding FROM articles WHERE shown_date IS NOT NULL AND shown_date >= ?",
        (cutoff,),
    ).fetchall()
    return [
        {"url": url, "category": category, "embedding": _unpack_embedding(emb) if emb else None}
        for url, category, emb in rows
    ]


def _already_shown(article: dict, embedding: "list[float] | None", shown_history: list[dict]) -> bool:
    for past in shown_history:
        if past["category"] != article["category"]:
            continue
        if past["url"] == article["url"]:
            return True
        # Same length only: history from before an embedding-model change is
        # not comparable (zip() would silently compare truncated vectors).
        if embedding is not None and past["embedding"] is not None and len(embedding) == len(past["embedding"]):
            if _cosine_similarity(embedding, past["embedding"]) >= CROSS_DAY_THRESHOLD:
                return True
    return False


def _record_seen(conn: sqlite3.Connection, article: dict, embedding: "list[float] | None", today: date) -> None:
    """Upsert this article into history.db as 'seen' (not shown) -- for
    debuggability and so re-running the same day doesn't lose its
    embedding. shown_date is left alone (stays NULL until
    mark_as_shown() is called; stays whatever it already was otherwise)."""
    conn.execute(
        """
        INSERT INTO articles (url, category, source, title, embedding, first_seen_date, shown_date)
        VALUES (?, ?, ?, ?, ?, ?, NULL)
        ON CONFLICT(url) DO UPDATE SET embedding = excluded.embedding
        """,
        (
            article["url"], article["category"], article["source"], article["title"],
            _pack_embedding(embedding) if embedding else None,
            today.isoformat(),
        ),
    )


def comparison_subset(cluster: list[dict]) -> "list[dict] | None":
    """Returns the largest same-MARKET subset of `cluster` that genuinely
    qualifies for a Comparativa entry, or None if no subset does. A
    subset qualifies when it spans >= MIN_COMPARISON_SOURCES distinct
    outlets AND >= MIN_COMPARISON_LEANS distinct OPINION_LEANS among
    those outlets -- see the v2.1.3 comment above OPINION_LEANS for the
    live bug (two ANSA - Mondo duplicates) this replaced a looser check
    to fix. Public because core/comparison.py re-runs this same check
    after micro-summarization (some articles in the original cluster may
    have dropped out by then) and again on the model's actual cited
    subset (the model isn't guaranteed to use every article it was
    given)."""
    by_market: dict[str, list[dict]] = {}
    for article in cluster:
        market = article.get("market", "unknown")
        if market == "unknown":
            continue  # can't safely group a cross-market comparison around an unlabeled source
        by_market.setdefault(market, []).append(article)

    best: "list[dict] | None" = None
    for group in by_market.values():
        sources = {a["source"] for a in group}
        leans = {a.get("lean", "unknown") for a in group if a.get("lean", "unknown") in OPINION_LEANS}
        if len(sources) >= MIN_COMPARISON_SOURCES and len(leans) >= MIN_COMPARISON_LEANS:
            if best is None or len(group) > len(best):
                best = group
    return best


def dedupe_articles(
    articles: list[dict], db_path: Path = HISTORY_DB_PATH
) -> tuple[list[dict], list[list[dict]]]:
    """Cross-day-filter, then same-day-cluster, a flat list of article
    dicts (as produced by core.classification.classify_all(), which by
    v2.1's pipeline order has already run by this point). Never raises --
    an Ollama failure degrades to "no dedup today," it doesn't crash the
    run. Does NOT record anything as "shown" -- see mark_as_shown().

    Returns a tuple (deduped, comparison_clusters):
    * deduped: a flat list of representative articles, each with a
      "sources" list merged in (e.g. ["ANSA - Politica", "Adnkronos"]) --
      same as before v2.1.
    * comparison_clusters (new, v2.1): the raw, UNMERGED same-day clusters
      that (a) belong to a category in COMPARISON_CATEGORIES and (b) span
      at least MIN_COMPARISON_LEANS distinct source leans. Computed from
      the exact same clustering pass as `deduped`, just captured before
      _merge_cluster() collapses each cluster to one representative --
      no extra embedding calls. core/comparison.py consumes this list to
      build the "Comparativa" section; every other caller can ignore it."""
    today = datetime.now().astimezone().date()
    conn = connect_history_db(db_path)
    try:
        shown_history = _load_shown_history(conn, today)

        by_category: dict[str, list[dict]] = {}
        for article in articles:
            by_category.setdefault(article["category"], []).append(article)

        deduped: list[dict] = []
        comparison_clusters: list[list[dict]] = []
        for category, cat_articles in by_category.items():
            items = []
            skipped_already_shown = 0
            for article in cat_articles:
                embedding = get_embedding(_embedding_text(article))
                if _already_shown(article, embedding, shown_history):
                    skipped_already_shown += 1
                    continue
                items.append((article, embedding))
                _record_seen(conn, article, embedding, today)

            clusters = _cluster_by_similarity(items)
            for cluster in clusters:
                deduped.append(_merge_cluster(cluster))
                if category in COMPARISON_CATEGORIES and len(cluster) > 1:
                    # v2.1.3: comparison_subset() replaces a bare "distinct
                    # lean count" check -- see its docstring and the
                    # OPINION_LEANS/MIN_COMPARISON_SOURCES comment above.
                    # It also trims the cluster down to just the qualifying
                    # same-market subset, so e.g. a lone ANSA - Mondo
                    # article riding along in an otherwise-qualifying
                    # US/UK cluster doesn't get pulled into a cross-market
                    # "comparison" it was never part of.
                    subset = comparison_subset(cluster)
                    if subset is not None:
                        comparison_clusters.append(subset)

            logger.info(
                "%s: %d ingested -> %d already shown (skipped) -> %d after same-day clustering",
                category, len(cat_articles), skipped_already_shown, len(clusters),
            )
        conn.commit()
        logger.info(
            "%d cross-lean comparison cluster(s) found across %s",
            len(comparison_clusters), sorted(COMPARISON_CATEGORIES),
        )
        return deduped, comparison_clusters
    finally:
        conn.close()


def mark_as_shown(articles: list[dict], db_path: Path = HISTORY_DB_PATH) -> None:
    """Record that these articles (by URL) actually appeared in a
    delivered digest -- future runs will suppress them via cross-day
    dedup. Intended to be called by delivery.py once a digest has
    genuinely been rendered and deployed. Do NOT call this from ad hoc
    testing of this module -- it commits to "the user has seen this,"
    and a test run isn't a real digest."""
    today = datetime.now().astimezone().date().isoformat()
    conn = connect_history_db(db_path)
    try:
        for article in articles:
            conn.execute("UPDATE articles SET shown_date = ? WHERE url = ?", (today, article["url"]))
        conn.commit()
    finally:
        conn.close()


def _setup_logging() -> None:
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"run_{datetime.now():%Y-%m-%d}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


if __name__ == "__main__":
    from core.ingestion import ingest_all  # local import: only needed for this manual test entry point

    _setup_logging()

    print("Running ingestion...")
    articles = ingest_all()
    print(f"Ingested {len(articles)} article(s) total.\n")

    print("Running deduplication (this run does NOT mark anything as shown)...")
    deduped, comparison_clusters = dedupe_articles(articles)

    print(f"\n=== {len(deduped)} article(s) after dedup (from {len(articles)} ingested) ===")
    print(f"=== {len(comparison_clusters)} cross-lean comparison cluster(s) found ===")
    by_category: dict[str, list[dict]] = {}
    for a in deduped:
        by_category.setdefault(a["category"], []).append(a)
    for cat, items in by_category.items():
        print(f"\n-- {cat} ({len(items)}) --")
        for a in items:
            sources = ", ".join(a.get("sources", [a["source"]]))
            print(f"  [{sources}] {a['title']}")
