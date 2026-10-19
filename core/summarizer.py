"""
core/summarizer.py

Stage 3 of the pipeline (see news_pipeline_v2.md, section 5): map-reduce
summarization via local Ollama models (SUMMARY_MODEL for the map stage,
REDUCE_MODEL for the reduce stage -- see their comments, and CHANGELOG.md).

Map (micro_summarize): each deduplicated article gets its own small,
fast call -- a 1-2 sentence factual extraction (who/what/numbers/
timeline), always written in Italian regardless of the article's
original language (ANSA/Adnkronos are already Italian; BBC/Politico/
Wired/etc. are English -- see prompts/micro_summary.txt). Doing the
Italian conversion at this stage, not just at the final digest, keeps
stage 2's input uniform instead of asking the model to synthesize a
single Italian section from a mix of Italian and English summaries.

Reduce (synthesize_category): the per-category batch of micro-summaries
(not raw article bodies -- avoids "lost in the middle" on a 7B model, per
news_pipeline_v2.md 5) is sent to the "executive editor" prompt, which
produces the final bulleted, quantitative, high-density Italian section
for that category.

Both calls hit Ollama's local REST endpoint (POST /api/chat, stream:
false) with temperature 0.15-0.20 per news_pipeline_v2.md 5.3 -- literal
extraction over creative variation.

Caching: micro-summaries are cached in history.db (same `articles` table
core/deduplication.py already writes -- a `micro_summary` column is
added to it here) keyed by URL. This matters because Opinion/Geopolitics
articles can appear in more than one day's run (4-day lookback_days) and
would otherwise get re-summarized by the LLM every single day for
content that hasn't changed -- pure wasted GPU time. Cross-day-shown
suppression (core/deduplication.py) already filters out anything already
delivered, but an article can be "seen but not yet shown" across
multiple runs while its digest is still being iterated on/tested, so
caching by URL is worth doing independently of that.

Failure handling: if Ollama is unreachable, the model isn't pulled, or a
single call fails/times out, that's logged and the affected article (map
stage) or category (reduce stage) is skipped rather than crashing the
whole run -- same "degrade, don't crash" philosophy as every other
stage. A run that produces digest text for 4 of 5 categories because the
5th's Ollama call kept timing out is still a useful morning digest.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import requests

from core.deduplication import HISTORY_DB_PATH, connect_history_db

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
SUMMARY_MODEL = "qwen3.5:9b-q4_K_M"  # was qwen2.5:7b-instruct until Oct 12, 2026
# The reduce stage stays on the old model: given the same prompt and the same
# 13 Italy summaries, qwen3.5 paired unrelated neighbours into one bullet in
# 3 runs out of 3 (Conte with Casini, the ex Ilva strike with a column on the
# government), qwen2.5 in none. Ollama swaps the two once per run.
REDUCE_MODEL = "qwen2.5:7b-instruct"
TEMPERATURE = 0.175  # per news_pipeline_v2.md 5.3 (0.15-0.20 range)
MAP_TIMEOUT = 60  # seconds -- one small article, should be fast, but a cold model load can be slow once
REDUCE_TIMEOUT = 180  # seconds -- a whole category's worth of summaries in one prompt, more to generate
MAX_BODY_CHARS = 4000  # keep the map-stage prompt small/fast per news_pipeline_v2.md 5.1

# v2.1: cap on how many micro-summaries go into a single reduce-stage call.
# Opinion/Geopolitics and Italian National/Regional were routinely sending
# 30+ summaries into one prompt, and that turned out to correlate with two
# real bugs on a live run: the model merging unrelated stories into one
# bullet (its "same theme" merge instruction over-triggering under a long,
# hard-to-track list) and occasional non-Italian token leakage (context-size
# stress on a 7B model). synthesize_category() now sub-batches instead --
# see its docstring.
REDUCE_BATCH_SIZE = 15

# Matches a trailing "[n]"/"[n,m,...]" citation tag on a bullet line -- same
# pattern core/delivery.py's _CITATION_RE uses to parse the final digest
# text; kept in sync deliberately (see _remap_citation_tags).
_LINE_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]\s*$")

# v2.1.3 fix: CJK/other non-Latin-script token drift, confirmed AGAIN on a
# live run (Sept 27, 2026) even after the v2.1.2 fix in micro_summarize()
# (the empty-body/duplicate-title fallback below). That earlier fix only
# addressed ONE trigger (a near-empty body prompt); this run's corrupted
# bullet came from an article with a perfectly normal, non-empty body (The
# Daily Beast, "Laura Coates Reprimands Scott Jennings During CNN Debate on
# Iran") -- so the root cause is broader: a quantized multilingual 7B model
# occasionally drifting into Chinese-script tokens mid-generation, a known
# failure mode independent of input degeneracy. Every source article in
# this pipeline is Italian or English; nothing legitimate should ever
# contain CJK/Hangul/Kana script in the OUTPUT, so any match here is
# unambiguously corruption, never a false positive (e.g. a Chinese place
# name quoted in the original language). Fix: detect it generically on
# EVERY Ollama call (map, reduce, and comparison, since they all funnel
# through _call_ollama_chat), retry once, and if the retry is also
# corrupted, degrade to None (skip this article/bullet) rather than let
# corrupted text reach the digest -- same "degrade, don't crash" policy as
# every other failure mode in this pipeline.
_UNWANTED_SCRIPT_RE = re.compile(
    r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]"
)


def _contains_unwanted_script(text: str) -> bool:
    return bool(_UNWANTED_SCRIPT_RE.search(text))

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(filename: str) -> str:
    with open(PROMPTS_DIR / filename, "r", encoding="utf-8") as f:
        return f.read()


def _call_ollama_chat(
    prompt: str, timeout: int, _retry_on_corruption: bool = True, model: str = SUMMARY_MODEL
) -> str | None:
    """POST to Ollama's local chat endpoint. Returns the assistant's
    message text, or None on any failure -- never raises, so one bad
    call degrades (skip this article/category) rather than crashing the
    run.

    v2.1.3: also guards against CJK/non-Latin-script token drift (see
    _contains_unwanted_script's comment) by retrying the exact same call
    once, then giving up (None) if the retry is corrupted too."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
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
            logger.warning("Ollama chat returned empty content for model %s", model)
            return None
        content = content.strip()
        if _contains_unwanted_script(content):
            if _retry_on_corruption:
                logger.warning(
                    "Ollama chat output contained CJK/non-Latin-script drift (model %s) -- "
                    "retrying this call once. Corrupted output started: %r",
                    model, content[:120],
                )
                return _call_ollama_chat(prompt, timeout=timeout, _retry_on_corruption=False, model=model)
            logger.warning(
                "Ollama chat output still contained CJK/non-Latin-script drift after retry "
                "(model %s) -- giving up on this call, degrading to None. Corrupted output "
                "started: %r",
                model, content[:120],
            )
            return None
        return content
    except requests.RequestException as exc:
        logger.warning(
            "Ollama chat request failed -- is Ollama running with '%s' pulled? "
            "('ollama pull %s' if not): %s",
            model, model, exc,
        )
        return None


def call_summary_model(prompt: str, timeout: int) -> str | None:
    """Public wrapper around the primary-model (qwen2.5:7b-instruct) Ollama
    call, for reuse by other reduce-style callers outside this module --
    specifically core.comparison.py's cross-lean synthesis (v2.1), which
    needs the same model/temperature and "never raises, degrades to None"
    behavior as the executive-editor reduce stage, without duplicating
    _call_ollama_chat's logic."""
    return _call_ollama_chat(prompt, timeout=timeout)


def _truncate_body(body: str, max_chars: int = MAX_BODY_CHARS) -> str:
    body = body or ""
    if len(body) <= max_chars:
        return body
    return body[:max_chars].rsplit(" ", 1)[0] + "…"


def ensure_micro_summary_column(conn: sqlite3.Connection) -> None:
    """core/deduplication.py owns the `articles` table's core schema;
    this adds the one column summarizer.py itself needs, if missing.
    SQLite has no "ADD COLUMN IF NOT EXISTS", hence the PRAGMA check.
    Public (v2.1) because core/comparison.py also calls
    get_or_create_micro_summary() and needs this run first -- it runs
    BEFORE summarize_all() in main.py's pipeline order, so it can't rely
    on summarize_all() having already ensured the column exists."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(articles)").fetchall()}
    if "micro_summary" not in cols:
        conn.execute("ALTER TABLE articles ADD COLUMN micro_summary TEXT")
        conn.commit()


def _get_cached_micro_summary(conn: sqlite3.Connection, url: str) -> str | None:
    row = conn.execute("SELECT micro_summary FROM articles WHERE url = ?", (url,)).fetchone()
    return row[0] if row and row[0] else None


def _cache_micro_summary(conn: sqlite3.Connection, url: str, summary: str) -> None:
    """Requires a row for this URL to already exist in `articles` -- true
    in the normal pipeline flow, since core.deduplication.dedupe_articles()
    (via its internal _record_seen) always writes a row for every article
    it returns, before summarize_all() ever sees it. If this is ever
    called out of that order (e.g. summarize_all() invoked directly on
    articles that weren't run through dedupe_articles() against the same
    db_path), the UPDATE matches no row and silently no-ops -- so we check
    rowcount and log loudly, since a silent cache miss every run is a
    wasted-GPU-time bug that's easy to miss otherwise."""
    cur = conn.execute("UPDATE articles SET micro_summary = ? WHERE url = ?", (summary, url))
    if cur.rowcount == 0:
        logger.warning(
            "Could not cache micro-summary for %s -- no matching row in articles table "
            "(was this article run through dedupe_articles() against the same db_path first?)",
            url,
        )


def get_or_create_micro_summary(article: dict, conn: sqlite3.Connection, prompt_template: str) -> str | None:
    """Returns a cached micro-summary for this article's URL if present in
    history.db, otherwise calls the map-stage LLM and caches the result.
    Factored out (v2.1) so both summarize_all()'s normal per-category map
    stage and core/comparison.py's cross-lean comparison synthesis share
    identical caching behavior, and a URL that happens to feed both paths
    in the same run is only ever summarized once."""
    cached = _get_cached_micro_summary(conn, article["url"])
    if cached:
        return cached
    summary = micro_summarize(article, prompt_template)
    if summary is None:
        return None
    _cache_micro_summary(conn, article["url"], summary)
    conn.commit()
    return summary


NO_BODY_PREFIX = "Testo completo non disponibile"
MIN_BODY_CHARS_FOR_LLM = 40  # a body shorter than this (including empty, the
# common case) carries no more information than the title alone -- see
# micro_summarize()'s docstring for why this is handled without an LLM call.


_NUMBER_RE = re.compile(r"\d[\d.,]*\d|\d")


def _number_key(number: str) -> str:
    # "74.000", "74,000" and the "74" of "74 mila" all compare equal.
    return re.sub(r"[.,]", "", number).rstrip("0") or "0"


def unsupported_numbers(summary: str, source: str) -> list[str]:
    """Numbers in a micro-summary that appear nowhere in the article it
    summarizes. Oct 12, 2026: a Washington Post piece on this year's
    midterms came out dated "novembre 2022", a Politico one grew a 2029
    election. Small numbers are skipped (the article may spell them out,
    "six" -> 6) and so is the current year.
    ponytail: a rounded figure ("6,1 milioni" for 6.118.775) is flagged too;
    it only costs one extra rewrite call."""
    known = {_number_key(n) for n in _NUMBER_RE.findall(source)} | {_number_key(str(datetime.now().year))}
    return [
        n for n in _NUMBER_RE.findall(summary)
        if int(re.sub(r"[.,]", "", n)) > 12 and _number_key(n) not in known
    ]


def micro_summarize(article: dict, prompt_template: str) -> str | None:
    """Stage 1 (map): one small Ollama call per article. Returns a 1-2
    sentence Italian factual summary, or None if Ollama failed.

    v2.1.2 fix: when full-text extraction failed at ingestion (empty
    "body" -- see core/ingestion.py's "keeping title/link only" warning;
    common for NYT under this environment's current 403s on article
    pages), this used to fall back to feeding the model the title AS the
    body, i.e. a prompt whose "Testo:" field just repeats "Titolo:"
    verbatim. Traced every language-corrupted digest bullet from a live
    run back to exactly this case: that near-empty, self-duplicating
    input is what was triggering qwen2.5:7b's occasional bilingual/
    Chinese-token drift (see news_pipeline_v2.md changelog for the
    root-cause trace). There is no real summarization task in "summarize
    this title using itself as the source text," so this case is now
    handled deterministically with no LLM call at all -- zero corruption
    risk, and the article still surfaces in the digest (honestly flagged
    as full-text-unavailable) rather than being silently dropped."""
    sources = ", ".join(article.get("sources") or [article.get("source", "")])
    body = (article.get("body") or "").strip()
    if len(body) < MIN_BODY_CHARS_FOR_LLM:
        title = article["title"]
        return f"{NO_BODY_PREFIX} per questo articolo; titolo originale: «{title}»."
    prompt = prompt_template.format(
        source=sources,
        title=article["title"],
        body=_truncate_body(body),
    )
    summary = _call_ollama_chat(prompt, timeout=MAP_TIMEOUT)
    invented = unsupported_numbers(summary, f"{article['title']} {body}") if summary else []
    if invented:
        # Second pass, only for the summaries that fail the check: the model
        # is shown its own summary and the numbers it made up.
        logger.warning("Micro-summary of '%s' has numbers not in the article: %s -- rewriting", article["title"], invented)
        fix = load_prompt("micro_fix.txt").format(numbers=", ".join(invented), today=f"{datetime.now():%d/%m/%Y}")
        rewritten = _call_ollama_chat(f"{prompt} {summary}{fix}", timeout=MAP_TIMEOUT)
        if rewritten:
            summary = rewritten
            still = unsupported_numbers(summary, f"{article['title']} {body}")
            if still:
                logger.warning("...rewrite of '%s' still has unsupported numbers: %s", article["title"], still)
    return summary


def remap_citation_tags(text: str, offset: int) -> str:
    """Rewrites every trailing "[n]"/"[n,m,...]" tag in a block of bullet
    lines by adding `offset` to each number. Needed because a sub-batched
    reduce call (see synthesize_category) only ever shows the model
    chunk-local 1-based numbering (1..len(batch)), but core/delivery.py's
    citation resolution (news_pipeline_v2.md 5.5/6.7) expects numbers that
    index into the category's FULL articles/summaries lists. A line with
    no resolvable tag is passed through unchanged (delivery.py already
    treats a missing/malformed tag as "plain bullet, no expansion", so
    there's nothing to fix up). Public (not `_`-prefixed) because
    core/comparison.py (v2.1) reuses it for the same reason -- its
    per-cluster comparison calls are also locally numbered and need
    remapping to the section-wide index."""
    out_lines = []
    for line in text.splitlines():
        m = _LINE_CITATION_RE.search(line)
        if not m:
            out_lines.append(line)
            continue
        remapped = ",".join(str(int(n.strip()) + offset) for n in m.group(1).split(","))
        out_lines.append(f"{line[:m.start()]}[{remapped}]")
    return "\n".join(out_lines)


# A bullet is a condensed rewrite of the summaries it came from, so nearly all
# of its words appear in the right summary and almost none in a wrong one
# (measured on every debug/*.md up to Oct 12, 2026: >0.9 vs <0.1).
# ponytail: plain word overlap; switch to the stored embeddings if the model
# starts paraphrasing enough for right matches to fall under this.
MIN_CITATION_OVERLAP = 0.25

# The example sentence in prompts/executive_editor.txt, which the model copied
# into the digest as if it were news on 5 of the 14 days up to Oct 12, 2026.
# Rewording the example made the tags worse, so it is cut from the output here.
_PROMPT_EXAMPLE_RE = re.compile(r"Il PIL è cresciuto del 2% nel secondo trimestre[.,;]?\s*")


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"\w{4,}", text.lower())}


def repair_citations(text: str, batch: list[str], category: str = "") -> str:
    """Checks every bullet's "[n]" tag against the summaries it claims to
    come from, instead of trusting the model's numbering. Oct 12, 2026: the
    Italy section came back with its tags shifted by one from the fifth
    bullet on, so core/delivery.py rendered each bullet above another
    story's sources. A cited summary sharing too few words with the bullet
    is dropped from the tag; if none is left, the best-matching summary is
    used; if nothing matches at all the line is left as the model wrote it.
    Summaries no bullet ends up citing are appended as their own bullets, so
    an article is never marked as shown without being on the page."""
    summary_stems = [_stems(s) for s in batch]
    used: set[int] = set()
    out = []
    for line in _PROMPT_EXAMPLE_RE.sub("", text).splitlines():
        m = _LINE_CITATION_RE.search(line)
        body = line[: m.start()].rstrip() if m else line.rstrip()
        cited = [int(n) - 1 for n in m.group(1).split(",")] if m else []
        words = _stems(body)
        if not words:
            continue
        scores = [len(words & s) / len(words) if words else 0.0 for s in summary_stems]
        keep = [i for i in cited if 0 <= i < len(batch) and scores[i] >= MIN_CITATION_OVERLAP]
        if not keep and max(scores, default=0.0) >= MIN_CITATION_OVERLAP:
            keep = [scores.index(max(scores))]
        if keep and keep != cited:
            logger.warning(
                "%s: citation tag %s -> %s on bullet %r", category,
                [i + 1 for i in cited], [i + 1 for i in keep], body[:80],
            )
            line = f"{body} [{','.join(str(i + 1) for i in keep)}]"
        if len(keep) == 1:
            # A one-source bullet is its summary verbatim: rewording it only
            # added errors. Oct 19, 2026: "Marattin offers himself to the
            # centre-right, Calenda attacks him" became "Marattin joined the
            # centre-right, criticising Calenda". The reduce model still
            # groups and orders bullets.
            if keep[0] in used:
                continue  # the model split one story into two bullets
            line = f"- {' '.join(batch[keep[0]].split())} [{keep[0] + 1}]"
        used.update(keep or cited)
        out.append(line)
    for i, summary in enumerate(batch):
        if i not in used:
            logger.warning("%s: summary %d not cited by any bullet -- appending it as its own bullet", category, i + 1)
            out.append(f"- {' '.join(summary.split())} [{i + 1}]")
    return "\n".join(out)


def _reduce_batch(category: str, batch: list[str], prompt_template: str) -> str | None:
    # Title-only articles are not shown to the model: with nothing but a
    # title to go on it filled the gap from the neighbouring summaries (Oct
    # 12, 2026: a cardinal's comment on the Pope's surgery became "the Pope's
    # choice to marry a young woman", next to a child-marriage story).
    # repair_citations() appends them as their own bullets, unedited.
    numbered = [f"{i + 1}. {s}" for i, s in enumerate(batch) if not s.startswith(NO_BODY_PREFIX)]
    if not numbered:
        return repair_citations("", batch, category)
    prompt = prompt_template.format(category=category, count=len(numbered), summaries="\n".join(numbered))
    text = _call_ollama_chat(prompt, timeout=REDUCE_TIMEOUT, model=REDUCE_MODEL)
    return repair_citations(text, batch, category) if text else None


def synthesize_category(category: str, micro_summaries: list[str], prompt_template: str) -> str | None:
    """Stage 2 (reduce): batches a category's micro-summaries (not raw
    article bodies) into the executive-editor prompt. Returns a Markdown
    bulleted section, or None if every reduce call failed.

    v2.1: categories with more than REDUCE_BATCH_SIZE micro-summaries are
    split into several sequential reduce calls instead of one giant prompt
    -- see the REDUCE_BATCH_SIZE constant's comment for why (a 30+-item
    single call was implicated in both cross-topic bullet merges and
    non-Italian token leakage on a live run). Each batch is numbered
    locally (1..len(batch)) for the model, then _remap_citation_tags()
    rewrites those numbers back to the category-wide indices delivery.py
    expects before the batches' bullet lists are concatenated. If one
    batch's call fails, its bullets are simply missing from the section
    (logged, not fatal) -- same "degrade, don't crash" rule as everywhere
    else; the whole category only comes back None if every batch failed."""
    if len(micro_summaries) <= REDUCE_BATCH_SIZE:
        return _reduce_batch(category, micro_summaries, prompt_template)

    batch_count = (len(micro_summaries) + REDUCE_BATCH_SIZE - 1) // REDUCE_BATCH_SIZE
    logger.info(
        "%s: %d micro-summaries exceeds REDUCE_BATCH_SIZE=%d -- splitting reduce stage into %d batches",
        category, len(micro_summaries), REDUCE_BATCH_SIZE, batch_count,
    )
    sections: list[str] = []
    for offset in range(0, len(micro_summaries), REDUCE_BATCH_SIZE):
        batch = micro_summaries[offset : offset + REDUCE_BATCH_SIZE]
        chunk_text = _reduce_batch(category, batch, prompt_template)
        if chunk_text is None:
            logger.warning(
                "%s: reduce sub-batch for summaries %d-%d (of %d) failed -- those bullets will be "
                "missing from today's section, not the whole category.",
                category, offset + 1, offset + len(batch), len(micro_summaries),
            )
            continue
        sections.append(remap_citation_tags(chunk_text, offset))

    if not sections:
        return None
    return "\n".join(sections)


def summarize_all(articles: list[dict], db_path: Path = HISTORY_DB_PATH) -> dict[str, dict]:
    """Run the full map-reduce over a deduplicated article list (as
    produced by core.deduplication.dedupe_articles()). Returns:

        {category: {"text": digest_markdown_text, "articles": [article, ...], "summaries": [str, ...]}}

    -- only for categories where at least one article was successfully
    micro-summarized AND the reduce call succeeded. "articles" is the
    exact list of article dicts whose micro-summaries fed that section
    (i.e. excluding any article that was skipped because its own
    micro-summary call failed) -- core/delivery.py needs this precise
    list, not the original input list, to call
    core.deduplication.mark_as_shown() only on what actually made it
    into the rendered digest (see news_pipeline_v2.md 6.2). Never raises;
    failures are logged and degrade the output (missing/partial
    sections) rather than crashing the run.

    "summaries" is index-aligned with "articles" (summaries[i] is the
    micro-summary for articles[i]) and is exactly the numbered list sent
    to the reduce-stage prompt (1-based in the prompt, so prompt item
    "n" == summaries[n - 1] == articles[n - 1]). This alignment is what
    lets core/delivery.py resolve the "[n]" source-citation tags the
    reduce prompt is now instructed to append to each bullet (see
    prompts/executive_editor.txt) back to a specific article -- added
    so the rendered digest can offer a per-bullet "read the detailed
    summary + source" expansion instead of only the compact synthesized
    line (see news_pipeline_v2.md 5.5/6.7).

    Micro-summaries are sequential, one Ollama call at a time -- there's
    a single local GPU behind this, so concurrent requests wouldn't
    actually run in parallel, just queue up at the server."""
    map_prompt = load_prompt("micro_summary.txt")
    reduce_prompt = load_prompt("executive_editor.txt")

    conn = connect_history_db(db_path)
    ensure_micro_summary_column(conn)

    try:
        by_category: dict[str, list[tuple[dict, str]]] = {}
        for article in articles:
            summary = get_or_create_micro_summary(article, conn, map_prompt)
            if summary is None:
                logger.warning(
                    "Skipping article (micro-summary failed): '%s' (%s)",
                    article["title"], article["url"],
                )
                continue
            by_category.setdefault(article["category"], []).append((article, summary))

        digest: dict[str, dict] = {}
        for category, pairs in by_category.items():
            cat_articles = [a for a, _ in pairs]
            summaries = [s for _, s in pairs]
            section = synthesize_category(category, summaries, reduce_prompt)
            if section is None:
                logger.error(
                    "Category synthesis failed for '%s' (%d micro-summaries available) -- "
                    "section will be missing from today's digest",
                    category, len(summaries),
                )
                continue
            digest[category] = {"text": section, "articles": cat_articles, "summaries": summaries}
            logger.info("%s: synthesized from %d micro-summar(y/ies)", category, len(summaries))

        return digest
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
    from core.classification import classify_all
    from core.deduplication import dedupe_articles
    from core.ingestion import ingest_all

    _setup_logging()

    print("Running ingestion...")
    articles = ingest_all()
    print(f"Ingested {len(articles)} article(s) total.")

    print("Running classification (reclassify + drop junk)...")
    classified = classify_all(articles)
    print(f"{len(classified)} article(s) kept after classification.")

    print("Running deduplication (does NOT mark anything as shown)...")
    deduped, _comparison_clusters = dedupe_articles(classified)
    print(f"{len(deduped)} article(s) after dedup.\n")

    print("Running summarization (map-reduce via Ollama -- this can take a while, "
          "one sequential call per article plus one per category)...")
    digest = summarize_all(deduped)

    total_categories = len({a["category"] for a in deduped}) if deduped else 0
    print(f"\n=== Digest sections produced: {len(digest)}/{total_categories} ===")
    print("(this test run does NOT call mark_as_shown() -- that's delivery.py's job, "
          "once a digest is actually rendered and deployed)")
    for category, section in digest.items():
        print(f"\n## {category} ({len(section['articles'])} article(s)) ##\n")
        print(section["text"])
