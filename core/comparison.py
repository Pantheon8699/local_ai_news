"""
core/comparison.py

New stage (v2.1): builds an optional "Comparativa" digest section showing
how outlets of different political/editorial lean covered the same
story, side by side -- added on request, to counter the fact that the
rest of the digest necessarily reflects whatever mix of sources'
framing happened to win each dedup cluster (see core/deduplication.py's
_merge_cluster(), which keeps one representative article and discards
the rest).

This does NOT change how the normal per-category sections are built.
core.deduplication.dedupe_articles() (v2.1) now returns a second value,
`comparison_clusters`: the raw, unmerged same-day clusters that span at
least two distinct source leans within the categories where that
distinction is meaningful (COMPARISON_CATEGORIES, in deduplication.py).
A story in one of those clusters still gets its normal single-
representative bullet in its normal section too -- Comparativa is an
addition, not a replacement, so a genuinely divisive story gets both the
compact factual bullet and the multi-perspective breakdown.

Pipeline position: this runs AFTER dedupe_articles() (needs its cluster
output) and can run independently of / in parallel with summarize_all()
conceptually, but in practice main.py runs it right after summarize_all()
and merges its result into that same digest dict, since both need the
per-article micro-summary step and share its cache (see
core.summarizer.get_or_create_micro_summary).

Design choices, matching the rest of the pipeline:
* Reuses the map-stage micro-summary cache (core.summarizer.
  get_or_create_micro_summary) rather than re-summarizing or feeding raw
  article bodies to the comparison prompt -- an article that's ALSO the
  representative of its normal-section dedup cluster will already have a
  cached micro-summary by the time this runs (or vice versa); either way
  the LLM call happens at most once per URL per run.
* One reduce-style Ollama call per qualifying cluster (not one giant call
  for all of them), each producing exactly one bullet -- clusters are
  small (typically 2-4 articles) so this stays cheap, and keeping each
  call scoped to one topic avoids the same cross-topic-merge risk that
  motivated summarizer.py's reduce-stage sub-batching.
* Degrade, don't crash: a cluster whose micro-summarization or comparison
  call fails is skipped (logged), not fatal to the section; if EVERY
  cluster fails, build_comparison_section() returns None and main.py
  simply doesn't add a "Comparativa" key to the digest that day --
  core/delivery.py already renders whatever categories are present in
  the digest dict, so a missing Comparativa section just doesn't appear
  (no special-casing needed there).
* Output shape matches core.summarizer.summarize_all()'s per-category
  contract exactly -- {"text": markdown, "articles": [...],
  "summaries": [...]} -- specifically so core/delivery.py's existing
  citation-resolution and <details>/<summary> rendering (news_pipeline_v2.md
  5.5/6.7) works on "Comparativa" with ZERO changes to delivery.py: it's
  just another category dict as far as that module is concerned.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from core.deduplication import HISTORY_DB_PATH, comparison_subset, connect_history_db
from core.summarizer import (
    REDUCE_TIMEOUT,
    call_summary_model,
    ensure_micro_summary_column,
    get_or_create_micro_summary,
    load_prompt,
    remap_citation_tags,
)

logger = logging.getLogger(__name__)

# v2.1.3: matches a trailing "[n]"/"[n,m,...]" tag, same pattern as
# core/summarizer.py's _LINE_CITATION_RE / core/delivery.py's _CITATION_RE
# (kept in sync deliberately). Used to validate the model's own bullet
# BEFORE accepting it -- see _validate_bullet()'s docstring for why this
# check was added (a live run produced a bullet whose "[1]" tag sat mid-
# sentence, not at the end, and cited only one of the cluster's outlets).
_TRAILING_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]\s*$")

_LEAN_LABELS = {
    "left": "sinistra",
    "center-left": "centro-sinistra",
    "center": "centro",
    "center-right": "centro-destra",
    "right": "destra",
    "wire": "agenzia",
    "financial": "finanziaria",
    "analysis": "analisi",
    "neutral": "neutra",
    "unknown": "non specificata",
}


def _lean_label(lean: str) -> str:
    return _LEAN_LABELS.get(lean, lean)


def _summarize_cluster(
    cluster: list[dict], conn, map_prompt: str
) -> tuple[list[dict], list[str]]:
    """Micro-summarizes every article in a cluster (via the shared cache),
    dropping any whose summarization failed. Returns index-aligned
    (articles, summaries) lists, same convention as summarize_all().

    Relies on core.deduplication.dedupe_articles() having already run
    against the same history.db and written an `articles` row for every
    one of these URLs (it does -- _record_seen() runs on every clustered
    article, not just the merged representative, before clustering even
    happens). That's what lets get_or_create_micro_summary()'s cache
    write succeed here without this module inserting its own placeholder
    row first, unlike core/classification.py (which runs BEFORE dedup
    and has no such guarantee)."""
    kept_articles: list[dict] = []
    kept_summaries: list[str] = []
    for article in cluster:
        summary = get_or_create_micro_summary(article, conn, map_prompt)
        if summary is None:
            logger.warning(
                "Comparison: dropping '%s' from its cluster (micro-summary failed)",
                article.get("title", ""),
            )
            continue
        kept_articles.append(article)
        kept_summaries.append(summary)
    return kept_articles, kept_summaries


def _validate_bullet(bullet: str, cat_articles: list[dict]) -> str | None:
    """v2.1.3: don't trust the model's comparison bullet just because it
    returned *something*. A live run (Sept 27, 2026) produced a bullet
    whose "[1]" citation tag sat mid-sentence rather than at the end of
    the line -- so core/delivery.py's citation-resolution regex (which
    core/summarizer.py's remap_citation_tags() and this module's
    _TRAILING_CITATION_RE both mirror) never found it -- and, worse, the
    tag cited only ONE of the cluster's outlets even though the cluster
    had qualified as a genuine two-outlet comparison. The rendered
    result fell back to delivery.py's "no citation resolved -> list
    every article" degradation path, which displayed two duplicate-URL
    ANSA entries as if that were "the comparison." Validating here,
    before the bullet is accepted, closes that hole instead of relying
    on a fallback to hide it:
      1. the bullet must end with a well-formed "[n]"/"[n,m,...]" tag
         (delivery.py can't resolve a citation that isn't at the end);
      2. whatever the model actually cited (which may be fewer articles
         than it was given -- that's allowed) must ITSELF still be a
         genuine same-market, multi-outlet, multi-lean comparison, via
         the same core.deduplication.comparison_subset() rule applied
         before this cluster ever reached comparison.py.
    Returns the bullet (stripped) if it passes, else None -- treated
    exactly like a failed Ollama call by the caller: skip this cluster,
    don't render a degenerate "comparison." """
    text = bullet.strip()
    m = _TRAILING_CITATION_RE.search(text)
    if not m:
        return None
    try:
        cited = sorted({int(n.strip()) for n in m.group(1).split(",")})
    except ValueError:
        return None
    if not cited or any(n < 1 or n > len(cat_articles) for n in cited):
        return None
    cited_articles = [cat_articles[n - 1] for n in cited]
    if comparison_subset(cited_articles) is None:
        return None
    return text


def synthesize_comparison_bullet(
    cat_articles: list[dict], cat_summaries: list[str], prompt_template: str
) -> str | None:
    """One reduce-style Ollama call for a single topic cluster. Returns a
    single Markdown bullet line (with a locally-numbered [n] tag, 1-based
    over `cat_articles`), or None if the call failed."""
    numbered = "\n".join(
        f"{i + 1}. [{a.get('source', '')}, orientamento {_lean_label(a.get('lean', 'unknown'))}] {s}"
        for i, (a, s) in enumerate(zip(cat_articles, cat_summaries))
    )
    prompt = prompt_template.format(count=len(cat_articles), summaries=numbered)
    return call_summary_model(prompt, timeout=REDUCE_TIMEOUT)


def build_comparison_section(
    comparison_clusters: list[list[dict]], db_path: Path = HISTORY_DB_PATH
) -> dict | None:
    """Runs the full comparison sub-pipeline over every qualifying cluster
    core.deduplication.dedupe_articles() found. Returns a dict shaped
    exactly like one entry of core.summarizer.summarize_all()'s return
    value -- {"text": ..., "articles": [...], "summaries": [...]} -- ready
    to be inserted into the digest dict under the "Comparativa" key, or
    None if there were no qualifying clusters today, or every one of them
    failed to synthesize (never raises either way)."""
    if not comparison_clusters:
        logger.info("Comparativa: no cross-lean clusters today -- section skipped.")
        return None

    map_prompt = load_prompt("micro_summary.txt")
    comparison_prompt = load_prompt("comparison.txt")

    conn = connect_history_db(db_path)
    ensure_micro_summary_column(conn)

    all_articles: list[dict] = []
    all_summaries: list[str] = []
    bullets: list[str] = []

    try:
        for cluster in comparison_clusters:
            cat_articles, cat_summaries = _summarize_cluster(cluster, conn, map_prompt)
            # v2.1.3: re-check qualification after any micro-summary
            # failures, using the SAME market+source+lean rule
            # core.deduplication.dedupe_articles() applied before this
            # cluster was ever handed to comparison.py -- a cluster that
            # qualified with 3 articles might no longer, if whichever
            # article(s) carried the "other side" of the comparison
            # dropped out here.
            summary_by_id = {id(a): s for a, s in zip(cat_articles, cat_summaries)}
            qualifying = comparison_subset(cat_articles)
            if qualifying is None:
                logger.warning(
                    "Comparativa: cluster around '%s' no longer qualifies (same-market, "
                    "multi-outlet, multi-lean) after micro-summarization -- skipping.",
                    cat_articles[0]["title"] if cat_articles else "?",
                )
                continue
            # Trim down to just the qualifying (possibly same-market-only)
            # subset -- comparison_subset() may drop an off-market or
            # same-outlet-duplicate article that rode along in the cluster.
            cat_articles = qualifying
            cat_summaries = [summary_by_id[id(a)] for a in cat_articles]

            offset = len(all_articles)
            bullet = synthesize_comparison_bullet(cat_articles, cat_summaries, comparison_prompt)
            if bullet is None:
                logger.warning(
                    "Comparativa: synthesis failed for cluster around '%s' -- skipping.",
                    cat_articles[0]["title"],
                )
                continue

            validated = _validate_bullet(bullet, cat_articles)
            if validated is None:
                logger.warning(
                    "Comparativa: model's bullet for cluster around '%s' failed validation "
                    "(missing/malformed trailing citation tag, or what it actually cited isn't "
                    "a genuine multi-outlet/multi-lean comparison) -- dropping instead of "
                    "rendering a degenerate comparison. Raw output: %r",
                    cat_articles[0]["title"], bullet[:200],
                )
                continue

            bullets.append(remap_citation_tags(validated, offset))
            all_articles.extend(cat_articles)
            all_summaries.extend(cat_summaries)

        if not bullets:
            logger.info("Comparativa: all qualifying clusters failed to synthesize -- section skipped.")
            return None

        logger.info(
            "Comparativa: synthesized %d comparison bullet(s) from %d article(s) across %d cluster(s)",
            len(bullets), len(all_articles), len(comparison_clusters),
        )
        return {"text": "\n".join(bullets), "articles": all_articles, "summaries": all_summaries}
    finally:
        conn.close()
