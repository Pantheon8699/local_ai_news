#!/usr/bin/env python3
"""
main.py

Master entry point for the pipeline (see news_pipeline_v2.md, section 8).
Chains the core/ stages in order (see each module's own docstring for
why it exists):

    ingest_all() -> classify_all() -> filter_by_interest()
        -> dedupe_articles() -> apply_votes() -> assign_sections()
        -> summarize_all() -> run_delivery()

The "Comparativa" section (core/comparison.py) is no longer built: each
political item now carries its sources' lean tags instead (see
core/delivery.py's _lean_tags_html).

This is what the systemd service (news_digest.service, section 7) actually
runs every morning. Each stage already has its own "degrade, don't crash"
error handling internally (a bad source, a failed Ollama call, a failed
Cloudflare deploy all log and continue rather than raising) -- main.py's
job is just to wire the stages together, set up one shared log file for
the whole run, and report a clear pass/fail signal to systemd via its
exit code. That exit code is a different, higher-level thing from any
single stage's internal error handling:

    exit 0 -- the digest was rendered AND deployed (run_delivery() returned
              True), or this was a --dry-run that completed normally.
              Cross-day "shown" suppression is now active for today's
              articles (skipped entirely in --dry-run mode).
    exit 1 -- the run completed without crashing, but the digest was NOT
              delivered -- nothing to summarize (e.g. every source down,
              or a very quiet news day), the Cloudflare deploy
              failed/wasn't configured, or the GPU pre-flight check
              (core/gpu_guard.py) found the GPU still busy with something
              else after its full wait budget and skipped the run rather
              than contend with it. Nothing was marked as shown, so
              tomorrow's run will reconsider today's articles. Worth a
              look in logs/run_YYYY-MM-DD.log, but not an emergency --
              the pipeline itself didn't break.
    exit 2 -- something actually crashed. Every core/ function is already
              designed not to raise for the failure modes it knows about
              (a bad feed, Ollama being down, wrangler missing, ...), so
              reaching this means something unexpected happened and is
              worth investigating properly.

Logging: each core/ module defines its own logger via
logging.getLogger(__name__) and, when run standalone (`python3 -m
core.ingestion`, etc. -- still useful for isolated debugging), sets up
its own file handler. Here, main.py configures the root logger once
before calling anything, so every module's logger propagates up to the
same logs/run_YYYY-MM-DD.log for a real end-to-end run -- one log file
per day covering the whole pipeline, not four separate ones.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"

logger = logging.getLogger(__name__)


def _setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"run_{datetime.now():%Y-%m-%d}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    return log_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full morning news digest pipeline once.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run ingestion, dedup, and summarization for real, but skip the Cloudflare deploy and "
            "mark_as_shown(). Use this to inspect output/index.html locally (e.g. after tweaking a "
            "prompt) without touching the live site or cross-day dedup history."
        ),
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help=(
            "Bypass the pre-flight GPU-availability check (core/gpu_guard.py) and start immediately. "
            "Useful for manual/local runs when you don't want to wait; the scheduled systemd run always "
            "keeps the check on."
        ),
    )
    return parser.parse_args(argv)


def run_pipeline(dry_run: bool = False, skip_gpu_check: bool = False) -> int:
    """Runs the full pipeline once and returns the process exit code (see
    module docstring for what each code means), rather than exiting
    directly -- keeps this importable/testable without a real process
    exit."""
    from core.gpu_guard import wait_for_gpu_availability
    from core.ingestion import ingest_all
    from core.classification import classify_all
    from core.interest import apply_votes, assign_sections, filter_by_interest
    from core.deduplication import dedupe_articles
    from core.summarizer import summarize_all
    from core.delivery import load_config, run_delivery

    logger.info("=== Pipeline run starting%s ===", " (--dry-run)" if dry_run else "")

    if skip_gpu_check:
        logger.info("GPU pre-flight check skipped (--skip-gpu-check).")
    elif not wait_for_gpu_availability():
        # Already logged the specifics (utilization/memory/how long it waited) inside
        # wait_for_gpu_availability() -- nothing more to add here.
        return 1

    articles = ingest_all()
    logger.info("Ingestion: %d article(s) total.", len(articles))
    if not articles:
        logger.warning(
            "No articles ingested -- every source may be down, or it's a very quiet news day. "
            "Nothing to classify/deduplicate/summarize/deliver."
        )
        return 1

    classified = classify_all(articles)
    logger.info(
        "Classification: %d/%d article(s) kept (rest discarded as sport/weather/gossip/reviews, "
        "or reclassified into their real topic).", len(classified), len(articles),
    )
    if not classified:
        logger.warning("Nothing left after classification (everything was junk?) -- nothing to deliver.")
        return 1

    classified = filter_by_interest(classified)
    if not classified:
        logger.warning("Nothing left after the interest filter -- nothing to deliver.")
        return 1

    deduped, _ = dedupe_articles(classified)
    logger.info(
        "Deduplication: %d article(s) after same-day clustering + cross-day filtering.", len(deduped),
    )
    if not deduped:
        logger.warning(
            "Nothing left after dedup (everything was already shown recently, or all embeddings "
            "failed) -- nothing to summarize/deliver."
        )
        return 1

    deduped = apply_votes(deduped, load_config()["votes"])
    deduped = assign_sections(deduped)

    digest = summarize_all(deduped)
    total_categories = len({a["category"] for a in deduped})
    logger.info("Summarization: %d/%d categor(y/ies) synthesized.", len(digest), total_categories)

    if not digest:
        logger.error(
            "Summarization produced no sections at all (Ollama likely unreachable, or the model "
            "isn't pulled) -- nothing to deliver."
        )
        return 1

    delivered = run_delivery(digest, deploy=not dry_run)

    if dry_run:
        logger.info(
            "=== Dry run complete: rendered to output/index.html. Cloudflare deploy and "
            "mark_as_shown() were skipped (--dry-run) -- nothing was marked as shown. ==="
        )
        return 0

    if delivered:
        shown_count = sum(len(section["articles"]) for section in digest.values())
        logger.info(
            "=== Pipeline run complete: delivered and deployed. %d article(s) marked as shown. ===",
            shown_count,
        )
        return 0

    logger.error(
        "=== Pipeline run complete, but NOT delivered -- see the delivery-stage log lines above for "
        "why (commonly: wrangler not installed/authenticated, or config.yaml missing "
        "cloudflare.project_name). Digest is still available locally in output/index.html. Nothing "
        "was marked as shown, so today's articles will be reconsidered tomorrow. ==="
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_file = _setup_logging()
    try:
        return run_pipeline(dry_run=args.dry_run, skip_gpu_check=args.skip_gpu_check)
    except Exception:
        logger.exception("Pipeline crashed with an unexpected error -- see traceback above/in %s", log_file)
        return 2


if __name__ == "__main__":
    sys.exit(main())
