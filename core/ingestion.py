"""
core/ingestion.py

Stage 1 of the pipeline (see news_pipeline_v2.md, section 3).

Pulls entries from RSS/Atom feeds via `feedparser`, filters them to a
per-source lookback window (default: today or yesterday), and extracts
full article text via `trafilatura`. No early-exit heuristic: every
entry in every feed is checked individually, because RSS feeds don't
always sort strictly chronologically (corrections/updates can get
bumped to the top), so bailing out after the first N stale-looking
entries risks dropping legitimately fresh stories.

The lookback window is configurable per source via `"lookback_days"` in
sources.json (default 2, i.e. today + yesterday). Wire services (ANSA,
BBC, Adnkronos, ...) publish dozens of times a day, so 2 days is already
generous for "this morning's news." Opinion/analysis outlets (the
Opinion/Geopolitics category) publish far less often -- every few days,
not daily -- so a strict 2-day window left that section empty most
mornings even though nothing was actually broken; those sources use a
longer window (currently 4 days) instead. Note this is a different
concern from cross-day dedup: this module doesn't track what's already
been shown in a previous digest, it only filters by publish date. Not
re-showing an article that was already read is `core/deduplication.py`'s
job (see news_pipeline_v2.md section 4.3, history.db).

One bad source (403, timeout, malformed feed, extraction failure) is
logged and skipped -- it must never take down the whole run.

Fetching is polite by default: every request is checked against the
site's robots.txt (RFC 9309), sent with an honest User-Agent, rate-limited
per domain, and retried with backoff. A source that blocks this reader is
logged and skipped, never worked around.

Browser impersonation (IMPERSONATE below) is possible because curl_cffi,
the HTTP client used here, supports it, but it is OFF and the author does
not recommend turning it on: copying a browser's TLS fingerprint to get
past a site's bot protection may breach its terms of use and, in the EU,
fall outside the text-and-data-mining exception where the publisher has
opted out. See README, "Browser impersonation". Whoever enables it does so
on their own responsibility.

Output: a flat list of article dicts matching the schema in
news_pipeline_v2.md section 4.1 (updated v2.1 -- "Tech" renamed to
"Tech & AI", and a "lean" field added, sourced from sources.json;
updated v2.1.3 -- a "market" field added, see below):

    {
        "category": "Italian National/Regional" | "Italian Economy/Finance"
                     | "US/World" | "Tech & AI" | "Opinion/Geopolitics",
        "source": "ANSA - Politica",
        "lean": "wire" | "left" | "center-left" | "center" | "right"
                | "financial" | "analysis" | "neutral" | "unknown",
        "market": "IT" | "US/UK" | "unknown",
        "title": "...",
        "body": "...",
        "url": "https://...",
        "published_at": "2026-09-06T07:03:00+00:00" | None,
    }

NOTE (v2.1.3): "market" is new -- sourced from sources.json's "market"
field the same way "lean" already is. Added specifically for
core/deduplication.py's Comparativa clustering: "US/World" mixes
Italian-language wire coverage of world events (ANSA - Mondo, market
"IT") with actual US/UK outlets (BBC, Politico, Fox News, ..., market
"US/UK") in the SAME category bucket, since that bucket is organized by
topic ("about the world," not Italy), not by who's covering it. Without
this field, a same-story cluster could pair an ANSA - Mondo piece with a
Fox News piece and call that a "comparison" -- which is not what was
asked for (see news_pipeline_v2.md changelog, v2.1.3): the ask was
Italian-press-vs-Italian-press and US/UK-press-vs-US/UK-press, never
cross-market.

NOTE (v2.1): "category" above is still just the sources.json bucket this
feed lives under at ingestion time -- it does NOT mean the article is
actually about that topic (see _build_article's inline comment). The
pipeline's real classification now happens in core/classification.py,
run between ingestion and dedup.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import feedparser
import trafilatura
from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)

# Matched against robots.txt rules; add a contact URL or e-mail in the
# parentheses so site owners can reach you.
ROBOTS_AGENT = "MorningBrief"
USER_AGENT = f"{ROBOTS_AGENT}/1.0 (personal news reader)"
# OFF on purpose -- read the module docstring and README before changing it.
# Example value: "chrome124" (with a matching Chrome USER_AGENT). Using it is
# the user's own choice and responsibility.
IMPERSONATE = None
REQUEST_TIMEOUT = 15  # seconds, per request
MAX_RETRIES = 3
BACKOFF_BASE = 1.5  # seconds; attempt N waits BACKOFF_BASE ** N
MIN_DOMAIN_DELAY = 1.0  # seconds between consecutive requests to the same domain
DEFAULT_SECTION_LIMIT = 15
DEFAULT_LOOKBACK_DAYS = 2  # today + yesterday; per-source override via sources.json "lookback_days"

SOURCES_PATH = Path(__file__).resolve().parent.parent / "sources.json"


def load_sources(path: Path = SOURCES_PATH) -> dict:
    """Load the {category: [ {source, url, limit, lookback_days?}, ... ]} config."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class PoliteFetcher:
    """A curl_cffi Session wrapper that checks robots.txt and adds a
    User-Agent, a request timeout, retry-with-backoff on failure, and a
    minimum delay between consecutive requests to the same domain."""

    def __init__(self):
        self.session = cffi_requests.Session(impersonate=IMPERSONATE)
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_request_at: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        """robots.txt check, one fetch per site per run. Per RFC 9309: a
        missing robots.txt (4xx) allows everything, an unreachable one (5xx,
        network error) allows nothing."""
        parts = urlparse(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self._robots:
            rules = RobotFileParser()
            self._wait_for_domain(parts.netloc)
            self._last_request_at[parts.netloc] = time.monotonic()
            try:
                resp = self.session.get(f"{root}/robots.txt", timeout=REQUEST_TIMEOUT)
                if resp.status_code >= 500:
                    rules.disallow_all = True
                elif resp.status_code >= 400:
                    rules.allow_all = True
                else:
                    rules.parse(resp.text.splitlines())
            except Exception:
                rules.disallow_all = True
            self._robots[root] = rules
        return self._robots[root].can_fetch(ROBOTS_AGENT, url)

    def _wait_for_domain(self, domain: str) -> None:
        last = self._last_request_at.get(domain)
        if last is not None:
            elapsed = time.monotonic() - last
            if elapsed < MIN_DOMAIN_DELAY:
                time.sleep(MIN_DOMAIN_DELAY - elapsed)

    def get(self, url: str):
        if not self.allowed(url):
            logger.warning("Not fetching %s: disallowed by robots.txt", url)
            return None
        domain = urlparse(url).netloc
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            self._wait_for_domain(domain)
            self._last_request_at[domain] = time.monotonic()
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp
            except Exception as exc:
                # curl_cffi raises its own exception hierarchy (CurlError
                # and subclasses), not requests.RequestException, so this
                # catches broadly -- any failure here just means "retry or
                # give up on this one source," never a crash.
                last_exc = exc
                logger.warning(
                    "Fetch failed (attempt %d/%d) for %s: %s",
                    attempt, MAX_RETRIES, url, exc,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_BASE ** attempt)
        logger.error("Giving up on %s after %d attempts (%s)", url, MAX_RETRIES, last_exc)
        return None


def _entry_published_at(entry) -> datetime | None:
    """Best-effort timezone-aware publish datetime from a feedparser entry."""
    for key in ("published_parsed", "updated_parsed"):
        struct = getattr(entry, key, None)
        if struct:
            return datetime.fromtimestamp(time.mktime(struct), tz=timezone.utc)
    return None


def _within_lookback(dt: datetime | None, today: date, lookback_days: int) -> bool:
    """True if dt falls within [today - (lookback_days - 1), today], in the
    local timezone. lookback_days=2 (the default) reproduces the original
    "today or yesterday" rule; a source can widen this via
    sources.json's "lookback_days" for content that's still relevant
    days later (see module docstring)."""
    if dt is None:
        return False
    local_date = dt.astimezone().date()
    age_days = (today - local_date).days
    return 0 <= age_days < lookback_days


def _select_candidates(entries, limit: int, today: date, lookback_days: int) -> list[tuple]:
    """Check every entry (no early exit) against the date filter, then cap
    to the configured per-section limit, keeping the newest first."""
    candidates = []
    for entry in entries:
        link = getattr(entry, "link", None)
        title = getattr(entry, "title", "").strip()
        if not link or not title:
            continue
        published_at = _entry_published_at(entry)
        if not _within_lookback(published_at, today, lookback_days):
            continue
        candidates.append((published_at, title, link))
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[:limit]


def _run_trafilatura(html: str | None, url: str) -> dict | None:
    """Run trafilatura extraction on already-fetched HTML. Returns the
    parsed trafilatura JSON dict (text, date, title, ...) or None on any
    failure -- never raises."""
    if not html:
        return None
    try:
        raw_json = trafilatura.extract(
            html,
            url=url,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    except Exception as exc:  # trafilatura can raise on malformed markup
        logger.warning("trafilatura extraction raised for %s: %s", url, exc)
        return None
    if not raw_json:
        return None
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("trafilatura returned invalid JSON for %s", url)
        return None


def _build_article(
    category: str,
    source_name: str,
    title: str,
    link: str,
    published_at: datetime | None,
    extracted: dict | None,
    lean: str = "unknown",
    market: str = "unknown",
) -> dict:
    body = (extracted.get("text") or "").strip() if extracted else ""
    final_published_at = published_at
    if extracted and final_published_at is None and extracted.get("date"):
        try:
            final_published_at = datetime.fromisoformat(extracted["date"])
        except ValueError:
            pass
    if not body:
        logger.warning("No body extracted for '%s' (%s) -- keeping title/link only", title, link)
    return {
        # "category" here is the SOURCE-BUCKET category from sources.json --
        # i.e. which feed this came from, not necessarily what the article
        # is actually about (ANSA's "Mondo" feed, or a general-purpose feed
        # like Il Fatto Quotidiano's, can carry any topic). core/classification.py
        # (new, v2.1) re-derives the real category from content right after
        # ingestion and overwrites this field before dedup ever partitions
        # by it -- see that module's docstring for why this bucket value
        # can't be trusted on its own (news_pipeline_v2.md changelog, v2.1).
        "category": category,
        "source": source_name,
        "lean": lean,  # political/editorial lean from sources.json, e.g.
                       # "left"/"center-left"/"center"/"center-right"/"right"/
                       # "wire"/"financial"/"analysis"/"neutral"/"unknown" --
                       # used by core/comparison.py (new, v2.1) to find
                       # same-story clusters spanning multiple leans.
        "market": market,  # "IT" | "US/UK" | "unknown" (v2.1.3) -- which
                           # press market this source belongs to, so
                           # core/deduplication.py's Comparativa clustering
                           # never pairs an Italian outlet against a US/UK
                           # one (see module docstring above).
        "title": title,
        "body": body,
        "url": link,
        "published_at": final_published_at.isoformat() if final_published_at else None,
    }


def _extract_article(fetcher: PoliteFetcher, url: str) -> dict | None:
    resp = fetcher.get(url)
    if resp is None:
        return None
    return _run_trafilatura(resp.text, url)


def ingest_feed(
    category: str,
    source_cfg: dict,
    fetcher: PoliteFetcher,
    today: date,
) -> list[dict]:
    """Ingest one feed from sources.json. Never raises -- logs and returns
    [] on failure so one bad source doesn't kill the run."""
    source_name = source_cfg["source"]
    feed_url = source_cfg["url"]
    limit = source_cfg.get("limit", DEFAULT_SECTION_LIMIT)
    lookback_days = source_cfg.get("lookback_days", DEFAULT_LOOKBACK_DAYS)
    lean = source_cfg.get("lean", "unknown")
    market = source_cfg.get("market", "unknown")

    try:
        resp = fetcher.get(feed_url)
        if resp is None:
            logger.error("Skipping %s (%s): feed unreachable", source_name, feed_url)
            return []
        parsed = feedparser.parse(resp.content)
        if parsed.bozo and not parsed.entries:
            logger.error(
                "Skipping %s (%s): feed did not parse (%s)",
                source_name, feed_url, getattr(parsed, "bozo_exception", "unknown error"),
            )
            return []
    except Exception as exc:
        logger.error("Skipping %s (%s): %s", source_name, feed_url, exc)
        return []

    candidates = _select_candidates(parsed.entries, limit, today, lookback_days)

    articles = []
    for published_at, title, link in candidates:
        extracted = _extract_article(fetcher, link)
        articles.append(_build_article(category, source_name, title, link, published_at, extracted, lean, market))

    logger.info("%s: %d article(s) kept after date filter + extraction", source_name, len(articles))
    return articles


def ingest_all(sources_path: Path = SOURCES_PATH) -> list[dict]:
    """Run ingestion across every source in sources.json. Returns a flat
    list of article dicts. A failure in one feed is logged and skipped."""
    sources = load_sources(sources_path)
    fetcher = PoliteFetcher()
    today = datetime.now().astimezone().date()

    all_articles: list[dict] = []
    for category, feed_list in sources.items():
        for source_cfg in feed_list:
            try:
                all_articles.extend(ingest_feed(category, source_cfg, fetcher, today))
            except Exception as exc:
                # Belt-and-braces: ingest_feed already catches internally,
                # but nothing here may ever take down the whole run.
                logger.error("Unexpected error ingesting %s: %s", source_cfg.get("source"), exc)
    return all_articles


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
    _setup_logging()

    articles = ingest_all()

    print(f"\n=== Ingested {len(articles)} article(s) total ===")
    by_category: dict[str, list[dict]] = {}
    for a in articles:
        by_category.setdefault(a["category"], []).append(a)

    for cat, items in by_category.items():
        print(f"\n-- {cat} ({len(items)}) --")
        for a in items[:3]:
            body_len = len(a["body"])
            flag = "" if body_len > 0 else "  [!] EMPTY BODY"
            print(f"  [{a['source']}] {a['title']}")
            print(f"    published_at={a['published_at']}  body_chars={body_len}{flag}")
