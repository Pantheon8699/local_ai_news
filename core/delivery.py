"""
core/delivery.py

Stage 4 of the pipeline (see news_pipeline_v2.md, section 6): renders the
day's digest (as produced by core.summarizer.summarize_all()) into a
static HTML page, writes it to output/, records it in history.db's
`digests` table for the archive nav, and deploys it to Cloudflare Pages
via the `wrangler` CLI. Also installable as a PWA (templates/manifest.json
+ templates/icons/) for a home-screen icon on the phone.

No templating library is added as a dependency for this -- the page is a
handful of placeholders in templates/digest.html filled in with stdlib
string.Template, which is all a single-page "date + N bullet-list
sections + an archive nav strip" layout needs. The reduce-stage prompt
(prompts/executive_editor.txt) is constrained to plain Markdown bullet
lists with only ** bold ** as inline formatting, so the Markdown->HTML
step here is a small hand-rolled converter (_render_bullets_html), not a
full Markdown library.

Per-bullet source expansion (news_pipeline_v2.md 5.5/6.7): the compact
synthesized bullet is often the whole point of the digest, but it can
merge several articles into one dense line and drops attribution
entirely. Since prompts/executive_editor.txt now instructs the model to
tag each bullet with the numbers of the micro-summaries ("riepiloghi")
it drew on -- e.g. "...al 44% dei voti. [2,5]" -- and
core.summarizer.summarize_all() returns those micro-summaries
index-aligned with the article list (digest[category]["summaries"][i]
<-> digest[category]["articles"][i], 1-based in the tag), this module
can resolve each tag back to the specific articles behind that bullet
and render them as a native <details>/<summary> expansion: no JS
needed, works offline in the static output, and degrades gracefully on
a phone. Each expanded entry shows the article's title, a real excerpt
of the original article text (see _excerpt_body/DETAIL_EXCERPT_CHARS --
this is NOT the 1-2 sentence micro-summary; that reads as barely more
than the bullet it fed, which is exactly what this feature was built to
get past), and a link to the source for the rest. This needed no
history.db schema change: the article dicts flowing from ingestion ->
dedup -> summarize -> delivery all stay in memory for the one process
that runs main.py, and none of those stages strip the "body" field --
delivery.py is simply the first stage to actually read it. The tradeoff
is that this excerpt is only ever as good as trafilatura's extraction
for that article (occasional boilerplate/leftover fragments are
possible); a missing/empty body falls back to the micro-summary rather
than showing nothing.

Because a 7B model won't hit the "[n]" tag on every bullet 100% of the
time, this degrades in two layers rather than silently dropping the
feature: (1) a bullet whose tag is missing, malformed, or references an
out-of-range number just renders as a plain <li>, exactly as before
this feature existed; (2) if an entire category comes back with *no*
resolvable tags at all (e.g. an older cached digest predating this
change, or a model that ignored the instruction across the board), the
whole section still gets one collapsed "Fonti di questa sezione"
block listing every article that fed it, so there's always a way to
see sources -- never worse than the old no-sources rendering, often
better.

Cross-day "shown" gating (the actual design decision worth flagging):
mark_as_shown() (core/deduplication.py) is what permanently suppresses an
article from ever showing up again (see news_pipeline_v2.md 4.3). This
module only calls it if the Cloudflare deploy actually succeeds --
rendering a local HTML file is not the same as the user being able to
open it on their phone away from home, which is the entire premise of
this delivery stage (6.1). A digest that only exists locally because
wrangler isn't installed/authenticated yet, or config.yaml has no
Cloudflare project name configured, must NOT be marked as shown -- it
would otherwise be silently and permanently lost the next time dedup
runs. See run_delivery()'s docstring for the exact rule and how to
opt out for a local-only dry run.

Cloudflare setup this module assumes but does not itself perform (per
news_pipeline_v2.md 6.3): the `wrangler` CLI installed (`npm install -g
wrangler`) and authenticated -- either interactively once via
`wrangler login`, or non-interactively by setting `cloudflare.api_token`
in config.yaml (exported as CLOUDFLARE_API_TOKEN for the subprocess
call). Neither has been verified to be set up on this machine yet; if
they aren't, deploy_to_cloudflare() logs a clear, actionable error and
degrades (skips deploy, does not crash the run) rather than assuming
credentials exist.

Failure handling: matches every other stage -- a missing config.yaml, a
missing wrangler binary, a failed deploy, or a missing digests-table row
are all logged and degrade (skip that piece) rather than crashing. The
one exception is deliberate, not a bug: an empty digest short-circuits
the whole function, since there is nothing to render.

Debug Markdown mirror (v2.1, added on request): run_delivery() also
writes a plain-.md copy of the same digest content to debug/ --
render_digest_markdown() + write_debug_markdown() below. This exists
purely so a --dry-run (or any run) can be inspected/diffed/grepped in a
text editor without spinning up a local server or opening
output/index.html in a browser. It is a flat mirror, not a re-derivation:
same category order, same bullets, same citation-tag resolution
(_resolve_citation/_CITATION_RE, shared with the HTML path) as
render_digest_html(), just rendered as Markdown with nested sub-bullets
for cited sources instead of a collapsed <details>. debug/ is a sibling
of output/ (not a subdirectory of it) specifically because
deploy_to_cloudflare() runs `wrangler pages deploy ./output`, which only
uploads whatever is under output/ -- putting debug/ anywhere under
output/ would risk it getting published by accident. Written
unconditionally (before the deploy=False / deploy-failure early returns)
so it's produced even on a dry run or a failed deploy, which is
precisely when it's most useful for debugging. Degrades like everything
else here: a write failure is logged and does not fail the run.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from datetime import date, datetime
from html import escape as _html_escape
from pathlib import Path
from string import Template

import yaml

from core.deduplication import HISTORY_DB_PATH, connect_history_db, mark_as_shown
from core.interest import POLITICS_SECTIONS, SECTIONS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
DEBUG_DIR = PROJECT_ROOT / "debug"  # sibling of output/, deliberately NOT under it --
# see this module's docstring ("Debug Markdown mirror") for why: `wrangler pages
# deploy ./output` must never see this directory.

WRANGLER_TIMEOUT = 180  # seconds -- a static-site upload, shouldn't need long

DEFAULT_CATEGORY_ORDER = list(SECTIONS)

# Political lean of the outlet (sources.json "lean"), shown as a small tag
# on items in POLITICS_SECTIONS. Wire/centre/analysis outlets get no tag.
_LEAN_TAGS = {"left": "L", "center-left": "CL", "center-right": "CR", "right": "R"}

DEFAULT_CONFIG = {
    "cloudflare": {"project_name": None, "api_token": None},
    "category_order": DEFAULT_CATEGORY_ORDER,
}

_ITALIAN_WEEKDAYS = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
_ITALIAN_MONTHS = [
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
]

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

# Matches a trailing "[n]" / "[n,m,...]" source-citation tag the reduce
# prompt (prompts/executive_editor.txt) is instructed to append to every
# bullet, e.g. "...al 44% dei voti. [2,5]". Captures the digit list;
# consumed and stripped from the visible bullet text before rendering.
_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]\s*$")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Reads config.yaml. Missing file or missing keys degrade to
    DEFAULT_CONFIG (no Cloudflare project configured -- deploy will be
    skipped with a clear log message, not a crash)."""
    if not path.exists():
        logger.warning(
            "config.yaml not found at %s -- using defaults. Deploy will be skipped until "
            "cloudflare.project_name is set there.", path,
        )
        return {"cloudflare": dict(DEFAULT_CONFIG["cloudflare"]), "category_order": list(DEFAULT_CATEGORY_ORDER), "votes": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as exc:
        logger.error("Could not read/parse config.yaml (%s) -- using defaults.", exc)
        loaded = {}

    cloudflare = {**DEFAULT_CONFIG["cloudflare"], **(loaded.get("cloudflare") or {})}
    category_order = loaded.get("category_order") or list(DEFAULT_CATEGORY_ORDER)
    return {"cloudflare": cloudflare, "category_order": category_order, "votes": loaded.get("votes") or {}}


def _format_date_italian(d: date) -> str:
    """Hand-rolled instead of strftime('%A %d %B') -- that relies on the
    it_IT locale being installed on this machine, which we can't assume
    (news_pipeline_v2.md section 2: avoid OS-specific assumptions)."""
    return f"{_ITALIAN_WEEKDAYS[d.weekday()]} {d.day} {_ITALIAN_MONTHS[d.month - 1]} {d.year}"


def _inline_markdown_to_html(text: str) -> str:
    """HTML-escape first (article/LLM text is not trusted input), then
    convert the one inline markup the executive-editor prompt is allowed
    to use (**bold**)."""
    escaped = _html_escape(text)
    return _BOLD_RE.sub(r"<strong>\1</strong>", escaped)


def _source_name(article: dict) -> str:
    """Same "merged sources" convention used elsewhere (e.g.
    core.summarizer.micro_summarize): an article coming out of dedup
    clustering may carry a "sources" list (multiple outlets covering the
    same story); fall back to the single "source" field otherwise."""
    return ", ".join(article.get("sources") or [article.get("source", "")])


DETAIL_EXCERPT_CHARS = 700  # long enough to read as a real paragraph (~100-130 words),
# short of the full article -- the source link is still what gets the reader to the
# rest. Deliberately longer than core.summarizer.MAX_BODY_CHARS's use case (that one
# feeds an LLM prompt and can afford to be big; this one is read by a human on a phone).


def _excerpt_body(body: str | None, max_chars: int = DETAIL_EXCERPT_CHARS) -> str:
    """Trims the article's extracted body text (still present on the article
    dict at this point in the pipeline -- see this module's docstring) to a
    word-boundary-safe excerpt for the expanded source view. Returns "" for a
    missing/empty body so the caller can fall back to the micro-summary."""
    body = (body or "").strip()
    if not body:
        return ""
    if len(body) <= max_chars:
        return body
    return body[:max_chars].rsplit(" ", 1)[0].rstrip(",;:—-") + "…"


def _lean_tags_html(articles: list[dict], category: str) -> str:
    """L/CL/CR/R tags for the outlets behind these articles, only in the
    politics sections. A merged article carries every clustered outlet's
    lean in "leans" (core.deduplication._merge_cluster)."""
    if category not in POLITICS_SECTIONS:
        return ""
    leans = {lean for a in articles for lean in (a.get("leans") or [a.get("lean")])}
    return "".join(f' <span class="lean">{tag}</span>' for lean, tag in _LEAN_TAGS.items() if lean in leans)


def _render_source_item_html(article: dict, summary: str, category: str = "") -> str:
    """One expandable "card" for a single article backing a bullet (or a
    category-level fallback list): title, a real excerpt of the original
    article text, and a link back to the source for the rest. Falls back to
    the 1-2 sentence micro-summary only if no body text is available (e.g.
    extraction failed for that article at ingestion time) -- the
    micro-summary alone reads as barely more than the compact bullet it was
    built from, so it's the fallback here, not the primary detail text."""
    title = _html_escape(article.get("title") or "")
    source = _html_escape(_source_name(article))
    detail_text = _excerpt_body(article.get("body")) or (summary or "")
    detail = _html_escape(detail_text)
    url = _html_escape(article.get("url") or "", quote=True)
    return (
        '<div class="source-item">'
        f'<div class="source-title">{title}{_lean_tags_html([article], category)}</div>'
        f'<p class="source-detail">{detail}</p>'
        f'<a class="source-link" href="{url}" target="_blank" rel="noopener noreferrer">{source} ↗</a>'
        "</div>"
    )


def _resolve_citation(
    tag_numbers: str, cat_articles: list[dict], cat_summaries: list[str]
) -> list[tuple[dict, str]]:
    """Parses a "[n]" / "[n,m,...]" tag's digit list into the (article,
    micro_summary) pairs it refers to (1-based, matching the numbering
    prompts/executive_editor.txt was given). Out-of-range or duplicate
    numbers are dropped rather than raising -- this is parsing
    free-form LLM output, not a trusted format. Returns [] if nothing in
    the tag resolves to a valid article, which the caller treats the
    same as "no tag at all" (plain bullet, no expansion)."""
    seen_idx: set[int] = set()
    resolved: list[tuple[dict, str]] = []
    for raw_n in tag_numbers.split(","):
        raw_n = raw_n.strip()
        if not raw_n.isdigit():
            continue
        idx = int(raw_n) - 1
        if idx in seen_idx or not (0 <= idx < len(cat_articles)):
            continue
        seen_idx.add(idx)
        resolved.append((cat_articles[idx], cat_summaries[idx] if idx < len(cat_summaries) else ""))
    return resolved


def _render_bullets_html(
    markdown_text: str, cat_articles: list[dict], cat_summaries: list[str], category: str = ""
) -> tuple[str, bool]:
    """Converts a Markdown bullet list (the only format
    prompts/executive_editor.txt is instructed to produce) into <li>
    elements. Tolerates a stray numbered line or a line missing its
    bullet marker rather than dropping it, since this is rendering LLM
    output -- format drift happens.

    Each bullet may end with a "[n]" / "[n,m,...]" source-citation tag
    (see prompts/executive_editor.txt); when it resolves to one or more
    articles, the tag is stripped from the visible text and the bullet
    becomes a <details>/<summary> that expands to show the cited
    articles' micro-summaries and source links (see this module's
    docstring). A missing/malformed/out-of-range tag just renders a
    plain bullet, same as before this feature existed.

    Returns (html, any_citation_resolved) -- the second value tells
    _build_sections_html() whether this category needs the
    whole-section "Fonti di questa sezione" fallback."""
    items = []
    any_resolved = False
    for raw_line in (markdown_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line[:2] in ("- ", "* "):
            line = line[2:].strip()
        else:
            m = re.match(r"^\d+[.)]\s+(.*)$", line)
            if m:
                line = m.group(1).strip()
        if not line:
            continue

        cited = []
        tag_match = _CITATION_RE.search(line)
        if tag_match:
            cited = _resolve_citation(tag_match.group(1), cat_articles, cat_summaries)
            line = line[: tag_match.start()].rstrip()
        if not line:
            continue

        bullet_html = _inline_markdown_to_html(line)
        if cited:
            any_resolved = True
            sources_html = "".join(_render_source_item_html(a, s, category) for a, s in cited)
            bullet_html += _lean_tags_html([a for a, _ in cited], category)
            # data-urls + the two buttons feed the like/dislike script in
            # templates/digest.html (see core/interest.py); a vote on a
            # bullet counts for every article it cites.
            urls = _html_escape(json.dumps([a.get("url") or "" for a, _ in cited]), quote=True)
            items.append(
                f'<li data-urls="{urls}"><details><summary>'
                f"{bullet_html}</summary>"
                f'<div class="sources">{sources_html}</div>'
                "</details>"
                '<span class="vote"><button data-v="1">👍</button><button data-v="-1">👎</button></span></li>'
            )
        else:
            items.append(f"<li>{bullet_html}</li>")
    return "\n".join(items), any_resolved


def _render_category_sources_fallback_html(cat_articles: list[dict], cat_summaries: list[str]) -> str:
    """Whole-section "Fonti di questa sezione" block, collapsed by
    default, used only when none of a category's bullets carried a
    resolvable citation tag (see this module's docstring)."""
    if not cat_articles:
        return ""
    items = "".join(
        _render_source_item_html(a, cat_summaries[i] if i < len(cat_summaries) else "")
        for i, a in enumerate(cat_articles)
    )
    label = f"Fonti di questa sezione ({len(cat_articles)} articol{'o' if len(cat_articles) == 1 else 'i'})"
    return (
        '<details class="category-sources">'
        f"<summary>{_html_escape(label)}</summary>"
        f'<div class="sources">{items}</div>'
        "</details>"
    )


def _build_sections_html(digest: dict, category_order: list[str]) -> str:
    ordered = [c for c in category_order if c in digest]
    ordered += [c for c in digest if c not in ordered]
    parts = []
    for category in ordered:
        section = digest[category]
        cat_articles = section.get("articles") or []
        cat_summaries = section.get("summaries") or []
        bullets, any_resolved = _render_bullets_html(section["text"], cat_articles, cat_summaries, category)
        if not bullets:
            continue
        fallback = "" if any_resolved else _render_category_sources_fallback_html(cat_articles, cat_summaries)
        parts.append(
            f'<section class="category">\n<h2>{_html_escape(category)}</h2>\n'
            f"<ul>\n{bullets}\n</ul>\n{fallback}\n</section>"
        )
    return "\n".join(parts)


def _build_missing_note_html(missing: list[str]) -> str:
    if not missing:
        return ""
    names = ", ".join(_html_escape(c) for c in missing)
    return f'<div class="missing-note">Nessun contenuto disponibile oggi per: {names}.</div>'


def _build_archive_nav_html(archive_dates: list[str], current_date_iso: str) -> str:
    if len(archive_dates) <= 1:
        return ""
    links = []
    for d in archive_dates:
        label = d[5:]  # "MM-DD" -- compact enough for a horizontal strip
        cls = ' class="current"' if d == current_date_iso else ""
        links.append(f'<a{cls} href="/{d}/">{label}</a>')
    return '<nav class="archive">\n' + "\n".join(links) + "\n</nav>"


def render_digest_html(
    digest: dict,
    date_obj: date,
    archive_dates: list[str],
    missing_categories: list[str],
    category_order: list[str],
) -> str:
    """Fills templates/digest.html via stdlib string.Template (safe_substitute,
    so a stray '$' anywhere in scraped article text can't blow up formatting
    the way str.format's '{' would)."""
    template_text = (TEMPLATES_DIR / "digest.html").read_text(encoding="utf-8")
    template = Template(template_text)
    return template.safe_substitute(
        date_human=_format_date_italian(date_obj),
        archive_nav_html=_build_archive_nav_html(archive_dates, date_obj.isoformat()),
        sections_html=_build_sections_html(digest, category_order),
        missing_categories_html=_build_missing_note_html(missing_categories),
        generated_at=datetime.now().strftime("%d/%m/%Y %H:%M"),
    )


def _render_bullets_markdown(
    markdown_text: str, cat_articles: list[dict], cat_summaries: list[str]
) -> tuple[str, bool]:
    """Debug-only Markdown twin of _render_bullets_html(): identical
    parsing (bullet-marker stripping, "[n]" citation-tag resolution via
    the shared _CITATION_RE/_resolve_citation), but rendered as plain
    Markdown -- an indented sub-bullet per cited source instead of a
    collapsed <details>/<summary>, since this file is meant to be read
    top-to-bottom in a text editor, not clicked through in a browser. No
    HTML-escaping and the original **bold** is left as-is (it's already
    the right syntax for a .md file). Returns (markdown, any_citation_resolved)
    -- same convention as the HTML version, for the same reason (deciding
    whether a category needs the whole-section fallback)."""
    lines: list[str] = []
    any_resolved = False
    for raw_line in (markdown_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line[:2] in ("- ", "* "):
            line = line[2:].strip()
        else:
            m = re.match(r"^\d+[.)]\s+(.*)$", line)
            if m:
                line = m.group(1).strip()
        if not line:
            continue

        cited = []
        tag_match = _CITATION_RE.search(line)
        if tag_match:
            cited = _resolve_citation(tag_match.group(1), cat_articles, cat_summaries)
            line = line[: tag_match.start()].rstrip()
        if not line:
            continue

        lines.append(f"- {line}")
        if cited:
            any_resolved = True
            for article, summary in cited:
                title = article.get("title") or "(senza titolo)"
                source = _source_name(article)
                url = article.get("url") or ""
                detail = _excerpt_body(article.get("body")) or (summary or "")
                lines.append(f"    - **{source}** — {title}")
                if detail:
                    lines.append(f"      {detail}")
                if url:
                    lines.append(f"      {url}")
    return "\n".join(lines), any_resolved


def _render_category_sources_fallback_markdown(cat_articles: list[dict], cat_summaries: list[str]) -> str:
    """Debug-only Markdown twin of _render_category_sources_fallback_html()
    -- used under the same condition (no bullet in the category carried a
    resolvable citation tag), listing every article that fed the section."""
    if not cat_articles:
        return ""
    lines = [f"_Fonti di questa sezione ({len(cat_articles)} articol{'o' if len(cat_articles) == 1 else 'i'}):_"]
    for i, article in enumerate(cat_articles):
        summary = cat_summaries[i] if i < len(cat_summaries) else ""
        title = article.get("title") or "(senza titolo)"
        source = _source_name(article)
        url = article.get("url") or ""
        detail = _excerpt_body(article.get("body")) or summary
        lines.append(f"- **{source}** — {title}")
        if detail:
            lines.append(f"  {detail}")
        if url:
            lines.append(f"  {url}")
    return "\n".join(lines)


def _build_sections_markdown(digest: dict, category_order: list[str]) -> str:
    """Debug-only Markdown twin of _build_sections_html() -- same category
    ordering rule (category_order first, anything else appended after)."""
    ordered = [c for c in category_order if c in digest]
    ordered += [c for c in digest if c not in ordered]
    parts = []
    for category in ordered:
        section = digest[category]
        cat_articles = section.get("articles") or []
        cat_summaries = section.get("summaries") or []
        bullets, any_resolved = _render_bullets_markdown(section["text"], cat_articles, cat_summaries)
        if not bullets:
            continue
        block = f"## {category}\n\n{bullets}"
        if not any_resolved:
            fallback = _render_category_sources_fallback_markdown(cat_articles, cat_summaries)
            if fallback:
                block += f"\n\n{fallback}"
        parts.append(block)
    return "\n\n".join(parts)


def render_digest_markdown(
    digest: dict,
    date_obj: date,
    missing_categories: list[str],
    category_order: list[str],
) -> str:
    """Debug-only plain-Markdown mirror of render_digest_html() -- same
    content, same category order, same resolved citations/sources -- as a
    flat .md file instead of the styled HTML page (see this module's
    top-level docstring, "Debug Markdown mirror", for why this exists and
    where it's written). Never touches templates/digest.html; this is a
    separate, simpler render path built for reading raw pipeline output,
    not for the deployed site."""
    lines = [
        f"# Morning Brief — {_format_date_italian(date_obj)}",
        "",
        f"_Generato localmente il {datetime.now().strftime('%d/%m/%Y %H:%M')}_",
    ]
    sections = _build_sections_markdown(digest, category_order)
    if sections:
        lines.append("")
        lines.append(sections)
    else:
        lines.append("")
        lines.append("_(nessuna sezione prodotta)_")
    if missing_categories:
        names = ", ".join(missing_categories)
        lines.append("")
        lines.append(f"> Nessun contenuto disponibile oggi per: {names}.")
    return "\n".join(lines)


def write_debug_markdown(markdown_text: str, date_obj: date, debug_dir: Path = DEBUG_DIR) -> Path | None:
    """Writes debug/YYYY-MM-DD.md (archive copy) and debug/latest.md
    ("today's"/"most recent run's" copy) -- mirrors write_output()'s
    two-file convention for output/, but into DEBUG_DIR, which is a
    sibling of output/ and never touched by `wrangler pages deploy
    ./output` (see this module's top-level docstring). Returns the dated
    file's path, or None if the write failed (logged, not raised -- a
    debug convenience file is never worth failing an otherwise-successful
    run over)."""
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        dated_path = debug_dir / f"{date_obj.isoformat()}.md"
        dated_path.write_text(markdown_text, encoding="utf-8")
        (debug_dir / "latest.md").write_text(markdown_text, encoding="utf-8")
        return dated_path
    except OSError as exc:
        logger.warning("Could not write debug Markdown mirror (%s) -- continuing.", exc)
        return None


def _copy_static_assets(output_dir: Path) -> None:
    """manifest.json and the icons live at the site root (not under a
    dated subfolder) since every page links them with an absolute path
    ('/manifest.json') -- that resolves correctly regardless of which
    day's page you're viewing."""
    try:
        shutil.copy(TEMPLATES_DIR / "manifest.json", output_dir / "manifest.json")
        icons_dst = output_dir / "icons"
        icons_dst.mkdir(exist_ok=True)
        for icon in (TEMPLATES_DIR / "icons").glob("*.png"):
            shutil.copy(icon, icons_dst / icon.name)
    except OSError as exc:
        logger.warning("Could not copy PWA assets (manifest/icons) into output/: %s", exc)


def write_output(html: str, date_obj: date, output_dir: Path = OUTPUT_DIR) -> Path:
    """Writes output/YYYY-MM-DD/index.html (archive copy) and
    output/index.html (today's copy, i.e. "the latest") -- both identical
    content, per news_pipeline_v2.md 6.2. Returns the dated file's path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dated_dir = output_dir / date_obj.isoformat()
    dated_dir.mkdir(exist_ok=True)
    dated_index = dated_dir / "index.html"
    dated_index.write_text(html, encoding="utf-8")
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    _copy_static_assets(output_dir)
    return dated_index


def _ensure_digests_table(conn) -> None:
    """delivery.py's own table in the shared history.db (see
    news_pipeline_v2.md 4.3 for the ownership split with
    core/deduplication.py's `articles` table)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS digests (
            date TEXT PRIMARY KEY,
            categories TEXT NOT NULL,
            missing_categories TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _record_digest(conn, date_iso: str, categories: list[str], missing: list[str]) -> None:
    conn.execute(
        """
        INSERT INTO digests (date, categories, missing_categories, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            categories = excluded.categories,
            missing_categories = excluded.missing_categories,
            created_at = excluded.created_at
        """,
        (
            date_iso,
            json.dumps(categories, ensure_ascii=False),
            json.dumps(missing, ensure_ascii=False),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def _get_archive_dates(conn, limit: int = 14) -> list[str]:
    rows = conn.execute("SELECT date FROM digests ORDER BY date DESC LIMIT ?", (limit,)).fetchall()
    return [r[0] for r in rows][::-1]  # oldest -> newest, so "today" lands rightmost in the nav strip


def deploy_to_cloudflare(output_dir: Path, config: dict) -> bool:
    """Shells out to `wrangler pages deploy`. Returns False (never raises)
    on any failure: wrangler not installed, no project_name configured,
    a timeout, or a non-zero exit (commonly: not logged in, or the named
    Cloudflare Pages project doesn't exist yet -- `wrangler pages project
    create <name>` creates it once, outside this pipeline)."""
    project_name = (config.get("cloudflare") or {}).get("project_name")
    if not project_name:
        logger.error(
            "No cloudflare.project_name set in config.yaml -- skipping deploy. "
            "Digest is still available locally at %s.", output_dir / "index.html",
        )
        return False

    import os
    env = os.environ.copy()
    api_token = (config.get("cloudflare") or {}).get("api_token")
    if api_token:
        env["CLOUDFLARE_API_TOKEN"] = api_token

    cmd = ["wrangler", "pages", "deploy", str(output_dir), "--project-name", project_name]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=WRANGLER_TIMEOUT, env=env, cwd=PROJECT_ROOT)
    except FileNotFoundError:
        logger.error(
            "wrangler CLI not found -- install it with 'npm install -g wrangler', then either run "
            "'wrangler login' once (interactive OAuth) or set cloudflare.api_token in config.yaml. "
            "Digest is still available locally at %s.", output_dir / "index.html",
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("wrangler pages deploy timed out after %ds.", WRANGLER_TIMEOUT)
        return False

    if result.returncode != 0:
        logger.error(
            "wrangler pages deploy failed (exit %d): %s",
            result.returncode, (result.stderr or result.stdout).strip(),
        )
        return False

    logger.info("Deployed to Cloudflare Pages project '%s'.", project_name)
    if result.stdout:
        logger.debug(result.stdout.strip())
    return True


def run_delivery(
    digest: dict,
    db_path: Path = HISTORY_DB_PATH,
    output_dir: Path = OUTPUT_DIR,
    config_path: Path = CONFIG_PATH,
    deploy: bool = True,
    debug_dir: Path = DEBUG_DIR,
    write_debug_md: bool = True,
) -> bool:
    """Render -> write locally -> write debug Markdown mirror -> record in
    history.db's `digests` table -> (if deploy=True) deploy to Cloudflare
    Pages -> (only if that deploy succeeds) mark_as_shown() on every
    article that made it into the digest.

    That gate on mark_as_shown() is deliberate, not an oversight:
    mark_as_shown() permanently suppresses an article from ever appearing
    again (see core/deduplication.py, news_pipeline_v2.md 4.3). A digest
    that exists only as a local file was never actually reachable from the
    phone -- the entire premise of this delivery stage -- so nothing
    should be marked shown until a real deploy lands. Call with
    deploy=False for a local-only dry run (e.g. eyeballing
    output/index.html in a browser before Cloudflare is set up); that
    intentionally skips both the deploy attempt and mark_as_shown for the
    same reason.

    The debug Markdown mirror (debug/YYYY-MM-DD.md + debug/latest.md --
    see render_digest_markdown()/write_debug_markdown() and this module's
    top-level docstring) is written unconditionally, before either of
    those deploy-gated steps, so it's produced on a --dry-run and even on
    a failed deploy -- pass write_debug_md=False to skip it.

    Returns True only when the digest was both rendered and deployed."""
    if not digest:
        logger.warning("Empty digest -- nothing to deliver.")
        return False

    config = load_config(config_path)
    today = datetime.now().astimezone().date()
    today_iso = today.isoformat()

    conn = connect_history_db(db_path)
    try:
        _ensure_digests_table(conn)
        archive_dates = _get_archive_dates(conn)
    finally:
        conn.close()
    if today_iso not in archive_dates:
        archive_dates.append(today_iso)

    category_order = config.get("category_order") or DEFAULT_CATEGORY_ORDER
    missing = sorted(c for c in (set(category_order) | set(digest.keys())) if c not in digest)

    # No "nothing today for: ..." note on the page any more: with eleven topical
    # sections an empty one is normal, not a failure. `missing` is still recorded
    # in history.db and the debug Markdown.
    html = render_digest_html(digest, today, archive_dates, [], category_order)
    dated_path = write_output(html, today, output_dir)
    logger.info("Digest rendered to %s", dated_path)

    if write_debug_md:
        markdown_text = render_digest_markdown(digest, today, missing, category_order)
        debug_path = write_debug_markdown(markdown_text, today, debug_dir)
        if debug_path:
            logger.info("Debug Markdown mirror written to %s (and debug/latest.md)", debug_path)

    conn = connect_history_db(db_path)
    try:
        _record_digest(conn, today_iso, sorted(digest.keys()), missing)
    finally:
        conn.close()

    if not deploy:
        logger.info("deploy=False -- local dry run, skipping Cloudflare deploy and mark_as_shown().")
        return False

    if not deploy_to_cloudflare(output_dir, config):
        logger.error(
            "Deploy failed -- digest stays local-only; articles were NOT marked as shown "
            "(they'll be reconsidered, and possibly re-summarized, on the next run)."
        )
        return False

    shown_articles = [a for section in digest.values() for a in section["articles"]]
    mark_as_shown(shown_articles, db_path=db_path)
    logger.info("Delivered and marked %d article(s) as shown.", len(shown_articles))
    return True


def _setup_logging() -> None:
    log_dir = PROJECT_ROOT / "logs"
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
    from core.interest import assign_sections
    from core.ingestion import ingest_all
    from core.summarizer import summarize_all

    _setup_logging()

    print("Running ingestion...")
    articles = ingest_all()
    print(f"Ingested {len(articles)} article(s) total.")

    print("Running classification (reclassify + drop junk)...")
    classified = classify_all(articles)
    print(f"{len(classified)} article(s) kept after classification.")

    print("Running deduplication (does NOT mark anything as shown)...")
    deduped, _ = dedupe_articles(classified)
    deduped = assign_sections(deduped)
    print(f"{len(deduped)} article(s) after dedup.\n")

    print("Running summarization (map-reduce via Ollama)...")
    digest = summarize_all(deduped)
    print(f"Digest sections produced: {len(digest)}\n")

    print("Running delivery (render -> write output/ -> deploy to Cloudflare Pages)...")
    delivered = run_delivery(digest)
    if delivered:
        print("\n=== Delivered and deployed. Articles marked as shown. ===")
    else:
        print(
            "\n=== NOT deployed (see log above for why -- likely wrangler/config.yaml isn't set up "
            "yet, or digest was empty). Local copy is still in output/index.html. "
            "Nothing was marked as shown. ==="
        )
