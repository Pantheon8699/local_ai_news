# System Architecture Blueprint: Personal AI News Aggregator & Summarizer (v2.0)

This document supersedes v1.0. Key changes from the original design: the public-API ingestion layer has been dropped in favor of a unified RSS/scraping approach; deduplication now uses local embeddings instead of keyword overlap; summarization uses a map-reduce pattern instead of single large-context calls; and delivery has moved from a Telegram bot to a self-contained web app deployed to Cloudflare Pages, installable on the phone as a PWA.

---

## 1. Project Objectives & Core Philosophy

* **Zero Cloud AI Costs:** All summarization runs locally via Ollama. No paid LLM APIs.
* **Information Density:** No sensationalism, no filler. Bulleted, factual, quantitative where possible.
* **Automation:** Triggered automatically at boot/login or a fixed morning time, with no manual intervention and no silent failure if the PC was off at the scheduled time.
* **Read Anywhere:** The digest must be reachable from the phone at any time of day, independent of whether the home PC is on — this is the reason delivery moved from Telegram/local-server to a deployed static web app.

---

## 2. Environment & Hardware Specifications

* **OS:** Linux (Ubuntu / Debian / Fedora / Arch)
* **Hardware Profile:**
  * **RAM:** 24 GB system RAM
  * **GPU:** NVIDIA RTX 5060 (8 GB VRAM)
* **Local AI Runtime:** **Ollama**
  * *Primary model:* `qwen2.5:7b-instruct`, 4-bit/5-bit quantized (~4.7 GB VRAM, comfortable headroom on 8 GB).
  * *Embedding model:* `nomic-embed-text` (used for deduplication — see Section 4).
  * *Fast fallback:* `llama3.2:3b`, useful if a run is time-constrained.

---

## 3. Data Acquisition: Unified RSS & Scraping Layer

**Decision:** the public-API layer (NewsAPI, GNews, Mediastack) has been dropped. None of them are a good fit for this project:

* NewsAPI's free tier delays articles 24 hours — incompatible with a "this morning's news" digest.
* Mediastack's free tier is 100 requests/**month**.
* GNews's free tier is small (100 req/day, 10 articles/request) and burns fast.
* None of them have meaningful Italian regional coverage — this was true from the start and is the actual reason a scraping layer was already planned.

Since RSS/scraping already had to carry the Italian layer, and the major global outlets (Reuters, AP, BBC, Foreign Affairs, Foreign Policy, Wired) all publish open RSS feeds directly, there's no benefit to running two ingestion mechanisms. Everything goes through one path.

### 3.1 Sources (from `sources.json`)

Organized by category, each entry is `(section_name, feed_url)`:

* **Italian National/Regional:** ANSA (Politica, Mondo, Economia, Cronaca, Tecnologia), Il Post (Politica, Italia, Mondo, Economia, Europa, Cronaca), Adnkronos
* **Italian Economy/Finance:** Il Sole 24 Ore (Mondo, Finanza)
* **US / World:** Reuters, AP, BBC, Politico, Foreign Affairs, Foreign Policy
* **Tech:** Wired (IT + international)

### 3.2 Mechanism

1. Parse RSS/Atom via `feedparser` for entry links, titles, and publish timestamps.
2. Fetch full article text via `trafilatura` (replaces `newspaper3k`, which is unmaintained and has unreliable date/text extraction on modern markup). `trafilatura` gives more reliable extraction and built-in metadata parsing including publish date.
3. Filter to entries published today or yesterday. Apply this filter uniformly per entry — **no early-exit heuristic on the first N entries** (the v1 design's "skip section if first 3 entries are stale" was removed: RSS feeds don't always sort strictly chronologically, e.g. corrections/updates get bumped, so this heuristic risked dropping legitimately fresh stories).
4. Basic politeness: custom User-Agent, timeouts, retry with backoff, small delay between requests to the same domain.
5. Per-section article limits stay configurable (as in the original notebook), just applied without the early-skip logic.

### 3.3 Optional future addition

If broader English-language coverage is ever wanted beyond RSS: **Currents API** (real-time, ~600–1,000 req/day free, no credit card) or **The Guardian API** (fully free, full article text, no restrictions) are the only free options actually worth adding later. Not required for launch.

---

## 4. Normalization, Deduplication & Context Aggregation

### 4.1 Schema

Every ingested article is flattened into:

```python
{
    "category": "Economy" | "US News" | "Italian Local" | "World News",
    "source": "Source Name (e.g., Reuters, ANSA)",
    "title": "Cleaned Headline String",
    "body": "Extracted article text",
    "url": "https://source-url-link.com/article",
    "published_at": "2026-07-19T07:03:00Z"
}
```

### 4.2 Deduplication: embeddings instead of keyword overlap

**Decision:** replaced Jaccard/keyword-overlap similarity with embedding-based clustering.

Rationale: Italian outlets frequently rephrase or translate the same AP/Reuters story rather than sharing headline vocabulary, so token-overlap methods miss a large share of real duplicates. Since Ollama is already running locally, embeddings are free to compute.

* Generate an embedding per article (title + first ~2 sentences) via `nomic-embed-text`.
* Cluster articles within a category using cosine similarity (threshold tunable, start ~0.85).
* Within a cluster, retain the article with the longest/most complete body text; merge source attributions from the rest; drop the redundant entries.

### 4.3 Local history (new)

**Decision:** add a SQLite database (`history.db`) that v1 didn't have.

* Stores every processed article (dedup key, URL, category, date, embedding hash) and every generated digest.
* Enables:
  * **Cross-day dedup** — an ongoing story doesn't get re-summarized as "new" every morning.
  * **Archive** — past digests remain browsable in the web app (Section 6).
  * **Debuggability** — if something looks wrong in a digest, you can inspect exactly what the pipeline ingested that day.

---

## 5. Local LLM Synthesis (Map-Reduce Pattern)

**Decision:** replaced single large "dump all articles in one category into one prompt" calls with a two-stage map-reduce pipeline. A 7B model degrades on very long contexts ("lost in the middle"), and one giant prompt per category doesn't scale as source count grows.

### 5.1 Stage 1 — Map: per-article micro-summary

* Each deduplicated article gets its own short, fast Ollama call: 1–2 sentence factual extraction (who/what/numbers/timeline).
* Cheap, parallelizable, and produces a reusable intermediate artifact (stored in `history.db`).

### 5.2 Stage 2 — Reduce: per-category synthesis

* The micro-summaries for a category (not raw article bodies) are batched into the "executive editor" prompt.
* This is the pass that produces the final structured, high-density Italian-language output for that category.

### 5.3 Inference configuration

* Endpoint: local Ollama REST (`http://localhost:11434/api/chat`), `"stream": false`.
* `temperature`: 0.15–0.20 for both stages — literal extraction over creative variation.
* **Role:** elite, objective, zero-fluff media intelligence officer.
* **Output constraints:** bulleted, quantitative, dated, causally structured.
* **Linguistic rule:** ingest Italian or English equally; always output final digest in refined, high-density Italian.

---

## 6. Delivery: Web App on Cloudflare Pages (replaces Telegram)

### 6.1 Why this replaced the Telegram bot

Telegram's 4,096-character limit forced chunking mid-digest, MarkdownV2 requires escaping special characters throughout the generated text (a single unescaped character breaks the API call), and a chat log has no real archive/search/typography. A deployed static web page solves all three, and — since the requirement is "readable anywhere, anytime, independent of whether the PC is on" — the output has to live on a persistent host rather than a local server anyway.

### 6.2 Generation

* `delivery.py` (renamed from the Telegram-era module) renders the day's category summaries into a static HTML page using a simple template (`templates/digest.html`).
* Each day's output is written to `output/YYYY-MM-DD/index.html`, and `output/index.html` is regenerated to always point at the latest digest (with a lightweight nav to browse previous days, pulled from `history.db`).

### 6.3 Deployment

* Deploy via the Cloudflare Wrangler CLI as the final pipeline step — **direct upload, not git-linked** (a git-linked Pages project would mean committing an HTML diff every single morning forever, which is unnecessary repo churn for content that doesn't need version history beyond what's already in `history.db`):

  ```bash
  wrangler pages deploy ./output --project-name=morning-brief
  ```

* Free tier, no bandwidth cap that matters at this traffic level.
* **Privacy (optional, not required at launch):** Cloudflare Access can gate the URL behind email-OTP login, free for up to 50 users, if the obscure-but-technically-public URL isn't private enough later.

### 6.4 PWA install (feels like a native app)

* `manifest.json` + a few `<meta>` tags on the page enable "Add to Home Screen" on iOS/Android — full-screen launch, real icon, no browser chrome.
* No app store, no build step, no native code.

### 6.5 Known tradeoff

This delivery method has no push notification — you have to open the app rather than being pinged. If that turns out to matter, a lightweight notification path (e.g. a one-line Telegram or ntfy.sh ping saying "today's digest is ready," with the actual content living on the web app) can be layered on later without touching the rest of the architecture.

---

## 7. Automated Execution Framework on Linux

**Decision:** kept systemd timers as primary, with two reliability fixes over v1:

* `Persistent=true` on the timer unit — if the PC is off at the scheduled time, the job runs on next boot instead of silently never firing that day.
* `After=network-online.target` / `Wants=network-online.target` on the service unit — prevents the pipeline from starting before Wi-Fi is actually associated at boot/login, which would otherwise cause silent scraping failures.

```ini
# ~/.config/systemd/user/news_digest.timer
[Unit]
Description=Morning news digest timer

[Timer]
OnCalendar=*-*-* 07:15:00
Persistent=true

[Install]
WantedBy=timers.target
```

```ini
# ~/.config/systemd/user/news_digest.service
[Unit]
Description=Morning news digest pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 %h/ai_news_pipeline/main.py
```

### 7.1 Failure resilience (new)

* All stages log to a file (`logs/run_YYYY-MM-DD.log`).
* If a source fails (403, timeout, parse error), the pipeline logs it and continues — one bad scraper shouldn't kill the run.
* If the deployed digest is missing sections due to failures, the web app footer notes which sources didn't return data that day, rather than failing silently.

---

## 8. Modular Directory Architecture

```text
ai_news_pipeline/
│
├── main.py                    # Master entry point; coordinates the pipeline stages
├── config.yaml                # Cloudflare API token, project name, model names, thresholds
├── sources.json                # RSS feed URLs, section names, per-section article limits
├── history.db                  # SQLite: articles seen, embeddings, past digests
│
├── core/
│   ├── __init__.py
│   ├── ingestion.py            # feedparser + trafilatura scraping, date filtering
│   ├── deduplication.py        # nomic-embed-text embeddings, cosine clustering, cross-day dedup
│   ├── summarizer.py           # Map-reduce Ollama calls (per-article, then per-category)
│   └── delivery.py             # Renders HTML, deploys via wrangler, writes history.db
│
├── templates/
│   ├── digest.html              # Main digest template (category tabs, archive nav)
│   └── manifest.json            # PWA manifest for home-screen install
│
├── prompts/
│   ├── micro_summary.txt        # Stage 1 (map) prompt
│   └── executive_editor.txt     # Stage 2 (reduce) prompt
│
├── logs/
│   └── run_YYYY-MM-DD.log
│
└── requirements.txt            # requests, feedparser, trafilatura, PyYAML, sqlite3 (stdlib)
```

---

## 9. Summary of Changes from v1.0

| Area | v1.0 | v2.0 |
|---|---|---|
| Ingestion | Public APIs + scraping (dual layer) | RSS/scraping only, unified |
| Dedup | Jaccard/keyword overlap | Embedding cosine similarity (`nomic-embed-text`) |
| Summarization | Single large prompt per category | Map-reduce: per-article → per-category |
| History | None | SQLite (`history.db`) for cross-day dedup + archive |
| Delivery | Telegram bot, MarkdownV2, chunking | Static web app on Cloudflare Pages, PWA installable |
| Access | Wherever Telegram is logged in | Anywhere, anytime, independent of PC uptime |
| Automation reliability | Basic systemd/cron | `Persistent=true` + `network-online.target` |
