# Morning Brief

A personal morning news digest that runs unattended on a home PC with local
models only. It reads news sites' RSS feeds, drops what the reader doesn't care
about, merges duplicate stories, summarises each one, and publishes a dated
page to Cloudflare Pages that can be installed on a phone.

It ships configured for an **Italian reader following Italy and the world**
(see [Personalising](#personalising)). The digest is written in Italian.

## How it works

```
RSS feeds ─▶ ingestion ─▶ classification ─▶ interest filter ─▶ dedup ─▶ votes ─▶ sections ─▶ summaries ─▶ page
```

| Stage | Module | What it does |
|---|---|---|
| Ingestion | `core/ingestion.py` | Reads the feeds in `sources.json` and extracts each article's text with trafilatura. |
| Classification | `core/classification.py` | Re-derives each article's topic and drops sport, weather, gossip and reviews. |
| Interest filter | `core/interest.py` | One keep/drop question per article against `prompts/interest_filter.txt`. |
| Dedup | `core/deduplication.py` | `bge-m3` embeddings merge same-day duplicates and skip stories already shown on earlier days (`history.db`). |
| Votes | `core/interest.py` | 👍/👎 from the page train a small logistic regression that ranks articles (off until 200 votes). |
| Sections | `core/interest.py` | Files each article under a section: Italy, Europe, United States, Geopolitics, Economy… |
| Summaries | `core/summarizer.py` | One 1–2 sentence summary per article, then one bullet list per section, each bullet linked to its sources. |
| Delivery | `core/delivery.py` | Renders `templates/digest.html` and deploys it with `wrangler pages deploy`. |

Every stage degrades instead of crashing: a dead feed, a failed model call or a
failed deploy is logged and the run continues. `CHANGELOG.md` records what was
changed and why, with the measurements behind each change.

## Requirements

- Linux with an NVIDIA GPU (8 GB is enough) and Python 3.11+
- [Ollama](https://ollama.com) with these models:
  ```bash
  ollama pull qwen3.5:9b-q4_K_M
  ollama pull qwen2.5:7b-instruct
  ollama pull bge-m3
  ```
- Optional, for publishing: a Cloudflare account and
  [wrangler](https://developers.cloudflare.com/workers/wrangler/)

## Setup

```bash
git clone https://github.com/Pantheon8699/local_ai_news.git ~/ai_news_pipeline
cd ~/ai_news_pipeline
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Try a run without publishing anything; the page is written to `output/index.html`:

```bash
venv/bin/python main.py --dry-run --skip-gpu-check
```

### Publishing to Cloudflare Pages (optional)

1. `wrangler login`, then create the project:
   `wrangler pages project create morning-brief`, and set the same name as
   `cloudflare.project_name` in `config.yaml`.
2. For the like/dislike buttons: `wrangler kv namespace create VOTES`, paste
   the id into `wrangler.toml`, then set a secret key with
   `wrangler pages secret put VOTE_KEY --project-name morning-brief`. Put the site URL and the same key under
   `votes:` in `config.yaml`. The page asks for the key once per device.
3. `venv/bin/python main.py` runs the whole pipeline and deploys.

Leave `project_name` as `null` to only build the page locally.

### Running every morning

```bash
cp systemd/news_digest.service systemd/news_digest.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now news_digest.timer
loginctl enable-linger "$USER"   # so it runs while you're logged out
```

The timer fires at 07:15. Edit the paths in `news_digest.service` if you didn't
clone into `~/ai_news_pipeline`. Before starting, the run waits for the GPU
to be free (`gpu_guard` in `config.yaml`). Logs go to `logs/run_YYYY-MM-DD.log`.

## Personalising

The shipped configuration is the **Italy / world preset**. These are the files
that encode it:

| File | What it decides | To change it |
|---|---|---|
| `sources.json` | Which feeds are read, grouped into five buckets. `lean` drives the L/CL/CR/R tags on political bullets, `market` (`IT` or `US/UK`) which outlets count as the same press, `limit` and `lookback_days` how many articles per feed and how far back. | Add or remove feeds. Keep the five bucket names unless you also change `ALLOWED_CATEGORIES` in `core/classification.py` and `prompts/classifier.txt`. |
| `interest_profile.md` | Your interests as a y / m / n checklist. Read by you, not by the code. | Fill it in, then rewrite `prompts/interest_filter.txt` from it. That prompt is what the pipeline uses. |
| `prompts/interest_filter.txt` | Keep/drop rules: always drop, keep only if major, always keep. | Mirror your checklist. Lists beat long explanations for the local model. |
| `SECTIONS` in `core/interest.py`, `category_order` in `config.yaml`, `prompts/section.txt`, `prompts/section_country.txt` | The page's sections and their order. Politics is split by country (Italy, Europe, US, rest of the world), everything else by topic. | To centre another country, rename the first section and change the country mapping in `_country_section()` along with the two section prompts. |
| `prompts/*.txt` | All prompts are in Italian, and summaries are written in Italian. | Translate the prompts to change the output language. Test on a few days of articles first: the local models are sensitive to wording (see `CHANGELOG.md`). |
| `templates/digest.html`, `templates/manifest.json` | Page title ("Morning Brief"), styles, app icon. | Edit freely. |

After changing a prompt, run with `--dry-run` and read `output/index.html`
before letting the timer publish.

## Responsible use

This is a personal reader. Before you run it:

- **Your sources' terms apply to you.** Check each site's terms of use and
  `robots.txt` before adding it to `sources.json`. Prefer outlets that publish
  full-text RSS or permit automated access. Drop any source that blocks you
  instead of working around the block.
- **Keep the published page private.** It shows excerpts of the original
  articles under each bullet. Republishing those publicly can infringe
  copyright and press publishers' rights (in the EU, Art. 15 of Directive
  2019/790). Put the site behind
  [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/policies/access/)
  (free for up to 50 users) or don't deploy at all.
- **Don't redistribute what it collects.** `history.db`, `output/`, `debug/`
  and `logs/` contain article text and are gitignored for that reason.
- **Summaries can be wrong.** They are produced by small local models;
  `CHANGELOG.md` lists the error types seen so far. Open the source before
  relying on a bullet.

### What the fetcher does

Every request checks the site's `robots.txt` first (RFC 9309) and identifies
itself honestly as `MorningBrief/1.0`. It waits at least a second between
requests to the same site and gives up after three failed attempts. A source
that refuses it is logged and skipped. Checked on 19 Oct 2026 with the preset
sources: Il Fatto Quotidiano's `robots.txt` disallows it; Politico and the
Washington Post block the feed; Libero allows the feed but blocks article
pages, so its items show as title only. Everything else works.

### Browser impersonation

The HTTP client used here, `curl_cffi`, can imitate a real browser's TLS
fingerprint, which gets past some sites' bot protection. The constant
`IMPERSONATE` in `core/ingestion.py` exposes that option. **It is off and is
not recommended.** Bypassing a site's bot protection may breach its terms of
use, and in the EU it can fall outside the text-and-data-mining exception
(Art. 4 of Directive 2019/790) where the publisher has opted out. The author
neither uses nor endorses this setting in the published code. Anyone who turns
it on does so by their own choice, at their own risk and on their own legal
responsibility.

### Disclaimer

This software is provided for personal and educational use, as is, without
warranty of any kind (see `LICENSE`). The author is not responsible for how it
is configured or used, including which sites it is pointed at, whether their
terms are respected, and what is done with the content it retrieves. This
project is not affiliated with any of the publishers listed in `sources.json`.
