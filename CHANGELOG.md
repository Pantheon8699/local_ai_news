# Changelog — Morning Brief pipeline

Newest first.

Where each part comes from:
- **12 Oct 2026 onward:** written as the changes were made.
- **10–12 Oct 2026:** from the development sessions of those days.
- **6 Sept 2026:** from that session's notes.
- **v2.1 to v2.1.3 (Sept 2026):** reconstructed from the version notes left
  in the code, the run logs and the `debug/` mirrors. Dates are the earliest
  evidence found, not exact, and the reasoning behind each change is only as
  complete as those notes.
- **July 2026:** from `news_pipeline_v2.0_blueprint.md` (the v2.0 design,
  dated 19 July).

## 19 Oct 2026 — world stories filed under Italy, reworded bullets

Trigger: review of the 13–19 Oct digests. On 19 Oct the Italy section
carried Trump's endorsement of Netanyahu and the drone attack on St
Petersburg; on 13 Oct a Yemen strike on Riyadh and a US Senate story.

**Section second look** (`core/interest.py`, `prompts/section_country.txt`)
- Cause: the "where does the story take place?" check, asked to pick one of
  ITALIA/USA/EUROPA/ALTRO/INTERNAZIONALE, answered ITALIA for Russia, Israel,
  Yemen and Australia stories and moved them into Italy. Reordering the
  options fixed 19 Oct's cases but still got 7 of 17 older ones wrong.
- Fix: it now answers with the country's name, mapped to a section in code
  (`_country_section`, Italian and English names). 30 of 33 known cases right.
  The 3 misses are ambiguous: US senators on a Russia deal answer "Russia".

**One-source bullets** (`core/summarizer.py`, `repair_citations`)
- Cause: the reduce model rewrote correct micro-summaries into wrong bullets
  (19 Oct: "Marattin offers himself to the centre-right, Calenda attacks him"
  became "Marattin joined the centre-right, criticising Calenda"). Nearly
  every bullet cites one summary, so the rewrite added risk for nothing.
- Fix: a bullet citing one summary is that summary verbatim; two bullets
  citing the same summary are shown once. Merged bullets [1,4] keep the
  model's text.

**Looked at, not changed**
- Same-day dedup compares an article only to a cluster's first member.
  Comparing to any member merged one extra pair per bucket on 19 Oct but also
  pulled a US diesel-tax story into the Russia-sanctions cluster.
- A "name people, not 'the journalist'" rule in `micro_summary.txt` did not
  name Calabresi (his article's body is a 219-character paywall teaser) and
  made two other summaries worse.
- Errors that start in the micro-summary itself (House Ways and Means as
  "Comitato per i Mezzi di Sostegno", wrong roles) are not caught.
- Repubblica's Italian articles are paywalled: 3–9 title-only bullets a day.
- The Opinion/Geopolitics bucket at 0 on 19 Oct is correct: over the weekend
  its feeds only re-served pieces already shown.

Not run end to end; checked on the saved 13–19 Oct articles.

## 12 Oct 2026 — bullets linked to the wrong articles, double-checks, new model

Trigger: in the 12 Oct digest the Italy section showed bullets above another
story's sources ("Laika…" over the ex Ilva strike, "Amato…" over a Peru
article), and two stories were marked as shown without appearing at all.

**Bullet ↔ source linking** (`core/summarizer.py`, `repair_citations`)
- Cause: a bullet is tied to its articles only by the `[n]` tag the model
  appends, and the model numbered them one off. Replaying all saved digests,
  about 100 of 1,667 bullets carried a wrong or extra source.
- Fix: each tag is checked by word overlap against the summaries it cites;
  wrong numbers are dropped, and the best-matching summary is used if none is
  left. A summary no bullet cites is appended as its own bullet.
- The prompt's example sentence ("Il PIL è cresciuto del 2%…") had been copied
  into the digest as news on 5 of 14 days; it is now cut from the output.
  Rewording the example in the prompt made tagging worse, so the prompt is
  unchanged.

**Double-checks**
- Sections (`core/interest.py`, `prompts/section_country.txt`): a second
  question, "where does the story take place?", overrides a geographic or
  off-list section answer when the two disagree. Does not address thematic
  mistakes (an attack filed under Crime instead of Geopolitics).
- Invented numbers (`unsupported_numbers`, `prompts/micro_fix.txt`): every
  number in a micro-summary must appear in the article; otherwise the model
  rewrites it once, shown the numbers it made up. Catches wrong years
  ("midterms of November 2022"); does not catch wrong names or roles.
- Title-only articles (extraction failed) are no longer sent to the reduce
  model, which had filled the gap from neighbouring summaries; they appear as
  their own bullet with the original title.
- Tried and rejected: a model "second reading" of each summary against its
  article. With qwen2.5:7b it missed most of the day's real errors and
  changed a correct "1.000 euro" to "500 euro".

**Models**
- Classification, interest filter, section assignment and micro-summaries
  moved from `qwen2.5:7b-instruct` to `qwen3.5:9b-q4_K_M`, called with
  thinking off (55 s per answer with it on, 0.5 s off).
- The reduce stage (writing the bullets) stays on `qwen2.5:7b-instruct`:
  qwen3.5 merged unrelated neighbouring stories into one bullet in 3 runs of 3.
- Measured on the 65 articles of the 12 Oct digest: qwen3.5 files Peru,
  Germany and California correctly by itself and invents no years, but its
  interest filter is stricter (kept 47 of the 65, and 19 of the 116 the old
  model dropped) and its summaries are longer.
- Removed unused Ollama models: `gemma4:12b`, `deepseek-coder-v2:16b`,
  `llama3.2:3b`, `qwen2.5-coder:1.5b`.
- Embeddings moved from `nomic-embed-text` to `bge-m3`. Half the sources are
  Italian and half English, and nomic barely connected the two: on the day's
  ten known same-story pairs bge-m3 ranked all of them in the top 7 of 2,080
  pairs, nomic from 1st to 1,022nd.
- Two thresholds instead of one. Same-day merging: 0.65 (was 0.85). Cross-day
  "already shown": 0.85, kept strict because a follow-up to a running story
  scores 0.65-0.76 against the earlier piece and must not be suppressed
  (8 Oct vs 6 Oct: 12 of 222 new articles matched at 0.65, none at 0.80).
- Consequence: embeddings stored before this date are a different size and
  are skipped, so until about 22 Oct only exact-URL matches count as
  "already shown" against older articles.
- Replayed on the 12 Oct articles: 65 -> 62, merging Kharkiv/Zelensky
  (Adnkronos + Repubblica), the Riyadh attack (ANSA + BBC) and the energy
  truce (ANSA + NPR). Merging is still only within a source bucket, so an
  Italian outlet and a US outlet in different buckets are not merged.

Not run end to end yet with these changes.

## 11–12 Oct 2026 — two mornings without a digest

- The 07:15 runs of 11 and 12 Oct started but were killed by systemd after 60
  minutes, before delivery.
- Cause: the Ollama Docker container (`ollama-inference`) had lost the GPU
  after a systemd reload on the night of 10 Oct and was running on CPU;
  classification alone went from about 2 minutes to about 15.
- Fix: container restarted on 12 Oct and the digest re-run at 10:06.
- Still open: this recurs on any systemd reload. The usual permanent fix is
  `"exec-opts": ["native.cgroupdriver=cgroupfs"]` in `/etc/docker/daemon.json`
  followed by a Docker restart (needs sudo, restarts every container).

## 10 Oct 2026 — personalisation and new sections (first live in the 12 Oct digest)

Trigger: too much sport and local US news; the selection did not match what
the reader cares about.

- `interest_profile.md`: a checklist of news types marked y / m / n by hand,
  condensed into `prompts/interest_filter.txt`.
- Sections rule: politics is split by country, everything else by topic.
  On that day's 99 kept articles roughly eight were misfiled.
- Like/dislike model: logistic regression on the stored embeddings, chosen as
  the simplest option that works with few votes; inactive until 200 votes.
- Lean tags (L / CL / CR / R, from the `lean` field in `sources.json`) shown
  on bullets in the four country sections and Geopolitics.
- The empty-section note ("nothing today for…") was removed.

- `core/interest.py` added: an interest filter (one keep/drop call per
  article against `prompts/interest_filter.txt`, condensed from
  `interest_profile.md`); like/dislike votes from the published page, synced
  from Cloudflare KV (`functions/api/vote.js`, `wrangler.toml`) and used to
  rank and drop articles once there are 200 votes; and `assign_sections`,
  which files articles under eleven reader-facing sections (Italy, Europe,
  United States, …) instead of the five source buckets.
- The "Comparativa" section is no longer built; political bullets carry their
  sources' lean tags instead.

## v2.1.3 — after the 27 Sept 2026 run

- Classifier moved from `llama3.2:3b` to `qwen2.5:7b-instruct`: the 3B model
  had moved or discarded all 60 Italian-source articles on 27 Sept. A
  second-look confirmation was added before an article leaves an Italian
  category.
- Any CJK/Hangul/Kana character in model output now triggers one retry, then
  the call is dropped.
- "Comparativa" clusters must span genuinely different outlets and markets
  (`market` field added to `sources.json`); its bullets are validated before
  being shown.

## v2.1.2 — late Sept 2026

- Articles with no extracted body get a fixed "full text not available"
  summary without a model call; feeding the title as the body was the trigger
  for mixed-language output.
- Non-Italian sources can no longer be reclassified into the Italian
  categories.

## v2.1 — by 26 Sept 2026

- New classification stage between ingestion and dedup: re-derives each
  article's category from its content and drops sport, weather, gossip and
  product reviews.
- Reduce stage split into batches of 15 summaries, after 30-item prompts
  produced merged bullets and non-Italian text.
- Each bullet gets a `[n]` source tag, rendered as an expandable list of the
  articles behind it.
- "Comparativa" section added (same story across outlets of different lean).
- Plain-Markdown copy of each digest written to `debug/`.

## 6 Sept 2026 — first unattended run

Summarizer finished, delivery to Cloudflare
Pages (`morning-brief`), `main.py` orchestrator with exit codes and
`--dry-run`, systemd timer at 07:15, GPU pre-flight check. First full run:
184 articles ingested, 171 after dedup, 5 categories, deployed.

## 19 July 2026 — v2.0 design

`news_pipeline_v2.0_blueprint.md`. Replaced the v1.0 design:

| Area | v1.0 | v2.0 |
|---|---|---|
| Ingestion | Public APIs + scraping | RSS + scraping only |
| Dedup | Keyword overlap | Embedding similarity |
| Summarization | One large prompt per category | Map-reduce: per article, then per category |
| History | None | SQLite `history.db` for cross-day dedup |
| Delivery | Telegram bot | Static web app on Cloudflare Pages, installable on the phone |

Goals set there and unchanged since: no paid AI, everything local through
Ollama on an 8 GB GPU; dense factual bullets; unattended; readable from the
phone with the PC off.

No record of v1.0 itself, or of the build between July and 6 Sept, exists in
this folder.
