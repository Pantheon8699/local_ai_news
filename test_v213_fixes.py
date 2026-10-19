"""
Ad hoc verification script for the v2.1.3 fixes (mocked Ollama -- no live
model calls, same philosophy as the project's existing "unit tests against
mocked Ollama responses" per news_pipeline_v2.md 10.7). Not part of the
shipped pipeline; run once to sanity-check the new guards/validation logic
before trusting a real --dry-run against live Ollama.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core.classification as classification
import core.deduplication as deduplication
import core.comparison as comparison


def test_classification_drain_guard():
    print("=== classify_all(): drain guard + bleed guard ===")
    db_path = Path(tempfile.mkstemp(suffix=".db")[1])

    # Simulate today's exact failure pattern:
    #  - a genuine Italian crime story the first pass wrongly wants to SCARTA
    #  - a genuine Italian politics story the first pass wrongly wants to
    #    move to Opinion/Geopolitics
    #  - a legitimate Italian->US/World move (ANSA - Mondo style) that
    #    SHOULD go through once confirmed
    #  - a non-Italian source the first pass wrongly wants to pull into
    #    Italian National/Regional (the OLD bleed bug -- must stay blocked)
    articles = [
        {"url": "https://a/1", "category": "Italian National/Regional", "source": "Adnkronos",
         "title": "Messina, donna getta acido contro il suo avvocato", "body": "x" * 200},
        {"url": "https://a/2", "category": "Italian National/Regional", "source": "Il Fatto Quotidiano",
         "title": "Tajani a Vannacci: dichiarazione sulla foto ritoccata", "body": "x" * 200},
        {"url": "https://a/3", "category": "Italian National/Regional", "source": "ANSA - Mondo",
         "title": "Attacchi russi su Zaporizhzhia, tre morti", "body": "x" * 200},
        {"url": "https://a/4", "category": "US/World", "source": "BBC News - World",
         "title": "Some unrelated world story", "body": "x" * 200},
    ]

    def fake_call(prompt, timeout=None):
        # crude routing based on prompt content, mimicking what the real
        # (buggy, per today's log) first-pass model proposed, and what a
        # correct second-look reviewer should say.
        if "Rispondi con UNA SOLA PAROLA" in prompt:  # confirm prompt
            if "Messina, donna getta acido" in prompt:
                return "MANTIENI"   # a real crime story -- must NOT confirm SCARTA
            if "Tajani a Vannacci" in prompt:
                return "MANTIENI"   # Italian domestic politics -- must NOT confirm the move
            if "Attacchi russi su Zaporizhzhia" in prompt:
                return "CONFERMA"   # genuinely about another country -- OK to confirm
            return "MANTIENI"
        # first-pass classify prompt
        if "Messina, donna getta acido" in prompt:
            return "SCARTA"                       # today's actual (wrong) bug behavior
        if "Tajani a Vannacci" in prompt:
            return "Opinion/Geopolitics"           # today's actual (wrong) bug behavior
        if "Attacchi russi su Zaporizhzhia" in prompt:
            return "US/World"                      # legitimate reclassification
        if "Some unrelated world story" in prompt:
            return "Italian National/Regional"     # the OLD bleed bug
        return None

    classification._call_ollama_chat = fake_call

    result = classification.classify_all(articles, db_path=db_path)
    by_url = {a["url"]: a for a in result}

    assert by_url["https://a/1"]["category"] == "Italian National/Regional", \
        "FAIL: acid-attack crime story was wrongly discarded/moved"
    assert by_url["https://a/2"]["category"] == "Italian National/Regional", \
        "FAIL: Italian domestic politics story was wrongly moved to Opinion/Geopolitics"
    assert by_url["https://a/3"]["category"] == "US/World", \
        "FAIL: legitimate ANSA-Mondo-style reclassification was blocked"
    assert by_url["https://a/4"]["category"] == "US/World", \
        "FAIL: bleed guard regression -- BBC story ended up in an Italian category"

    print("PASS: crime story kept Italian, politics story kept Italian, "
          "legitimate world-affairs reclass went through, bleed guard still holds.")
    db_path.unlink(missing_ok=True)


def test_comparison_subset_and_validation():
    print("\n=== comparison_subset() + comparison._validate_bullet() ===")

    # Today's exact bug: two ANSA - Mondo duplicates (same source, same
    # lean, same market) -- must NOT qualify on their own.
    ansa_dup_cluster = [
        {"source": "ANSA - Mondo", "lean": "wire", "market": "IT",
         "title": "Lavrov, dichiarazione", "url": "https://ansa/europa"},
        {"source": "ANSA - Mondo", "lean": "wire", "market": "IT",
         "title": "Lavrov, dichiarazione", "url": "https://ansa/nordamerica"},
    ]
    assert deduplication.comparison_subset(ansa_dup_cluster) is None, \
        "FAIL: same-outlet ANSA duplicate pair qualified as a comparison"
    print("PASS: same-outlet duplicate pair correctly rejected.")

    # A genuine Italian-vs-Italian comparison: Repubblica (center-left) vs
    # Libero (right), same market.
    it_cluster = [
        {"source": "La Repubblica", "lean": "center-left", "market": "IT",
         "title": "Decreto scuola", "url": "https://repubblica/1"},
        {"source": "Libero Quotidiano", "lean": "right", "market": "IT",
         "title": "Decreto scuola", "url": "https://libero/1"},
    ]
    assert deduplication.comparison_subset(it_cluster) is not None, \
        "FAIL: genuine Italian left/right pair did not qualify"
    print("PASS: Italian-vs-Italian (center-left vs right) pair correctly qualifies.")

    # A genuine US/UK-vs-US/UK comparison: Fox (right) vs Daily Beast (left).
    us_cluster = [
        {"source": "Fox News - Politics", "lean": "right", "market": "US/UK",
         "title": "Trump policy", "url": "https://fox/1"},
        {"source": "The Daily Beast", "lean": "left", "market": "US/UK",
         "title": "Trump policy", "url": "https://beast/1"},
    ]
    assert deduplication.comparison_subset(us_cluster) is not None, \
        "FAIL: genuine US/UK left/right pair did not qualify"
    print("PASS: US/UK-vs-US/UK (right vs left) pair correctly qualifies.")

    # Cross-market: ANSA - Mondo (IT) + BBC (US/UK) -- must NOT qualify,
    # per the explicit "italian vs italian, US/UK vs same" requirement.
    cross_market_cluster = [
        {"source": "ANSA - Mondo", "lean": "wire", "market": "IT",
         "title": "Ukraine story", "url": "https://ansa/2"},
        {"source": "BBC News - World", "lean": "center", "market": "US/UK",
         "title": "Ukraine story", "url": "https://bbc/2"},
        {"source": "Fox News - Politics", "lean": "right", "market": "US/UK",
         "title": "Ukraine story", "url": "https://fox/2"},
    ]
    subset = deduplication.comparison_subset(cross_market_cluster)
    assert subset is not None, "FAIL: BBC+Fox US/UK pair should still qualify within the cluster"
    assert all(a["market"] == "US/UK" for a in subset), \
        "FAIL: cross-market ANSA article leaked into the qualifying subset"
    print("PASS: cross-market cluster correctly trims to just the US/UK subset.")

    # _validate_bullet: malformed tag (not at end of line) must be rejected.
    bad_bullet = "- Tema: qualcosa. [1] ANSA dice X, Repubblica dice Y, con altro testo dopo."
    assert comparison._validate_bullet(bad_bullet, it_cluster) is None, \
        "FAIL: mid-sentence citation tag was accepted"
    print("PASS: malformed (non-trailing) citation tag correctly rejected.")

    # _validate_bullet: well-formed tag citing only 1 of 2 -- not enough
    # diversity on its own -- must be rejected.
    single_cite_bullet = "- Tema: qualcosa. Repubblica dice X. [1]"
    assert comparison._validate_bullet(single_cite_bullet, it_cluster) is None, \
        "FAIL: a bullet citing only one outlet was accepted as a comparison"
    print("PASS: bullet citing only one outlet correctly rejected.")

    # _validate_bullet: well-formed tag citing both -- must be accepted.
    good_bullet = "- Tema: qualcosa. Repubblica dice X, Libero dice Y. [1,2]"
    assert comparison._validate_bullet(good_bullet, it_cluster) == good_bullet, \
        "FAIL: a valid two-outlet citation was rejected"
    print("PASS: well-formed, genuinely diverse citation correctly accepted.")


if __name__ == "__main__":
    test_classification_drain_guard()
    test_comparison_subset_and_validation()
    print("\nAll v2.1.3 checks passed.")
