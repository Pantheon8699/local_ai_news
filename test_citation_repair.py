"""
Check for core.summarizer.repair_citations(), on the Oct 12, 2026 failure:
the reduce model numbered its bullets one off, so each bullet was rendered
above another story's sources. No Ollama needed: `python test_citation_repair.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.summarizer import _reduce_batch, repair_citations, unsupported_numbers

SUMMARIES = [
    "Il presidente della Comunità ebraica di Milano, Walker Meghnagi, ha accusato il Comune di Reggio Emilia "
    "di metodi da Terzo Reich. Il sindaco Marco Massari ha respinto queste accuse.",
    "Laika ha realizzato un'opera d'arte intitolata \"Gli Intoccabili\" a Roma, nel ghetto ebraico, alla "
    "vigilia del voto sul Ddl Antisemitismo.",
    "Oggi lunedì 12 ottobre si verifica un sciopero di 24 ore in tutti gli stabilimenti ex Ilva per ogni "
    "turno, convocato dai sindacati Fiom, Fim e Uilm dopo l'esito dell'incontro a Palazzo Chigi.",
    "Sara Kelany di Fratelli d'Italia ha affermato che grazie a Giorgia Meloni l'Italia è forte in Europa.",
]


def test_repair_citations():
    drifted = (
        "- Meghnagi accusa Reggio Emilia di antisemitismo, Massari respinge le accuse. [1,2]\n"
        "- Laika esegue opera d'arte \"Gli Intoccabili\" a Roma, vicino al voto sul Ddl Antisemitismo. [3]\n"
        "- Sciopero di 24 ore in ex Ilva per ogni turno dopo l'incontro a Palazzo Chigi. [4]"
    )
    lines = repair_citations(drifted, SUMMARIES).splitlines()
    assert lines[0].endswith("accuse. [1]"), lines[0]
    assert lines[1].endswith("Antisemitismo. [2]"), lines[1]
    assert lines[2].endswith("Chigi. [3]"), lines[2]
    # Summary 4 lost its bullet to the drift: it comes back as its own.
    assert lines[3] == f"- {SUMMARIES[3]} [4]", lines[3]
    assert len(lines) == 4

    # Correct tags, including a real two-source merge, are left alone.
    correct = (
        "- Meghnagi accusa Reggio Emilia di metodi da Terzo Reich, Massari respinge; a Roma Laika realizza "
        "\"Gli Intoccabili\" alla vigilia del voto sul Ddl Antisemitismo. [1,2]\n"
        "- Sciopero di 24 ore in ex Ilva convocato da Fiom, Fim e Uilm. [3]\n"
        "- Kelany afferma che grazie a Meloni l'Italia è forte in Europa. [4]"
    )
    # The real merge [1,2] keeps the model's wording; a one-source bullet
    # becomes its summary verbatim (Oct 19, 2026: rewording added errors).
    expected = correct.splitlines()[0] + "".join(f"\n- {SUMMARIES[i]} [{i + 1}]" for i in (2, 3))
    assert repair_citations(correct, SUMMARIES) == expected

    # One story split into two bullets citing the same summary: shown once.
    split = correct + "\n- Kelany: l'Italia è forte in Europa grazie a Meloni. [4]"
    assert repair_citations(split, SUMMARIES) == expected

    # The prompt's own example sentence, copied into the output, is cut.
    leaked = correct + "\n- Il PIL è cresciuto del 2% nel secondo trimestre. [4]"
    assert repair_citations(leaked, SUMMARIES) == expected


def test_unsupported_numbers():
    article = "Republicans could lose the Senate in the November midterms. 74,000 people, $20 million, six seats."
    assert unsupported_numbers("Elezioni di novembre 2022, nuovo Senato a gennaio 2023.", article) == ["2022", "2023"]
    assert unsupported_numbers("74.000 persone, 20 milioni di dollari, 6 seggi.", article) == []


def test_title_only_articles_skip_the_model():
    placeholder = "Testo completo non disponibile per questo articolo; titolo originale: «Titolo»."
    # No model call at all when there is nothing but titles: this must not need Ollama.
    assert _reduce_batch("Italy", [placeholder], "{category}{count}{summaries}") == f"- {placeholder} [1]"


if __name__ == "__main__":
    test_repair_citations()
    test_unsupported_numbers()
    test_title_only_articles_skip_the_model()
    print("ok")
