"""Corpus-level BLEU / chrF / chrF++ with sacreBLEU."""

from sacrebleu import corpus_bleu, corpus_chrf


def score(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Return BLEU, chrF (character n-grams only) and chrF++ (plus word bigrams)."""
    return {
        "BLEU": corpus_bleu(predictions, [references]).score,
        "chrF": corpus_chrf(predictions, [references]).score,
        "chrF++": corpus_chrf(predictions, [references], word_order=2).score,
    }


def print_scores(scores: dict[str, float]) -> None:
    print("\nFinal scores:")
    for name, value in scores.items():
        print(f"{name}: {value:.2f}")
