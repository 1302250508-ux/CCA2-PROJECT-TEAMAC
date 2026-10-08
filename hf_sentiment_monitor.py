"""
Hugging Face Sentiment Monitor
------------------------------
Name : YOUR NAME HERE
PRN  : YOUR PRN HERE

Monitors the sentiment of text for several problem statements and raises an
ALERT when the share of negative messages crosses a threshold.

Uses the Hugging Face `transformers` sentiment-analysis pipeline.
If `transformers` is not installed (or the model cannot be downloaded) the
script falls back to a tiny keyword model so it still runs.

Run   : python hf_sentiment_monitor.py
        python hf_sentiment_monitor.py --text "The delivery was very late"
"""

import argparse

STUDENT_NAME = "YOUR NAME HERE"
PRN = "YOUR PRN HERE"

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# ---------------------------------------------------------------
# Problem statements: name -> (alert threshold on % negative, texts)
# Add / change problem statements to make this your own.
# ---------------------------------------------------------------
PROBLEM_STATEMENTS = {
    "1. Product reviews (e-commerce)": (
        0.60,
        [
            "The phone battery lasts two days, absolutely love it.",
            "Screen cracked within a week, terrible build quality.",
            "Great value for money and fast delivery.",
            "Stopped working after a month, very disappointed.",
        ],
    ),
    "2. Customer support tickets": (
        0.50,
        [
            "Thanks for resolving my issue so quickly!",
            "I have been waiting for a reply for five days, this is unacceptable.",
            "The agent was polite and very helpful.",
            "Still not fixed and nobody is responding to me.",
            "Refund processed, appreciate the support.",
        ],
    ),
    "3. Social media brand mentions": (
        0.35,
        [
            "Loving the new update, the app feels so smooth.",
            "Worst update ever, the app keeps crashing.",
            "Amazing offers this festival season!",
            "Customer care is rude and useless.",
        ],
    ),
    "4. Student course feedback": (
        0.30,
        [
            "The lectures were engaging and the examples were clear.",
            "Assignments were confusing and the deadlines were unfair.",
            "I learned a lot and enjoyed the project work.",
            "The pace was too fast, I could not follow anything.",
            "Excellent faculty support during labs.",
        ],
    ),
    "5. Food delivery app reviews": (
        0.40,
        [
            "Food arrived hot and the packaging was neat.",
            "Delivery was an hour late and the food was cold.",
            "Delicious meal, will order again.",
            "Wrong items delivered and no refund offered.",
        ],
    ),
    "6. Hospital / clinic feedback": (
        0.25,
        [
            "The doctor listened carefully and explained everything.",
            "Waiting time was far too long and the staff were rude.",
            "Clean rooms and caring nurses.",
            "Billing was confusing and full of hidden charges.",
        ],
    ),
}

# Fallback word lists (only used when transformers is unavailable)
POSITIVE_WORDS = {
    "love", "great", "good", "excellent", "amazing", "helpful", "polite",
    "thanks", "appreciate", "delicious", "smooth", "clean", "caring",
    "engaging", "enjoyed", "learned", "neat", "hot", "clear", "quickly",
}
NEGATIVE_WORDS = {
    "terrible", "worst", "disappointed", "unacceptable", "rude", "useless",
    "crashing", "late", "cold", "wrong", "confusing", "unfair", "cracked",
    "stopped", "waiting", "fast", "hidden", "nobody", "still", "long", "refund",
}


def keyword_classifier(texts):
    """Very small offline fallback: counts positive vs negative words."""
    results = []
    for text in texts:
        words = {w.strip(".,!?").lower() for w in text.split()}
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        label = "NEGATIVE" if neg > pos else "POSITIVE"
        total = max(pos + neg, 1)
        score = 0.5 + 0.5 * abs(pos - neg) / total
        results.append({"label": label, "score": round(score, 4)})
    return results


def load_classifier():
    """Return a function texts -> list of {label, score}."""
    try:
        from transformers import pipeline

        pipe = pipeline("sentiment-analysis", model=MODEL_NAME)
        print(f"[info] Using Hugging Face model: {MODEL_NAME}")
        return lambda texts: pipe(list(texts))
    except Exception as exc:  # transformers missing or no internet
        print(f"[warn] Hugging Face model unavailable ({type(exc).__name__}); "
              "using keyword fallback.")
        return keyword_classifier


def monitor(classify, name, threshold, texts):
    results = classify(texts)
    negatives = sum(1 for r in results if r["label"].upper().startswith("NEG"))
    neg_ratio = negatives / len(texts)
    avg_conf = sum(r["score"] for r in results) / len(results)
    status = "ALERT" if neg_ratio > threshold else "OK"

    print(f"\n{name}")
    print("-" * len(name))
    for text, r in zip(texts, results):
        print(f"  [{r['label']:<8} {r['score']:.2f}] {text}")
    print(f"  Negative: {negatives}/{len(texts)} = {neg_ratio:.0%} "
          f"(threshold {threshold:.0%}) | avg confidence {avg_conf:.2f}")
    print(f"  STATUS: {status}")
    return status


def main():
    parser = argparse.ArgumentParser(description="Hugging Face Sentiment Monitor")
    parser.add_argument("--text", help="Classify one custom sentence and exit")
    args = parser.parse_args()

    print(f"Name: {STUDENT_NAME} | PRN: {PRN}")
    print("=" * 60)
    classify = load_classifier()

    if args.text:
        result = classify([args.text])[0]
        print(f"\n'{args.text}' -> {result['label']} ({result['score']:.2f})")
        return

    statuses = {}
    for name, (threshold, texts) in PROBLEM_STATEMENTS.items():
        statuses[name] = monitor(classify, name, threshold, texts)

    print("\n=== SUMMARY ===")
    for name, status in statuses.items():
        print(f"  {status:<5} {name}")


if __name__ == "__main__":
    main()
