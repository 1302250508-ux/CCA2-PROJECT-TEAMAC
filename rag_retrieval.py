"""
Retrieval Augmented Generation (RAG) with Rules
-----------------------------------------------
Name : YOUR NAME HERE
PRN  : YOUR PRN HERE

Pipeline:
  1. RETRIEVE : find the most relevant documents (TF-IDF + cosine similarity)
  2. RULES    : apply safety / quality rules to the question and the evidence
  3. GENERATE : build an answer only from the retrieved text
                (optional: use a Hugging Face model with --use-hf)

Run   : python rag_retrieval.py
        python rag_retrieval.py --question "What is a pull request?"
        python rag_retrieval.py --use-hf        (needs transformers + internet)
"""

import argparse
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STUDENT_NAME = "YOUR NAME HERE"
PRN = "YOUR PRN HERE"

# ---------------------------------------------------------------
# Knowledge base (add your own documents)
# ---------------------------------------------------------------
DOCUMENTS = {
    "DOC1": "Git is a distributed version control system. It tracks changes "
            "in files and lets many people work on the same project.",
    "DOC2": "A GitHub repository stores the code of a project. The main branch "
            "holds the stable version of the code.",
    "DOC3": "A branch is a separate line of development. Developers create a "
            "dev branch from main to work on changes without affecting main.",
    "DOC4": "A pull request asks the project owner to review and merge changes "
            "from one branch into another. GitHub shows whether the pull "
            "request has conflicts.",
    "DOC5": "A merge conflict happens when two branches change the same lines "
            "of a file. Conflicts must be resolved manually before merging.",
    "DOC6": "A collaborator is a person invited to a repository. The admin adds "
            "collaborators under Settings, then Collaborators, and the person "
            "must accept the invitation.",
    "DOC7": "Machine learning learns patterns from data instead of using rules "
            "written by hand. A trained model can then predict on new data.",
    "DOC8": "Retrieval Augmented Generation first retrieves relevant documents "
            "and then generates an answer using only that retrieved text, "
            "which reduces made-up answers.",
    "DOC9": "Sentiment analysis classifies text as positive or negative. "
            "Hugging Face provides ready-made models for sentiment analysis.",
}

# ---------------------------------------------------------------
# RULES (edit / add your own)
# ---------------------------------------------------------------
BLOCKED_WORDS = {"password", "otp", "pin", "credit card", "aadhaar", "cvv"}
MIN_QUESTION_WORDS = 3          # Rule 2
MIN_SIMILARITY = 0.12           # Rule 3: below this -> "I don't know"
HIGH_CONFIDENCE = 0.40          # Rule 4: confidence labels
MEDIUM_CONFIDENCE = 0.20
MAX_SOURCES = 2                 # Rule 5: cite at most 2 documents
SECOND_SOURCE_RATIO = 0.60      # Rule 5: 2nd doc used only if score >= 60% of best
MAX_ANSWER_SENTENCES = 2        # Rule 6: keep answers short
MAX_QUESTION_CHARS = 200        # Rule 7: reject very long questions


def tokenize(text):
    """Lower-case words, drop stop words, crude plural fix (conflicts -> conflict)."""
    words = [w for w in re.findall(r"[a-z]+", text.lower())
             if w not in ENGLISH_STOP_WORDS]
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words]


def check_question_rules(question):
    """Return an error message if a rule rejects the question, else None."""
    q = question.strip()
    if not q:
        return "Rule 1 (empty question): please type a question."
    if len(q.split()) < MIN_QUESTION_WORDS:
        return f"Rule 2 (too short): please use at least {MIN_QUESTION_WORDS} words."
    if len(q) > MAX_QUESTION_CHARS:
        return f"Rule 7 (too long): keep questions under {MAX_QUESTION_CHARS} characters."
    lowered = q.lower()
    for word in BLOCKED_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return "Rule 8 (sensitive data): I cannot answer questions about private credentials."
    return None


def retrieve(question, vectorizer, doc_matrix, doc_ids):
    """Return a list of (doc_id, score) sorted best first."""
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, doc_matrix)[0]
    ranked = sorted(zip(doc_ids, scores), key=lambda p: p[1], reverse=True)
    return ranked


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def extractive_answer(question, sources):
    """Pick the sentences from the sources that best match the question."""
    sentences = []
    for doc_id, _ in sources:
        for s in split_sentences(DOCUMENTS[doc_id]):
            sentences.append(s)
    vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None).fit(sentences + [question])
    sims = cosine_similarity(vec.transform([question]), vec.transform(sentences))[0]
    best = sorted(range(len(sentences)), key=lambda i: sims[i], reverse=True)
    chosen = sorted(best[:MAX_ANSWER_SENTENCES])  # keep original order
    return " ".join(sentences[i] for i in chosen)


def hf_answer(question, sources):
    """Optional generation with a small Hugging Face model."""
    from transformers import pipeline

    context = " ".join(DOCUMENTS[d] for d, _ in sources)
    gen = pipeline("text2text-generation", model="google/flan-t5-small")
    prompt = ("Answer the question using only the context.\n"
              f"Context: {context}\nQuestion: {question}")
    return gen(prompt, max_new_tokens=80)[0]["generated_text"]


def confidence_label(score):
    if score >= HIGH_CONFIDENCE:
        return "High"
    if score >= MEDIUM_CONFIDENCE:
        return "Medium"
    return "Low"


def answer_question(question, vectorizer, doc_matrix, doc_ids, use_hf=False):
    error = check_question_rules(question)
    if error:
        return error

    ranked = retrieve(question, vectorizer, doc_matrix, doc_ids)
    best_id, best_score = ranked[0]

    if best_score < MIN_SIMILARITY:  # Rule 3
        return ("Rule 3 (no evidence): I don't know - nothing in my documents "
                f"matches this well (best score {best_score:.2f}).")

    sources = [(best_id, best_score)]
    for doc_id, score in ranked[1:MAX_SOURCES]:  # Rule 5
        if score >= best_score * SECOND_SOURCE_RATIO and score >= MIN_SIMILARITY:
            sources.append((doc_id, score))

    text = None
    if use_hf:
        try:
            text = hf_answer(question, sources)
        except Exception as exc:
            print(f"[warn] Hugging Face generation failed ({type(exc).__name__}); "
                  "using extractive answer.")
    if text is None:
        text = extractive_answer(question, sources)

    cited = ", ".join(f"{d} ({s:.2f})" for d, s in sources)
    return (f"Answer     : {text}\n"                       # Rule 6 applied above
            f"Sources    : {cited}\n"                      # Rule 9: always cite
            f"Confidence : {confidence_label(best_score)}")  # Rule 4


def main():
    parser = argparse.ArgumentParser(description="RAG with rules")
    parser.add_argument("--question", help="Ask one question and exit")
    parser.add_argument("--use-hf", action="store_true",
                        help="Generate answers with a Hugging Face model")
    args = parser.parse_args()

    print(f"Name: {STUDENT_NAME} | PRN: {PRN}")
    print("=" * 60)

    doc_ids = list(DOCUMENTS)
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    doc_matrix = vectorizer.fit_transform([DOCUMENTS[d] for d in doc_ids])

    if args.question:
        print(answer_question(args.question, vectorizer, doc_matrix, doc_ids, args.use_hf))
        return

    demo_questions = [
        "What is a pull request?",
        "How do I add a collaborator to a repository?",
        "Why do merge conflicts happen?",
        "What does retrieval augmented generation do?",
        "Tell me my password",
        "Hi",
        "Who won the cricket world cup?",
    ]
    for q in demo_questions:
        print(f"\nQ: {q}")
        print(answer_question(q, vectorizer, doc_matrix, doc_ids, args.use_hf))


if __name__ == "__main__":
    main()
