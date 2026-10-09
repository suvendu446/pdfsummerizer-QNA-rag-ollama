"""Small retrieval/answer evaluation to compare chunk sizes with real numbers.

1. Put 2-3 PDFs in data/.
2. Copy eval_questions.example.json to eval_questions.json and write 15-20 questions
   from those PDFs, each with keywords that MUST appear in the correct passage.
3. Run:  python evaluate.py --chunk-sizes 500 1000 2000 --k 4
   Add --answers to also check the generated answers (slower, since the local model answers every question).

Metric: retrieval hit rate = share of questions where at least one expected keyword
appears in the top-k retrieved chunks. It is a simple proxy, not a full evaluation.
"""
import argparse
import json
from pathlib import Path

import rag


def keyword_hit(texts, keywords) -> bool:
    haystack = " ".join(texts).lower()
    return any(k.lower() in haystack for k in keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[500, 1000, 2000])
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--answers", action="store_true", help="also evaluate generated answers")
    args = parser.parse_args()

    items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    print(f"{len(items)} questions, top-k={args.k}\n")
    header = "| chunk size | chunks | retrieval hit rate |"
    divider = "|---|---|---|"
    if args.answers:
        header += " answer keyword rate |"
        divider += "---|"
    print(header)
    print(divider)

    for size in args.chunk_sizes:
        overlap = int(size * args.overlap_ratio)
        index_dir = Path(f"faiss_index_eval_{size}")
        stats = rag.build_index(size, overlap, index_dir=index_dir)
        store = rag.load_index(index_dir)

        retrieval_hits = answer_hits = 0
        for item in items:
            results = rag.retrieve(store, item["question"], args.k)
            if keyword_hit([d.page_content for d, _ in results], item["expected_keywords"]):
                retrieval_hits += 1
            if args.answers:
                res = rag.answer_question(store, item["question"], k=args.k)
                if keyword_hit([res["answer"]], item["expected_keywords"]):
                    answer_hits += 1

        row = f"| {size} | {stats['chunks']} | {retrieval_hits}/{len(items)} ({retrieval_hits / len(items):.0%}) |"
        if args.answers:
            row += f" {answer_hits}/{len(items)} ({answer_hits / len(items):.0%}) |"
        print(row)


if __name__ == "__main__":
    main()
