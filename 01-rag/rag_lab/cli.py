import argparse
import json
import os

import numpy as np
from rag_lab.core import Index, citations_valid, load_chunks, prompt_for, source_recall_at_k

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def encoder(model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return lambda texts: np.asarray(model.encode(texts, convert_to_numpy=True, normalize_embeddings=True))


def main():
    parser = argparse.ArgumentParser(description="A citation-aware RAG lab")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--docs", default="docs")
    build.add_argument("--index", default="index.npz")
    build.add_argument("--model", default=DEFAULT_EMBEDDER)
    ask = sub.add_parser("ask")
    ask.add_argument("--index", default="index.npz")
    ask.add_argument("--question", required=True)
    ask.add_argument("--k", type=int, default=3)
    ask.add_argument("--generate", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--index", default="index.npz")
    ev.add_argument("--dataset", default="eval.json")
    ev.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    if args.command == "build":
        chunks = load_chunks(args.docs)
        vectors = encoder(args.model)([c.text for c in chunks])
        Index(chunks, vectors, args.model).save(args.index)
        print(f"Indexed {len(chunks)} chunks with {args.model}")
        return
    index = Index.load(args.index)
    embed = encoder(index.model)
    if args.command == "evaluate":
        with open(args.dataset, encoding="utf-8") as f:
            dataset = json.load(f)
        print(json.dumps({f"source_recall@{args.k}": source_recall_at_k(index, dataset, lambda q: embed([q])[0], args.k)}))
        return
    hits = index.search(embed([args.question])[0], args.k)
    for n, (_, score, c) in enumerate(hits, 1):
        print(f"[{n}] {c.source} page={c.page or '-'} score={score:.3f}: {c.text[:170]}")
    prompt = prompt_for(args.question, hits)
    if not args.generate:
        print("\nTo generate an answer, rerun with --generate and configure RAG_MODEL and OPENAI_API_KEY.")
        return
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("RAG_MODEL"):
        parser.error("--generate requires OPENAI_API_KEY and RAG_MODEL")
    from openai import OpenAI
    response = OpenAI().chat.completions.create(
        model=os.environ["RAG_MODEL"],
        messages=[{"role":"user","content":prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content or ""
    print("\n" + answer)
    if not citations_valid(answer, len(hits)):
        raise SystemExit("INVALID_CITATIONS: answer lacked valid evidence IDs")


if __name__ == "__main__":
    main()
