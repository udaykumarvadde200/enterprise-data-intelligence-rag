import json
from pathlib import Path

from langchain_core.documents import Document

from app.evaluation.dataset import load_evaluation_cases
from app.evaluation.runner import evaluate_retriever
from app.rag.bm25_retriever import BM25DocumentRetriever
from app.rag.rrf import ReciprocalRankFusion


BASE_DIR = Path(__file__).resolve().parents[2]
CORPUS_PATH = BASE_DIR / "data" / "evaluation" / "synthetic_policy_corpus.json"
CASES_PATH = BASE_DIR / "data" / "evaluation" / "retrieval_cases.json"

# Fixed alternatives keep this experiment deterministic.
ALTERNATIVE_QUERIES = {
    "What is the customer security policy?": [
        "How should customer accounts and sensitive records be protected?",
        "What authentication and access controls protect customers?",
    ],
    "How should confidential customer information be handled?": [
        "How can private customer information be stored and shared safely?",
        "What rules govern disclosure of confidential records?",
    ],
    "What is the employee leave policy?": [
        "What are the rules for employee annual leave?",
        "How does the employee leave approval process work?",
    ],
    "How can employees request annual leave?": [
        "Where should employees submit annual leave requests?",
        "Who approves employee leave requests?",
    ],
    "What are the rules for protecting private information?": [
        "How should confidential customer records be protected?",
        "Who can access or receive private customer information?",
    ],
}


def main() -> None:
    with CORPUS_PATH.open("r", encoding="utf-8-sig") as file:
        corpus_data = json.load(file)

    documents = [
        Document(
            page_content=item["content"],
            metadata={"source": item["id"]},
        )
        for item in corpus_data
    ]

    retriever = BM25DocumentRetriever(documents)
    fusion = ReciprocalRankFusion(k=60)
    cases = load_evaluation_cases(CASES_PATH)

    def single_query(query: str) -> list[str]:
        results = retriever.search(query, k=5)
        return [str(doc.metadata["source"]) for doc in results]

    def multi_query(query: str) -> list[str]:
        queries = [query, *ALTERNATIVE_QUERIES.get(query, [])]
        ranked_lists = [
            retriever.search(expanded_query, k=5)
            for expanded_query in queries
        ]
        results = fusion.fuse(ranked_lists, top_k=5)
        return [str(doc.metadata["source"]) for doc in results]

    single = evaluate_retriever(cases, single_query, k=5)
    multi = evaluate_retriever(cases, multi_query, k=5)

    print("\n=== Controlled Retrieval Benchmark ===")
    print(f"Corpus documents: {len(documents)}")
    print(f"Evaluation queries: {len(cases)}")
    print("Retriever: BM25; multi-query fusion: RRF")
    print("Alternative queries: fixed and deterministic\n")

    print(f"{'Metric':<15}{'Single-query':>16}{'Multi-query':>16}")
    print("-" * 47)
    print(
        f"{'Recall@5':<15}"
        f"{single.recall_at_k:>16.3f}"
        f"{multi.recall_at_k:>16.3f}"
    )
    print(
        f"{'MRR@5':<15}"
        f"{single.mrr_at_k:>16.3f}"
        f"{multi.mrr_at_k:>16.3f}"
    )

    print("\n=== Per-query results ===")
    for case in cases:
        expected = sorted(case.relevant_ids)
        single_ids = single_query(case.query)
        multi_ids = multi_query(case.query)

        print(f"\nQuery: {case.query}")
        print(f"Relevant:      {expected}")
        print(f"Single-query:  {single_ids}")
        print(f"Multi-query:   {multi_ids}")


if __name__ == "__main__":
    main()
