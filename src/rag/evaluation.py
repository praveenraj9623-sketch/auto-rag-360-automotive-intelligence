from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.rag.retriever import build_default_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = PROJECT_ROOT / "data" / "evaluation" / "golden_questions.csv"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "reports" / "rag_eval_report.json"


def _role_for_question(question: str, expected_component: str) -> str:
    normalized = f"{question} {expected_component}".lower()
    if "supplier" in normalized:
        return "Supplier"
    if "service manual" in normalized or "diagnostic" in normalized:
        return "Service Engineer"
    if "quality" in normalized or "policy" in normalized or "triage" in normalized:
        return "Quality Engineer"
    return "Executive"


def _chunk_matches(chunk: dict, expected_keyword: str) -> bool:
    return expected_keyword.lower() in chunk.get("text", "").lower()


def _precision_at_k(results: list[dict], expected_keyword: str, k: int) -> float:
    top_results = results[:k]
    if not top_results:
        return 0.0
    matches = sum(1 for result in top_results if _chunk_matches(result, expected_keyword))
    return matches / k


def run_golden_question_evaluation(top_k: int = 5, min_similarity: float = 0.05) -> dict:
    golden = pd.read_csv(GOLDEN_PATH)
    retriever = build_default_retriever()
    details = []

    for _, row in golden.iterrows():
        question = str(row["question"])
        expected_keyword = str(row["expected_keyword"])
        role = _role_for_question(question, str(row["expected_component"]))

        start = time.perf_counter()
        retrieval = retriever.retrieve(query=question, role=role, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        results = retrieval["results"]

        rank = None
        for index, result in enumerate(results, start=1):
            if _chunk_matches(result, expected_keyword):
                rank = index
                break

        max_score = max((float(result.get("score", 0.0)) for result in results), default=0.0)
        no_answer = max_score < min_similarity
        detail = {
            "question": question,
            "role": role,
            "expected_component": row["expected_component"],
            "expected_make": row["expected_make"],
            "expected_keyword": expected_keyword,
            "hit": rank is not None,
            "rank": rank,
            "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
            "precision_at_3": _precision_at_k(results, expected_keyword, 3),
            "precision_at_5": _precision_at_k(results, expected_keyword, 5),
            "retrieved_count": len(results),
            "max_similarity": max_score,
            "latency_ms": latency_ms,
            "no_answer": no_answer,
        }
        details.append(detail)

    total = max(1, len(details))
    report = {
        "top_k": top_k,
        "minimum_similarity_threshold": min_similarity,
        "question_count": len(details),
        "precision_at_3": sum(item["precision_at_3"] for item in details) / total,
        "precision_at_5": sum(item["precision_at_5"] for item in details) / total,
        "hit_rate": sum(1 for item in details if item["hit"]) / total,
        "mean_reciprocal_rank": sum(item["reciprocal_rank"] for item in details) / total,
        "average_latency_ms": sum(item["latency_ms"] for item in details) / total,
        "no_answer_rate": sum(1 for item in details if item["no_answer"]) / total,
        "details": details,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    print(json.dumps(run_golden_question_evaluation(), indent=2))


if __name__ == "__main__":
    main()
