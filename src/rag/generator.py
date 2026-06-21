from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def has_llm_key() -> bool:
    return bool(os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _format_sources(retrieved_chunks: list[dict]) -> list[dict]:
    sources = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        sources.append(
            {
                "source_id": f"S{index}",
                "source": metadata.get("source", "unknown"),
                "document_type": metadata.get("document_type", "unknown"),
                "sensitivity_level": metadata.get("sensitivity_level", "unknown"),
                "score": round(float(chunk.get("score", 0.0)), 4),
            }
        )
    return sources


def _context_from_chunks(retrieved_chunks: list[dict], max_chars: int = 6000) -> str:
    context_parts = []
    remaining = max_chars
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        citation = f"S{index}: {metadata.get('source', 'unknown')} | {metadata.get('document_type', 'unknown')}"
        text = " ".join(chunk.get("text", "").split())
        block = f"[{citation}]\n{text}"
        if len(block) > remaining:
            block = block[:remaining]
        if not block:
            break
        context_parts.append(block)
        remaining -= len(block)
        if remaining <= 0:
            break
    return "\n\n".join(context_parts)


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int = 18) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _call_groq(query: str, retrieved_chunks: list[dict]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    context = _context_from_chunks(retrieved_chunks)
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer only from the provided automotive context. "
                    "Cite sources using [S1], [S2], etc. If the answer is not in the context, say so."
                ),
            },
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
        ],
        "temperature": 0.1,
        "max_tokens": 500,
    }
    response = _post_json(GROQ_CHAT_COMPLETIONS_URL, payload, {"Authorization": f"Bearer {api_key}"})
    return response["choices"][0]["message"]["content"].strip()


def _call_gemini(query: str, retrieved_chunks: list[dict]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    context = _context_from_chunks(retrieved_chunks)
    url = GEMINI_GENERATE_URL.format(model=GEMINI_MODEL, key=api_key)
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Answer only from the provided automotive context. "
                            "Cite sources using [S1], [S2], etc. If the answer is not in the context, say so.\n\n"
                            f"Question: {query}\n\nContext:\n{context}"
                        )
                    }
                ]
            }
        ],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500},
    }
    response = _post_json(url, payload, {})
    return response["candidates"][0]["content"]["parts"][0]["text"].strip()


def generate_answer(query: str, retrieved_chunks: list[dict], max_chars: int = 1600) -> dict[str, object]:
    if not retrieved_chunks:
        return {
            "mode": "extractive_no_sources",
            "answer_mode": "extractive",
            "answer": (
                "No accessible source chunks matched this query for the selected role. "
                "No answer was generated."
            ),
            "sources": [],
        }

    answer_parts = []
    used_sources = []
    remaining = max_chars

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        citation = f"[S{index}: {metadata.get('source', 'unknown')} | {metadata.get('document_type', 'unknown')}]"
        text = " ".join(chunk.get("text", "").split())
        excerpt = text[: max(0, remaining - len(citation) - 4)]
        if not excerpt:
            break
        answer_parts.append(f"{excerpt} {citation}")
        remaining -= len(excerpt) + len(citation) + 4
        used_sources.append(
            {
                "source_id": f"S{index}",
                "source": metadata.get("source", "unknown"),
                "document_type": metadata.get("document_type", "unknown"),
                "sensitivity_level": metadata.get("sensitivity_level", "unknown"),
                "score": round(float(chunk.get("score", 0.0)), 4),
            }
        )
        if remaining <= 0:
            break

    mode = "extractive"
    prefix = "Extractive answer from retrieved source chunks only. No LLM generation was used:\n\n"

    return {"mode": mode, "answer_mode": "extractive", "answer": prefix + "\n\n".join(answer_parts), "sources": used_sources}


def generate_grounded_answer(query: str, retrieved_chunks: list[dict], max_chars: int = 1600) -> dict[str, object]:
    if not retrieved_chunks:
        return generate_answer(query, retrieved_chunks, max_chars=max_chars)

    sources = _format_sources(retrieved_chunks)
    errors = []

    try:
        groq_answer = _call_groq(query, retrieved_chunks)
        if groq_answer:
            return {"answer_mode": "groq", "mode": "groq", "answer": groq_answer, "sources": sources, "fallback_errors": []}
    except Exception as exc:
        errors.append(f"groq: {type(exc).__name__}")

    try:
        gemini_answer = _call_gemini(query, retrieved_chunks)
        if gemini_answer:
            return {
                "answer_mode": "gemini",
                "mode": "gemini",
                "answer": gemini_answer,
                "sources": sources,
                "fallback_errors": errors,
            }
    except Exception as exc:
        errors.append(f"gemini: {type(exc).__name__}")

    extractive = generate_answer(query, retrieved_chunks, max_chars=max_chars)
    extractive["fallback_errors"] = errors
    return extractive
