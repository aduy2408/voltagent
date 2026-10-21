from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.rag.kuphaldt import DocumentChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from", "how",
    "in", "is", "of", "on", "or", "state", "the", "to", "what", "which", "with",
}


class EvaluationQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    reference_answer: str
    relevant_source_sections: list[str] = Field(min_length=1)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class QueryEvaluation:
    question: str
    relevant_chunk_ids: frozenset[str]
    retrieved_chunk_ids: tuple[str, ...]
    reciprocal_rank: float
    recall_at_5: float
    recall_at_10: float


class LexicalRetriever:
    """Small deterministic lexical baseline for evaluating corpus chunk discoverability."""

    def __init__(self, chunks: Sequence[DocumentChunk]):
        self.chunks = tuple(chunks)
        self._terms: dict[str, tuple[str, ...]] = {}
        document_frequency: Counter[str] = Counter()
        for chunk in self.chunks:
            terms = _tokens(f"{chunk.section_title} {chunk.content}")
            self._terms[chunk.chunk_id] = terms
            document_frequency.update(set(terms))
        total = len(self.chunks) or 1
        self._idf = {
            term: math.log((total + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        query_terms = Counter(_tokens(query))
        scored: list[RetrievedChunk] = []
        for chunk in self.chunks:
            terms = Counter(self._terms[chunk.chunk_id])
            score = sum(
                query_frequency * terms[term] * self._idf.get(term, 1.0)
                for term, query_frequency in query_terms.items()
                if terms[term]
            )
            if score:
                scored.append(RetrievedChunk(chunk.chunk_id, score))
        scored.sort(key=lambda item: (-item.score, item.chunk_id))
        return scored[:top_k]


def evaluate_retrieval(
    chunks: Sequence[DocumentChunk], queries: Iterable[EvaluationQuery], candidate_k: int = 10
) -> tuple[list[QueryEvaluation], dict[str, float]]:
    if candidate_k < 10:
        raise ValueError("candidate_k must be at least 10 for Recall@10 and MRR@10")
    retriever = LexicalRetriever(chunks)
    evaluations: list[QueryEvaluation] = []
    for query in queries:
        relevant = frozenset(
            chunk.chunk_id
            for chunk in chunks
            if _section_key(chunk.volume, chunk.chapter_number, chunk.section_title)
            in set(query.relevant_source_sections)
        )
        if not relevant:
            raise ValueError(f"No chunks matched evaluation sections for: {query.question}")
        retrieved = tuple(item.chunk_id for item in retriever.search(query.question, candidate_k))
        rank = next((index + 1 for index, chunk_id in enumerate(retrieved) if chunk_id in relevant), None)
        evaluations.append(
            QueryEvaluation(
                question=query.question,
                relevant_chunk_ids=relevant,
                retrieved_chunk_ids=retrieved,
                reciprocal_rank=1.0 / rank if rank else 0.0,
                recall_at_5=float(bool(set(retrieved[:5]) & relevant)),
                recall_at_10=float(bool(set(retrieved[:10]) & relevant)),
            )
        )
    count = len(evaluations) or 1
    metrics = {
        "queries": float(len(evaluations)),
        "recall_at_5": sum(item.recall_at_5 for item in evaluations) / count,
        "recall_at_10": sum(item.recall_at_10 for item in evaluations) / count,
        "mrr_at_10": sum(item.reciprocal_rank for item in evaluations) / count,
    }
    return evaluations, metrics


def _section_key(volume: str, chapter_number: int, section_title: str) -> str:
    return f"{volume}|{chapter_number}|{section_title}"


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in _TOKEN_RE.findall(text.casefold()) if token not in _STOPWORDS)
