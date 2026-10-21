from app.rag.evaluation import EvaluationQuery, evaluate_retrieval
from app.rag.kuphaldt import DocumentChunk


def _chunk(chunk_id: str, title: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document",
        volume="Electrical Reference",
        chapter_number=1,
        chapter_title="Reference",
        section_title=title,
        content=content,
        processed_content=content,
        source_path="reference.html",
        source_url="file:///reference.html",
        token_count=len(content.split()),
        chunk_index=0,
        parent_section_index=0,
    )


def test_lexical_evaluation_reports_recall_and_mrr():
    chunks = [
        _chunk("kvl", "Kirchhoff's Laws", "Kirchhoff voltage law says loop voltages sum to zero."),
        _chunk("noise", "Capacitor sizing equation", "Capacitance depends on charge and voltage."),
    ]
    queries = [
        EvaluationQuery(
            question="What does Kirchhoff's Voltage Law state?",
            reference_answer="Loop voltages sum to zero.",
            relevant_source_sections=["Electrical Reference|1|Kirchhoff's Laws"],
        )
    ]

    evaluations, metrics = evaluate_retrieval(chunks, queries)

    assert evaluations[0].retrieved_chunk_ids[0] == "kvl"
    assert metrics == {"queries": 1.0, "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr_at_10": 1.0}


def test_evaluation_rejects_unmatched_source_section():
    query = EvaluationQuery(
        question="What is resonance?",
        reference_answer="A circuit condition.",
        relevant_source_sections=["Electrical Reference|1|Missing"],
    )

    try:
        evaluate_retrieval([_chunk("one", "Existing", "existing content")], [query])
    except ValueError as exc:
        assert "No chunks matched" in str(exc)
    else:
        raise AssertionError("unmatched evaluation section was accepted")
