from app.evaluation.citation_evaluator import CitationEvaluator


def test_extract_citations():
    evaluator = CitationEvaluator()

    answer = (
        "FastAPI is a modern Python web framework. "
        "[Source: evaluation.txt, chunk 1]"
    )

    citations = evaluator.extract_citations(answer)

    assert citations == [
        {
            "source": "evaluation.txt",
            "chunk_id": 1,
        }
    ]


def test_extract_pdf_citation_with_page():
    evaluator = CitationEvaluator()

    answer = (
        "FastAPI provides automatic API documentation. "
        "[Source: sample.pdf, page 3, chunk 2]"
    )

    citations = evaluator.extract_citations(answer)

    assert citations == [
        {
            "source": "sample.pdf",
            "page": 3,
            "chunk_id": 2,
        }
    ]


def test_valid_citation():
    evaluator = CitationEvaluator()

    answer = (
        "FastAPI is a Python web framework. "
        "[Source: evaluation.txt, chunk 1]"
    )

    retrieved_sources = [
        {
            "source": "evaluation.txt",
            "chunk_id": 1,
        }
    ]

    result = evaluator.evaluate(
        answer,
        retrieved_sources
    )

    assert result["citation_count"] == 1
    assert result["valid_citations"] == 1
    assert result["invalid_citations"] == 0
    assert result["citation_precision"] == 1.0


def test_invalid_citation():
    evaluator = CitationEvaluator()

    answer = (
        "FastAPI is a Python web framework. "
        "[Source: evaluation.txt, chunk 99]"
    )

    retrieved_sources = [
        {
            "source": "evaluation.txt",
            "chunk_id": 1,
        }
    ]

    result = evaluator.evaluate(
        answer,
        retrieved_sources
    )

    assert result["citation_count"] == 1
    assert result["valid_citations"] == 0
    assert result["invalid_citations"] == 1
    assert result["citation_precision"] == 0.0


def test_answer_without_citation():
    evaluator = CitationEvaluator()

    answer = "FastAPI is a Python web framework."

    result = evaluator.evaluate(
        answer,
        retrieved_sources=[]
    )

    assert result["citation_count"] == 0
    assert result["valid_citations"] == 0
    assert result["invalid_citations"] == 0
    assert result["citation_precision"] is None
    assert result["has_citation"] is False