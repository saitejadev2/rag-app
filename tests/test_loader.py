from pathlib import Path

from app.ingestion.loader import load_document
from app.ingestion.models import Document


def test_load_txt(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "This is a test document.",
        encoding="utf-8"
    )

    documents = load_document(str(file_path))

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].text == "This is a test document."
    assert documents[0].metadata["source"] == "sample.txt"


def test_unsupported_file_type(tmp_path: Path):
    file_path = tmp_path / "sample.docx"
    file_path.write_text(
        "test",
        encoding="utf-8"
    )

    try:
        load_document(str(file_path))
        assert False, "Expected ValueError"
    except ValueError as error:
        assert "Unsupported file type" in str(error)