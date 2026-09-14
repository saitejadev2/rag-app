from pathlib import Path

from pypdf import PdfReader

from ingestion.models import Document


def load_txt(file_path: str) -> list[Document]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    return [
        Document(
            text=text,
            metadata={
                "source": path.name,
            }
        )
    ]


def load_pdf(file_path: str) -> list[Document]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    reader = PdfReader(str(path))

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source": path.name,
                        "page": page_number,
                    }
                )
            )

    return documents


def load_document(file_path: str) -> list[Document]:

    path = Path(file_path)

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_txt(file_path)

    if suffix == ".pdf":
        return load_pdf(file_path)

    raise ValueError(
        f"Unsupported file type: {suffix}"
    )