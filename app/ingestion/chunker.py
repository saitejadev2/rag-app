from app.ingestion.models import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:

    chunked_documents = []

    for document in documents:

        chunks = recursive_split(
            document.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for chunk_id, chunk in enumerate(chunks):

            metadata = document.metadata.copy()

            metadata["chunk_id"] = chunk_id

            chunked_documents.append(
                Document(
                    text=chunk,
                    metadata=metadata
                )
            )

    return chunked_documents

def recursive_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:

    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    if not text.strip():
        return []

    def split_text(text: str, separator_index: int) -> list[str]:

        # If text already fits, we're done.
        if len(text) <= chunk_size:
            return [text.strip()]

        # If we've reached the final separator, split by characters.
        if separator_index >= len(separators):
            return [
                text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]

        separator = separators[separator_index]

        # Try the current separator.
        if separator:
            pieces = text.split(separator)
        else:
            pieces = list(text)

        chunks = []
        current = ""

        for piece in pieces:

            candidate = (
                current + separator + piece
                if current
                else piece
            )

            if len(candidate) <= chunk_size:
                current = candidate

            else:
                if current:
                    chunks.append(current.strip())

                # If this individual piece is still too large,
                # recursively split it using the next separator.
                if len(piece) > chunk_size:
                    smaller_chunks = split_text(
                        piece,
                        separator_index + 1
                    )

                    chunks.extend(smaller_chunks)
                    current = ""
                else:
                    current = piece

        if current:
            chunks.append(current.strip())

        return chunks

    chunks = split_text(text, 0)

    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk]

    # Add overlap
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []

        for i, chunk in enumerate(chunks):

            if i == 0:
                overlapped_chunks.append(chunk)
                continue

            previous = chunks[i - 1]

            overlap_text = previous[-chunk_overlap:]

            overlapped_chunks.append(
                overlap_text + " " + chunk
            )

        chunks = overlapped_chunks

    return chunks