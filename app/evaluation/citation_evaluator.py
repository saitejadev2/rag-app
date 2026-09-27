import re


class CitationEvaluator:

    CITATION_PATTERN = re.compile(
        r"\[Source:\s*"
        r"(?P<source>[^,\]]+)"
        r"(?:,\s*page\s+(?P<page>\d+))?"
        r",\s*chunk\s+(?P<chunk>\d+)"
        r"\]"
    )

    def extract_citations(self, answer: str):
        """
        Extract citations from a generated answer.

        Supported formats:

        [Source: evaluation.txt, chunk 0]

        [Source: sample.pdf, page 2, chunk 3]
        """

        matches = self.CITATION_PATTERN.finditer(answer)

        citations = []

        for match in matches:
            citation = {
                "source": match.group("source").strip(),
                "chunk_id": int(match.group("chunk")),
            }

            page = match.group("page")

            if page is not None:
                citation["page"] = int(page)

            citations.append(citation)

        return citations

    def evaluate(
        self,
        answer: str,
        retrieved_sources: list[dict]
    ):
        """
        Evaluate whether citations in the generated answer
        refer to chunks that were actually retrieved.
        """

        citations = self.extract_citations(answer)

        if not citations:
            return {
                "citations": [],
                "citation_count": 0,
                "valid_citations": 0,
                "invalid_citations": 0,
                "citation_precision": None,
                "has_citation": False,
            }

        valid_citations = 0
        invalid_citations = 0

        citation_results = []

        for citation in citations:

            matched = False

            for source in retrieved_sources:

                same_source = (
                    citation["source"]
                    == source.get("source")
                )

                same_chunk = (
                    citation["chunk_id"]
                    == source.get("chunk_id")
                )

                if "page" in citation:
                    same_page = (
                        citation["page"]
                        == source.get("page")
                    )
                else:
                    same_page = True

                if (
                    same_source
                    and same_chunk
                    and same_page
                ):
                    matched = True
                    break

            if matched:
                valid_citations += 1
            else:
                invalid_citations += 1

            citation_results.append(
                {
                    **citation,
                    "valid": matched,
                }
            )

        citation_precision = (
            valid_citations / len(citations)
        )

        return {
            "citations": citation_results,
            "citation_count": len(citations),
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "citation_precision": citation_precision,
            "has_citation": True,
        }