import json

from openai import OpenAI

from app.config import OPENAI_API_KEY


class GroundingEvaluator:

    def __init__(self, model: str = "gpt-5.5"):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        self.model = model

    def evaluate(
        self,
        question: str,
        answer: str,
        cited_chunks: list[dict]
    ):
        """
        Evaluate whether the generated answer is supported
        by the chunks cited in the answer.
        """

        if not cited_chunks:
            return {
                "grounded": None,
                "reason": "No cited chunks available.",
            }

        evidence_parts = []

        for index, chunk in enumerate(
            cited_chunks,
            start=1
        ):
            source = chunk.get(
                "source",
                "Unknown"
            )

            page = chunk.get("page")
            chunk_id = chunk.get("chunk_id")

            if page is not None:
                source_info = (
                    f"{source}, "
                    f"page {page}, "
                    f"chunk {chunk_id}"
                )
            else:
                source_info = (
                    f"{source}, "
                    f"chunk {chunk_id}"
                )

            evidence_parts.append(
                f"[Evidence {index}: {source_info}]\n"
                f"{chunk['text']}"
            )

        evidence = "\n\n".join(
            evidence_parts
        )

        prompt = f"""
You are evaluating whether an answer produced by a
Retrieval-Augmented Generation system is grounded in
the evidence that it cites.

Question:
{question}

Generated answer:
{answer}

Cited evidence:
{evidence}

Evaluation rules:

1. Return "grounded": true if the cited evidence
   directly supports the essential claims made in
   the generated answer.

2. Return "grounded": false if the answer contains
   an important claim that is not supported by the
   cited evidence.

3. Minor wording differences are acceptable.

4. Reasonable paraphrasing is acceptable.

5. Do not use outside knowledge.

6. If the answer contains multiple claims, the cited
   evidence should support the important claims.

Return ONLY valid JSON:

{{
    "grounded": true,
    "reason": "brief explanation"
}}
"""

        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        return json.loads(
            response.output_text
        )