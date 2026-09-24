import json

from openai import OpenAI

from app.config import OPENAI_API_KEY


class AnswerEvaluator:

    def __init__(self, model: str = "gpt-5.5"):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        self.model = model

    def evaluate(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str
    ):
        prompt = f"""
You are evaluating the answer produced by a
Retrieval-Augmented Generation system.

Evaluate whether the generated answer correctly
answers the question based on the expected answer.

Question:
{question}

Expected answer:
{expected_answer}

Generated answer:
{generated_answer}

Rules:

1. Return "correct": true if the generated answer
   contains the essential information from the
   expected answer.

2. Additional relevant information is allowed.

3. Do not require the wording to be identical.

4. Return "correct": false if the generated answer
   contradicts the expected answer or misses its
   essential information.

Return ONLY valid JSON:

{{
    "correct": true,
    "reason": "brief explanation"
}}
"""

        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        return json.loads(response.output_text)