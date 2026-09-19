from openai import OpenAI

from app.config import OPENAI_API_KEY


class Generator:

    def __init__(self, model: str = "gpt-5.5"):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        self.model = model

    def generate(
        self,
        query: str,
        context: str
    ) -> str:

        prompt = f"""
You are a helpful assistant answering questions
based on provided documents.

Rules:
1. Answer using only the provided context.
2. Do not use outside knowledge.
3. If the context does not contain enough information
   to answer the question, say:
   "I don't know based on the provided documents."
4. Be concise and directly answer the question.

Context:
{context}

Question:
{query}
"""

        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        return response.output_text