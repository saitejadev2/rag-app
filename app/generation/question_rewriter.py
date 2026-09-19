from openai import OpenAI

from app.config import OPENAI_API_KEY


class QuestionRewriter:

    def __init__(self, model: str = "gpt-5.5"):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        self.model = model

    def rewrite(
        self,
        question: str,
        history: list[dict]
    ):

        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in history
        )

        if not history_text:
            history_text = "No previous conversation."

        prompt = f"""
You are a question rewriting assistant for a
document-based question answering system.

Determine whether the user's latest question
can be understood without previous conversation.

Rules:

1. If the question is completely standalone,
   return the question unchanged.

2. If the question depends on previous conversation
   context, rewrite it into a standalone question.

3. If the question contains an ambiguous reference
   such as:
   "it", "its", "they", "them", "this", "that",
   "these", or "those"

   AND there is not enough information in the
   conversation to determine what it refers to,
   return exactly:

   AMBIGUOUS

4. Do not answer the question.

Previous conversation:
{history_text}

Latest user question:
{question}

Return only the standalone question or:

AMBIGUOUS
"""

        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        rewritten = response.output_text.strip()

        return rewritten