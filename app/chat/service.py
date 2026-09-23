from app.chat.memory import ConversationMemory
from app.generation.question_rewriter import QuestionRewriter
from app.rag.pipeline import RAGPipeline


class ChatService:

    def __init__(
        self,
        rag_pipeline: RAGPipeline
    ):
        self.rag_pipeline = rag_pipeline

        self.memory = ConversationMemory()

        self.question_rewriter = QuestionRewriter()

    def chat(
        self,
        conversation_id: str,
        question: str,
        k: int = 3
    ):

        # Get previous conversation
        history = self.memory.get_history(
            conversation_id
        )

        # Convert follow-up question into
        # a standalone question
        standalone_question = (
            self.question_rewriter.rewrite(
                question,
                history
            )
        )
        if standalone_question == "AMBIGUOUS":
            return {
                "answer": (
                    "I don't have enough conversation context "
                    "to understand what you are referring to."
                ),
                "sources": [],
                "question_used_for_retrieval": None
            }

        # Run normal RAG using standalone question
        result = self.rag_pipeline.query(
            standalone_question,
            k=k,
            conversation_id=conversation_id
            
        )

        answer = result["answer"]

        # Save user message
        self.memory.add_message(
            conversation_id,
            "user",
            question
        )

        # Save assistant response
        self.memory.add_message(
            conversation_id,
            "assistant",
            answer
        )

        return {
            "answer": answer,
            "sources": result["sources"],
            "question_used_for_retrieval": standalone_question
        }