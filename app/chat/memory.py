from app.chat.database import ChatDatabase


class ConversationMemory:

    def __init__(self):
        self.database = ChatDatabase()

    def get_history(
        self,
        conversation_id: str
    ):
        return self.database.get_messages(
            conversation_id
        )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str
    ):
        self.database.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content
        )