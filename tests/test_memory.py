from app.chat.database import ChatDatabase
from app.chat.memory import ConversationMemory


def test_messages_persist(tmp_path):
    db_path = tmp_path / "chat.db"

    database = ChatDatabase(
        db_path=str(db_path)
    )

    database.add_message(
        conversation_id="conversation-A",
        role="user",
        content="What is FastAPI?"
    )

    database.add_message(
        conversation_id="conversation-A",
        role="assistant",
        content="FastAPI is a Python web framework."
    )

    messages = database.get_messages(
        conversation_id="conversation-A"
    )

    assert messages == [
        {
            "role": "user",
            "content": "What is FastAPI?"
        },
        {
            "role": "assistant",
            "content": "FastAPI is a Python web framework."
        }
    ]


def test_conversations_are_isolated(tmp_path):
    db_path = tmp_path / "chat.db"

    memory = ConversationMemory()

    # Override the database used by memory so the test
    # does not touch the real data/chat.db.
    memory.database = ChatDatabase(
        db_path=str(db_path)
    )

    memory.add_message(
        conversation_id="conversation-A",
        role="user",
        content="Message A"
    )

    memory.add_message(
        conversation_id="conversation-B",
        role="user",
        content="Message B"
    )

    history_a = memory.get_history("conversation-A")
    history_b = memory.get_history("conversation-B")

    assert history_a == [
        {
            "role": "user",
            "content": "Message A"
        }
    ]

    assert history_b == [
        {
            "role": "user",
            "content": "Message B"
        }
    ]