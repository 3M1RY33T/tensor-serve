import sqlite3
import json
from datetime import datetime
from typing import List, Dict

CONV_DB = "conversations.db"


def init_conversations_db():
    """Initialize conversations database if it doesn't exist."""
    conn = sqlite3.connect(CONV_DB)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            updated_at TEXT,
            title TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            context TEXT,
            created_at TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
        """
    )

    conn.commit()
    conn.close()


def create_conversation(conversation_id: str, title: str = "New Conversation") -> Dict:
    """Create a new conversation."""
    init_conversations_db()
    conn = sqlite3.connect(CONV_DB)
    cursor = conn.cursor()

    now = datetime.utcnow().isoformat()
    cursor.execute(
        "INSERT INTO conversations (id, created_at, updated_at, title) VALUES (?, ?, ?, ?)",
        (conversation_id, now, now, title),
    )
    conn.commit()
    conn.close()

    return {"id": conversation_id, "title": title, "created_at": now}


def add_message(
    conversation_id: str, role: str, content: str, context: str = None
) -> Dict:
    """Add a message to a conversation."""
    init_conversations_db()
    conn = sqlite3.connect(CONV_DB)
    cursor = conn.cursor()

    now = datetime.utcnow().isoformat()
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, context, created_at) VALUES (?, ?, ?, ?, ?)",
        (conversation_id, role, content, context, now),
    )
    msg_id = cursor.lastrowid

    cursor.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
    )
    conn.commit()
    conn.close()

    return {"id": msg_id, "role": role, "content": content, "context": context}


def get_conversation_history(conversation_id: str, limit: int = 20) -> List[Dict]:
    """Get conversation history."""
    init_conversations_db()
    conn = sqlite3.connect(CONV_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT role, content, context, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
        (conversation_id, limit),
    )
    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return list(reversed(messages))  # Return in chronological order
