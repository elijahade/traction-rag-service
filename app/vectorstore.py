"""Helpers for working with Pinecone and embeddings."""
from __future__ import annotations

from typing import Dict, List, Optional

from . import models
from .deps import get_embeddings, get_index


def build_item_text(item: models.TractionItem) -> str:
    """Combine title and description into a single text blob."""

    parts: List[str] = [item.title]
    if item.description:
        parts.append(item.description)
    return "\n\n".join(parts)


def upsert_item_vector(user_id: str, item: models.TractionItem) -> None:
    """Generate an embedding and upsert it into Pinecone."""

    text = build_item_text(item)
    embedding = get_embeddings().embed_documents([text])[0]
    metadata: Dict[str, Optional[str]] = {
        "user_id": user_id,
        "type": item.type,
        "status": item.status,
        "energy": item.energy,
        "size": item.size,
        "title": item.title,
        "created_at": item.createdAt,
        "text": text,
        "item_id": item.id,
    }
    get_index().upsert(vectors=[{"id": item.id, "values": embedding, "metadata": metadata}])


def delete_item_vector(item_id: str) -> None:
    """Remove an item from Pinecone."""

    get_index().delete(ids=[item_id])


def query_user_items(
    *,
    user_id: str,
    question: str,
    top_k: int = 20,
    include_types: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Query Pinecone for the most relevant user items."""

    include_types = include_types or ["action", "outcome"]
    vector = get_embeddings().embed_query(question)
    metadata_filter = {
        "user_id": user_id,
        "status": "open",
    }
    if include_types:
        metadata_filter["type"] = {"$in": include_types}

    response = get_index().query(
        vector=vector,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
        filter=metadata_filter,
    )

    items: List[Dict[str, str]] = []
    for match in response.matches:
        metadata = match.metadata or {}
        items.append(
            {
                "id": metadata.get("item_id", match.id),
                "title": metadata.get("title", ""),
                "type": metadata.get("type", ""),
                "energy": metadata.get("energy", ""),
                "size": metadata.get("size", ""),
                "status": metadata.get("status", ""),
                "text": metadata.get("text", ""),
                "score": match.score or 0.0,
            }
        )
    return items
