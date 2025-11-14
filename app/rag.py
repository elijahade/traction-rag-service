"""RAG helpers for the Traction Brain service."""
from __future__ import annotations

import json
import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from . import models
from .deps import get_llm
from .vectorstore import query_user_items

PROMPT = ChatPromptTemplate.from_template(
    """You are Traction Coach, an expert planner helping users choose their highest-leverage actions.

Context items:
{context}

Question: {question}

Return up to {max_items} recommended actions as JSON with shape:
{{"items": [{{"itemId": "...", "reason": "...", "score": 0.9}}]}}
Only reference item ids that appear in the context list.
"""
)


def format_context(items: List[dict]) -> str:
    """Convert matches into a text block for the LLM."""

    if not items:
        return "No open actions found for this user."

    lines = []
    for item in items:
        lines.append(
            "- id={id}; type={type}; title={title}; energy={energy}; size={size}; score={score:.2f}\n  text: {text}".format(
                **{
                    "id": item.get("id", ""),
                    "type": item.get("type", ""),
                    "title": item.get("title", ""),
                    "energy": item.get("energy", ""),
                    "size": item.get("size", ""),
                    "score": item.get("score", 0.0),
                    "text": item.get("text", ""),
                }
            )
        )
    return "\n".join(lines)


def _extract_json_block(text: str) -> str:
    """Attempt to pull a JSON blob from the LLM output."""

    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(part for part in text.split("```") if part and not part.startswith("json"))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain JSON")
    return match.group(0)


def parse_top3_response(content: str) -> List[models.Top3Item]:
    """Convert the LLM output into structured items."""

    payload = json.loads(_extract_json_block(content))
    items = []
    for entry in payload.get("items", []):
        if "itemId" in entry and "reason" in entry:
            items.append(
                models.Top3Item(
                    itemId=str(entry["itemId"]),
                    reason=str(entry.get("reason", "")),
                    score=float(entry.get("score", 0.0)),
                )
            )
    return items


def suggest_top3(request: models.Top3Request) -> List[models.Top3Item]:
    """Run the Top 3 RAG chain."""

    question = request.question or "What should my top 3 actions be today?"
    items = query_user_items(user_id=request.userId, question=question, top_k=20)
    context = format_context(items)

    llm = get_llm()
    messages = PROMPT.format_messages(context=context, question=question, max_items=request.maxItems)
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    results = parse_top3_response(content)
    return results[: request.maxItems]
