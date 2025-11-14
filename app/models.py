"""Pydantic models for the Traction Brain service."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ItemType = Literal["outcome", "action", "note"]
Energy = Literal["energizing", "draining", "neutral"]
Size = Literal["S", "M", "L"]
Status = Literal["open", "done", "archived"]


class TractionItem(BaseModel):
    """Representation of an item stored in Firestore."""

    id: str = Field(..., description="Firestore id")
    type: ItemType
    title: str
    description: Optional[str] = None
    energy: Optional[Energy] = None
    size: Optional[Size] = None
    status: Status = "open"
    createdAt: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp"
    )


class UpsertItemRequest(BaseModel):
    userId: str
    item: TractionItem


class DeleteItemRequest(BaseModel):
    userId: str
    itemId: str


class Top3Request(BaseModel):
    userId: str
    question: Optional[str] = Field(
        default="What should my top 3 actions be today?",
        description="Prompt passed to the LLM"
    )
    timeWindow: Optional[str] = Field(
        default="today",
        description="Human friendly hint for the prompt"
    )
    maxItems: int = Field(default=3, ge=1, le=5)


class Top3Item(BaseModel):
    itemId: str
    reason: str
    score: float


class Top3Response(BaseModel):
    success: bool = True
    top3: List[Top3Item]


class SuccessResponse(BaseModel):
    success: bool = True
