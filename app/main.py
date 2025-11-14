"""FastAPI application exposing the Traction Brain API."""
from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, status

from . import models, rag
from .deps import get_settings
from .vectorstore import delete_item_vector, upsert_item_vector

app = FastAPI(title="Traction Brain", version="0.1.0")


def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-KEY"),
    settings=Depends(get_settings),
):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@app.post("/api/v1/items/upsert", response_model=models.SuccessResponse)
async def upsert_item(
    payload: models.UpsertItemRequest,
    _: None = Depends(verify_api_key),
):
    """Upsert or update a user's item embedding."""

    upsert_item_vector(user_id=payload.userId, item=payload.item)
    return models.SuccessResponse()


@app.post("/api/v1/items/delete", response_model=models.SuccessResponse)
async def delete_item(
    payload: models.DeleteItemRequest,
    _: None = Depends(verify_api_key),
):
    """Delete an item vector from Pinecone."""

    delete_item_vector(item_id=payload.itemId)
    return models.SuccessResponse()


@app.post("/api/v1/suggestions/top3-today", response_model=models.Top3Response)
async def suggest_top3(
    payload: models.Top3Request,
    _: None = Depends(verify_api_key),
):
    """Return the user's Top 3 suggested actions."""

    top3 = rag.suggest_top3(payload)
    return models.Top3Response(top3=top3)
