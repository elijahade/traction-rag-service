"""Application dependencies and singleton clients."""
from functools import lru_cache

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    pinecone_api_key: str = Field(..., alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(..., alias="PINECONE_INDEX_NAME")
    api_key: str = Field(..., alias="TRACTION_BRAIN_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()


@lru_cache
def get_embeddings():
    """Instantiate the Google embeddings client once."""

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=settings.google_api_key,
    )


@lru_cache
def get_llm():
    """Return a shared Gemini chat model instance."""

    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        google_api_key=settings.google_api_key,
    )


@lru_cache
def get_pinecone_client():
    """Return a Pinecone client."""

    from pinecone import Pinecone

    settings = get_settings()
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache
def get_index():
    """Return the Pinecone index handle."""

    settings = get_settings()
    client = get_pinecone_client()
    return client.Index(settings.pinecone_index_name)
