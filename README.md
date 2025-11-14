# Traction Brain Service

FastAPI sidecar that keeps Pinecone embeddings up to date and exposes Retrieval Augmented Generation (RAG) helpers for Traction OS clients.

## Features

- **Embeddings upsert/delete** – mirror Firestore item changes to Pinecone.
- **Top 3 suggestions** – RAG workflow that selects the user's highest-leverage actions.

## Project Structure

```
app/
  deps.py      # settings + shared clients
  main.py      # FastAPI entry point
  models.py    # request/response schemas
  rag.py       # LangChain helpers
  vectorstore.py # Pinecone helpers
```

## Environment Variables

```
GOOGLE_API_KEY=<gemini key>
PINECONE_API_KEY=<pinecone key>
PINECONE_INDEX_NAME=traction-items
TRACTION_BRAIN_API_KEY=<shared secret>
```

Create a `.env` file locally or export these before running the server.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| `POST` | `/api/v1/items/upsert` | Upsert / refresh an item vector |
| `POST` | `/api/v1/items/delete` | Delete an item from Pinecone |
| `POST` | `/api/v1/suggestions/top3-today` | Return today's Top 3 |

All routes require an `X-API-KEY` header that matches `TRACTION_BRAIN_API_KEY`.
