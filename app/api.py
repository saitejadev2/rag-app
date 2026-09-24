from fastapi import FastAPI
from pydantic import BaseModel

from pathlib import Path
from fastapi import UploadFile, File, HTTPException
import shutil

from app.ingestion.service import IngestionService
from app.chat.service import ChatService
from app.ingestion.embedder import Embedder
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.generation.generator import Generator
from app.rag.pipeline import RAGPipeline


app = FastAPI(
    title="RAG Application",
    description="A document-based question answering system",
    version="1.0.0"
)


# -------------------------
# Initialize RAG components
# -------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"


embedder = Embedder()

vector_store = VectorStore(
    persist_directory=str(CHROMA_DIR)
)
reranker = Reranker()
retriever = Retriever(
    vector_store=vector_store,
    embedder=embedder,
    reranker=reranker
)

generator = Generator()

rag = RAGPipeline(
    retriever=retriever,
    generator=generator
)
chat_service = ChatService(
    rag_pipeline=rag
)

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

ingestion_service = IngestionService(
    documents_dir=DOCUMENTS_DIR,
    chroma_dir=CHROMA_DIR
)
# -------------------------
# Request model
# -------------------------
class ChatRequest(BaseModel):
    conversation_id: str
    question: str
    k: int = 3
class QuestionRequest(BaseModel):
    question: str
    k: int = 3


# -------------------------
# Query endpoint
# -------------------------

@app.post("/query")
def query(request: QuestionRequest):

    result = rag.query(
        question=request.question,
        k=request.k
    )

    return result


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/documents/upload")
async def upload_document(
    conversation_id: str,
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".pdf",
        ".txt"
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported."
        )

    conversation_dir = DOCUMENTS_DIR / conversation_id
    conversation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = Path(file.filename).name
    file_path = conversation_dir / filename

    try:

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        result = ingestion_service.ingest_file(
            file_path,
            conversation_id=conversation_id
        )

        return {
            "message": "Document uploaded and indexed successfully.",
            **result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/documents")
def list_documents(conversation_id: str):
    results = vector_store.collection.get(
        where={
            "conversation_id": conversation_id
        },
        include=["metadatas"]
    )

    documents = {}

    for metadata in results["metadatas"]:
        if not metadata:
            continue

        source = metadata.get("source")

        if source not in documents:
            documents[source] = {
                "filename": source,
                "conversation_id": metadata.get(
                    "conversation_id"
                ),
                "chunks": 0
            }

        documents[source]["chunks"] += 1

    return {
        "documents": list(documents.values())
    }

@app.post("/chat")
def chat(request: ChatRequest):

    result = chat_service.chat(
        conversation_id=request.conversation_id,
        question=request.question,
        k=request.k
    )

    return result

@app.delete("/documents/{filename}")
def delete_document(
    filename: str,
    conversation_id: str
):
    vector_store.delete_by_source(
        filename,
        conversation_id=conversation_id
    )

    return {
        "message": "Document deleted successfully",
        "filename": filename,
        "conversation_id": conversation_id
    }

@app.post("/documents/{filename}/reindex")
def reindex_document(
    filename: str,
    conversation_id: str
):
    file_path = (
        DOCUMENTS_DIR
        / conversation_id
        / Path(filename).name
    )

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    result = ingestion_service.ingest_file(
        file_path=file_path,
        conversation_id=conversation_id
    )

    return {
        "message": "Document reindexed successfully",
        **result
    }