from fastapi import FastAPI
from pydantic import BaseModel

from pathlib import Path
from fastapi import UploadFile, File, HTTPException
import shutil

from app.ingestion.service import IngestionService
from app.chat.service import ChatService
from app.ingestion.embedder import Embedder
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

retriever = Retriever(
    vector_store=vector_store,
    embedder=embedder
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
def upload_document(
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

    file_path = DOCUMENTS_DIR / file.filename

    try:

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        result = ingestion_service.ingest_file(
            file_path
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
def list_documents():

    sources = ingestion_service.vector_store.get_sources()

    return {
        "documents": sources
    }

@app.post("/chat")
def chat(request: ChatRequest):

    result = chat_service.chat(
        conversation_id=request.conversation_id,
        question=request.question,
        k=request.k
    )

    return result