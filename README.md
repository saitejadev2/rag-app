# Conversational RAG API

A document-grounded question-answering service built with FastAPI, OpenAI,
Sentence Transformers, Chroma, and SQLite. Upload PDF or text files, ask
questions against them, continue with context-aware follow-up questions, and
receive answers with source citations.

## Why this project?

This project was built to explore the complete RAG lifecycle rather than simply
calling an LLM API. It covers document ingestion, retrieval, reranking,
grounded generation, conversational context, citations, and evaluation.

## What the project does

The application turns `.txt` and `.pdf` files into a searchable knowledge base.
It extracts text, recursively splits it into overlapping chunks, creates dense
embeddings, and persists those chunks in Chroma. At query time it retrieves a
larger candidate set, reranks the candidates with a cross-encoder, and gives the
best evidence to an OpenAI model. The model is instructed to answer only from
that evidence, cite its sources, and abstain when the documents do not contain
an answer.

The conversational API scopes uploaded documents and retrieval by
`conversation_id`. Chat history is stored in SQLite and is used to rewrite
context-dependent follow-up questions into standalone retrieval queries.

## Architecture and RAG pipeline

```text
PDF / TXT
   |
   v
text extraction -> recursive chunking (500 chars, 50-char overlap)
   |
   v
all-MiniLM-L6-v2 embeddings -> persistent Chroma collection
                                      |
User question                         |
   |                                  |
   +-> chat history -> question rewrite
                          |
                          v
              vector candidate retrieval (max(3k, 10))
                          |
                          v
           ms-marco-MiniLM-L-6-v2 cross-encoder
                          |
                          v
                    top-k chunks
                          |
                          v
             grounded OpenAI generation
                          |
                          v
               answer + source citations
```

For `/chat`, the retrieval query and documents are isolated by conversation.
For `/query`, the question is sent directly to the RAG pipeline without using
chat history.

## Key features

- **Recursive chunking:** splits on paragraphs, lines, sentences, spaces, and
  finally characters, while preserving source, page, and chunk metadata.
- **Embeddings and Chroma:** uses normalized `all-MiniLM-L6-v2` embeddings and
  a disk-backed Chroma collection.
- **Two-stage retrieval:** retrieves `max(3k, 10)` vector candidates and reranks
  them with `cross-encoder/ms-marco-MiniLM-L-6-v2` before selecting the top `k`.
- **Conversational memory:** persists messages by `conversation_id` in
  `data/chat.db` using SQLite.
- **Question rewriting:** turns follow-ups into standalone questions and asks
  for clarification when a reference cannot be resolved from the history.
- **Citations:** includes exact source/page/chunk labels in generated answers
  and returns source metadata, vector distance, and reranker score separately.
- **Document management:** upload, list, delete from the index, and reindex
  conversation-scoped documents. Reingestion replaces existing chunks for the
  same filename and conversation.
- **FastAPI API:** typed JSON requests, multipart uploads, health checks, and
  interactive OpenAPI documentation.

## Evaluation

The evaluation set contains 30 questions over a controlled four-chunk text
document: 20 answerable questions and 10 questions that require abstention. The
recorded evaluation run produced:

| Metric | Result |
| --- | ---: |
| Answerable questions answered correctly | 20/20 (100%) |
| Unanswerable questions correctly abstained | 10/10 (100%) |
| Citation precision | 100% |
| Grounding accuracy | 100% |

Retrieval recall is macro-averaged over the 20 answerable questions. The values
below were reproduced from the current evaluation index and models:

| Chunk recall | Vector retrieval | After reranking |
| --- | ---: | ---: |
| Recall@1 | 75% | 75% |
| Recall@3 | 100% | 95% |
| Recall@5 | 100% | 100% |

Citation precision measures whether generated citation labels match retrieved
chunks. Grounding is judged only where cited evidence is available. Answer and
grounding correctness are model-judged, so results can vary when the configured
OpenAI model changes.

To run the evaluation, first place the evaluation document where the ingestion
script expects it, then index and evaluate it:

```powershell
Copy-Item data/evaluation/evaluation.txt data/documents/evaluation/evaluation.txt
python -m app.evaluation.ingest_evaluation
python -m app.evaluate
```

The evaluator reports vector and reranked recall at 1, 3, and 5, per-question
reranking effects, answer accuracy, abstention accuracy, citation precision,
and grounding accuracy.

## Project structure

```text
rag-app/
|-- app/
|   |-- api.py                  # FastAPI application and endpoints
|   |-- config.py               # Environment configuration
|   |-- ingest.py               # Batch ingestion entry point
|   |-- evaluate.py             # Evaluation runner and summary
|   |-- chat/                   # SQLite memory and chat orchestration
|   |-- evaluation/             # Retrieval, answer, citation, grounding checks
|   |-- generation/             # Answer generation and question rewriting
|   |-- ingestion/              # Loading, chunking, embedding, indexing
|   |-- rag/                    # End-to-end RAG pipeline
|   `-- retrieval/              # Chroma retrieval and cross-encoder reranking
|-- data/
|   |-- documents/              # Uploaded/source documents (runtime data)
|   |-- evaluation/             # Evaluation document and question set
|   |-- chroma/                 # Persistent vector index (generated)
|   `-- chat.db                 # Conversation history (generated)
|-- tests/                      # Pytest test suite
|-- requirements.txt
`-- README.md
```

## Setup and running

Python 3.10 or newer is recommended. An OpenAI API key is required for answer
generation, question rewriting, and model-based evaluation.

1. Create and activate a virtual environment.

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies.

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Create `.env` in the repository root.

   ```dotenv
   OPENAI_API_KEY=your_api_key_here
   ```

4. Start the API.

   ```powershell
   uvicorn app.api:app --reload
   ```

5. Open `http://127.0.0.1:8000/docs` for the interactive API documentation,
   or check `http://127.0.0.1:8000/health`.

The first startup or test run may download the embedding and cross-encoder
models from Hugging Face. Chroma data and chat history are created under
`data/` and are excluded from Git.

For non-conversation-scoped batch ingestion, put `.txt` or `.pdf` files directly
inside `data/documents/` and run:

```powershell
python -m app.ingest
```

## Testing

Run the test suite from the repository root:

```powershell
python -m pytest --basetemp=.pytest_tmp
```

The tests cover recursive chunking, text loading, Chroma conversation isolation
and scoped deletion, cross-encoder ordering, SQLite memory, and citation
validation. Model-backed tests require the Sentence Transformer models to be
available locally or downloadable.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Return service health. |
| `POST` | `/query` | Run a direct, stateless RAG query. |
| `POST` | `/chat` | Run a conversation-scoped query with memory and rewriting. |
| `POST` | `/documents/upload` | Upload and immediately index a PDF or text file. |
| `GET` | `/documents` | List indexed documents for a conversation. |
| `DELETE` | `/documents/{filename}` | Delete a conversation's document chunks from Chroma. |
| `POST` | `/documents/{filename}/reindex` | Rebuild a saved document's index entries. |

### Chat

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"demo","question":"What does the document say?","k":3}'
```

The response contains `answer`, `sources`, and
`question_used_for_retrieval`.

### Upload and manage documents

```bash
curl -X POST "http://127.0.0.1:8000/documents/upload?conversation_id=demo" \
  -F "file=@example.pdf"

curl "http://127.0.0.1:8000/documents?conversation_id=demo"

curl -X POST \
  "http://127.0.0.1:8000/documents/example.pdf/reindex?conversation_id=demo"

curl -X DELETE \
  "http://127.0.0.1:8000/documents/example.pdf?conversation_id=demo"
```

### Direct query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is retrieval-augmented generation?","k":3}'
```

## Limitations and future improvements

- `/query` is not conversation-filtered and can search all records in the shared
  collection; production deployments should require a tenant or conversation
  scope on every retrieval path.
- Authentication, authorization, upload-size limits, rate limiting, and file
  malware scanning are not implemented.
- PDF support depends on extractable text; scanned documents need OCR.
- Deleting a document removes its Chroma entries but leaves the uploaded file
  on disk, and there is no endpoint for clearing conversation history.
- Chat history grows without truncation or summarization and SQLite is best
  suited to a single-instance deployment.
- Retrieval uses dense similarity only. Hybrid keyword/vector search, metadata
  filters, relevance thresholds, and calibrated abstention would improve
  robustness.
- Model names and chunking values are currently hard-coded. Moving them to
  environment or application settings would make deployments easier to tune.
- Background ingestion, batched uploads, structured logging, observability,
  streaming responses, and automated end-to-end API tests remain future work.
