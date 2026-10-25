# Enterprise Data Intelligence RAG

Local, source-grounded question answering over enterprise documents and a
PostgreSQL database. Everything runs locally with Ollama; no paid APIs.

## What it does

- **Document questions (RAG):** hybrid retrieval, cited answers (`[S1]`,
  `[S2]`), and claim-by-claim grounding verification. If the evidence does not
  support an answer, the system abstains.
- **Database questions:** natural language -> schema-checked, read-only SQL
  over PostgreSQL, with a clarification step when a customer name is ambiguous.
- **Routing:** a keyword-based router chooses `general`, `rag`, `sql` or
  `hybrid`. Follow-up questions are rewritten using recent conversation.
- **Memory:** per-thread conversation state (LangGraph, held in memory) and
  per-user long-term memory stored in PostgreSQL.

## Architecture (text flow)

```
Question
  -> query rewriting (follow-ups)  -> router (general | rag | sql | hybrid)

RAG:  Chroma dense (k=10) + BM25 (k=10) -> RRF (k=60)
      -> cross-encoder rerank (top 5) -> context filter
         (dedup, max 5 chunks / 6000 chars, rerank-score floor -10)
      -> cited answer (Ollama llama3.2:3b)
      -> per-claim grounding verification -> answer or abstention

SQL:  schema-aware generation -> syntax check -> schema check
      -> read-only check -> read-only transaction (10 s timeout, 500 rows)
```

## Requirements

- Python 3.11, PostgreSQL, and [Ollama](https://ollama.com)
- Ollama models: `llama3.2:3b` (generation, verification) and
  `nomic-embed-text` (embeddings)
- The reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`) is downloaded from
  Hugging Face on first run
- Developed and tested on Windows with PowerShell

## Setup

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Create a `.env` file (never commit it) with the PostgreSQL settings read by
`app/db/connection.py`: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, plus
the user and password variables that file reads.

## Run

```
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
streamlit run streamlit_app.py
```

- API docs: http://127.0.0.1:8000/docs
- Upload documents in the UI or with `POST /documents/upload`. Files are saved
  to `data/documents`, and the index is rebuilt on the next document question.
- Ask questions with `POST /ask` (`question`, `thread_id`, `user_id`).

## Tests

```
python -m pytest -q
```

Some tests are skipped unless you opt in:
`RUN_LIVE_OLLAMA=1` (needs Ollama) and `RUN_POSTGRES_INTEGRATION=1` (needs
PostgreSQL).

## Safety design

- **Grounding:** every sentence of an answer is checked separately against the
  evidence block it cites. Any unsupported claim, missing or invalid citation,
  verifier error, or malformed verifier output leads to abstention.
- **SQL:** a single SELECT only. Write statements anywhere in the query
  (including inside CTEs), `SELECT INTO`, row locks, and functions not
  recognized by the validator are rejected. Queries must match the database
  schema. Execution uses a read-only transaction with a statement timeout and
  a row cap.
- **Uploads:** extension allow-list, size limit, sanitized filenames, and no
  overwriting of existing files.

## Verified so far

- End-to-end `POST /ask` for a document question: correct route, correct
  answer, citation kept, only the relevant source returned.
- Live model checks: a correct answer is accepted; an unsupported claim and a
  contradicting claim are rejected.
- Unit tests for the SQL validator and executor (the executor against mocks),
  the context filter, the answer generator, the API, and memory.

## Not yet verified

- The read-only transaction against a real PostgreSQL server
- Off-topic abstention through the live API
- Document ingestion edge cases and prompt-injection resistance
- The Streamlit UI end to end, and behavior after an API restart
- Retrieval quality beyond a handful of queries; no benchmark numbers are
  reported here

## Known limitations

- Conversation state is in memory and is lost when the API restarts.
- The UI uses one shared user ID, so long-term memory is shared across UI
  sessions. The API has no authentication.
- The rerank-score floor (-10) was chosen from only three queries and applies
  to this specific reranker; re-tune it if the reranker changes.
- The claim splitter works on sentences, so abbreviations such as "Dr." can
  split a sentence; this fails closed (possible abstention, never approval).
- The grounding verifier is a 3B model; its judgments are not guaranteed.
- Multi-query retrieval exists but is off by default.
- The router is keyword-based.
