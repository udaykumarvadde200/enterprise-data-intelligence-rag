\# Enterprise Data Intelligence RAG



A conversational AI system for querying and reasoning over structured and unstructured enterprise data.



\## Project Goals



\- Query structured data using natural language and SQL

\- Retrieve information from unstructured documents using advanced RAG

\- Detect ambiguous requests and ask clarification questions

\- Combine SQL, retrieval, and external tools when required

\- Validate generated SQL before database execution

\- Evaluate retrieval, generation, routing, and SQL accuracy

\- Build a production-oriented backend using FastAPI, PostgreSQL, LangGraph, and a local LLM



\## Planned Architecture



```text

User

&#x20; ↓

Query Understanding

&#x20; ↓

Ambiguity Detection

&#x20; ↓

Query Router

&#x20; ├── SQL

&#x20; ├── RAG

&#x20; └── Tools

&#x20; ↓

Verification

&#x20; ↓

Response



Tech Stack

Python

LangGraph

Ollama

PostgreSQL

FastAPI

Vector Database

BM25

Hybrid Retrieval

Reranking

Docker

Status



🚧 Under active development.



