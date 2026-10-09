import os
import uuid

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Enterprise Knowledge Intelligence",
    page_icon="🧠",
    layout="wide",
)

st.title("Enterprise Knowledge Intelligence")
st.caption(
    "Ask questions across your enterprise documents and structured data."
)


def api_get(path: str):
    return requests.get(f"{API_BASE_URL}{path}", timeout=10)




def api_post(path: str, **kwargs):
    kwargs.setdefault("timeout", 180)
    return requests.post(f"{API_BASE_URL}{path}", **kwargs)

def get_error_message(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.json().get("detail")
            if detail:
                return str(detail)
        except ValueError:
            pass
    return str(exc)


def render_sources(result: dict):
    sources = result.get("sources", [])

    # Hybrid workflow stores document evidence inside result["rag"].
    if not sources and isinstance(result.get("rag"), dict):
        sources = result["rag"].get("sources", [])

    if not sources:
        return

    with st.expander(f"Document sources ({len(sources)})"):
        for source in sources:
            source_id = source.get("id", "Source")
            filename = source.get("source", "Unknown source")
            page = source.get("page")
            location = f"{filename}, page {page}" if page else filename

            st.markdown(f"**[{source_id}] {location}**")
            content = source.get("content", "")
            if content:
                st.write(content)


def render_result(result: dict):
    if not result:
        return

    route = result.get("route")
    if route:
        st.caption(f"Processing route: `{route}`")

    render_sources(result)

    # Display the structured SQL portion of hybrid responses separately.
    sql_part = result.get("sql")
    if isinstance(sql_part, dict):
        with st.expander("Database query results"):
            st.write(sql_part.get("answer", "No database answer returned."))
            details = sql_part.get("result", {})
            if details.get("sql"):
                st.code(details["sql"], language="sql")

    error = result.get("error")
    if error:
        st.warning(str(error))


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: connection status, uploads, and document inventory.
with st.sidebar:
    st.header("Workspace")

    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    try:
        health = api_get("/health")
        health.raise_for_status()
        st.success("API connected")
    except requests.RequestException:
        st.error("API unavailable")
        st.caption("Start FastAPI in the other terminal.")

    st.divider()
    st.subheader("Upload knowledge")

    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Maximum file size: 15 MiB.",
    )

    if st.button(
        "Upload document",
        disabled=uploaded_file is None,
        use_container_width=True,
    ):
        try:
            response = api_post(
                "/documents/upload",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                },
                timeout=30,
            )
            response.raise_for_status()
            st.success(response.json()["message"])
        except requests.RequestException as exc:
            st.error(f"Upload failed: {get_error_message(exc)}")

    st.divider()
    st.subheader("Document library")

    try:
        response = api_get("/documents")
        response.raise_for_status()
        document_data = response.json()
        st.caption(f"{document_data['count']} supported document(s)")

        for document in document_data["documents"]:
            size_kb = document["size_bytes"] / 1024
            st.text(f"{document['filename']} · {size_kb:.1f} KB")

    except requests.RequestException:
        st.caption("Document list unavailable until the API connects.")

    st.divider()
    st.caption("Local development · Ollama-powered RAG")


# Main area: conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_result(message.get("result", {}))

question = st.chat_input("Ask about a policy, document, customer, or order...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge and preparing an answer..."):
            try:
                response = api_post(
                    "/ask",
                    json={
                        "question": question,
                        "thread_id": st.session_state.thread_id,
                    },
                )
                response.raise_for_status()
                payload = response.json()

                answer = payload.get("answer", "No answer returned.")
                result = payload.get("result", {})

                st.markdown(answer)
                render_result(result)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "result": result,
                    }
                )

            except requests.RequestException as exc:
                message = (
                    "I couldn't reach the backend or process this question. "
                    f"{get_error_message(exc)}"
                )
                st.error(message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": message, "result": {}}
                )