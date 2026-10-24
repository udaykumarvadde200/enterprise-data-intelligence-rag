from unittest.mock import patch

from app.llm.ollama_client import OllamaClient


def test_get_chat_model_returns_underlying_chat_model():
    with patch("app.llm.ollama_client.ChatOllama") as chat_ollama:
        client = OllamaClient()

    assert client.get_chat_model() is chat_ollama.return_value


def test_invoke_returns_message_content():
    with patch("app.llm.ollama_client.ChatOllama") as chat_ollama:
        chat_ollama.return_value.invoke.return_value.content = "hello"
        client = OllamaClient()

    assert client.invoke("hi") == "hello"