from langchain_ollama import ChatOllama


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.0,
        response_format: str | None = None,
    ):
        options = {
            "model": model,
            "temperature": temperature,
        }

        if response_format is not None:
            options["format"] = response_format

        self.llm = ChatOllama(**options)

    def invoke(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content