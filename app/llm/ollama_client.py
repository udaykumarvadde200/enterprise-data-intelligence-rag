from langchain_ollama import ChatOllama


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.0,
    ):
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
        )

    def invoke(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content 
