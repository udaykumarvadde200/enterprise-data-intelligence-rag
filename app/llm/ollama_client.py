from langchain_ollama import ChatOllama


def main():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    response = llm.invoke(
        "Explain SQL in one sentence."
    )

    print(response.content)


if __name__ == "__main__":
    main()