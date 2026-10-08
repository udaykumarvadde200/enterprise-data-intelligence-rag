from app.llm.ollama_client import OllamaClient
from app.sql.customer_reference_extractor import CustomerReferenceExtractor


def main():
    llm = OllamaClient()
    extractor = CustomerReferenceExtractor(llm)

    questions = [
        "Show me John's orders.",
        "Show me Rahul Sharma's orders.",
        "How many orders were placed yesterday?",
    ]

    for question in questions:
        reference = extractor.extract(question)

        print("\nQUESTION:")
        print(question)

        print("CUSTOMER REFERENCE:")
        print(reference)


if __name__ == "__main__":
    main()