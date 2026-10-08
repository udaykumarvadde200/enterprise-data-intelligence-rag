from app.sql.sql_pipeline import SQLPipeline


def main():
    pipeline = SQLPipeline()

    questions = [
        "Show me John's orders.",
        "Show me Rahul Sharma's orders.",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print("USER QUESTION:")
        print(question)

        response = pipeline.run(question)

        print("\nSTATUS:")
        print(response["status"])

        if response["status"] == "clarification_required":
            print("\nMESSAGE:")
            print(response["message"])

        elif response["status"] == "not_found":
            print("\nMESSAGE:")
            print(response["message"])

        elif response["status"] == "success":
            print("\nGENERATED SQL:")
            print(response["sql"])

            print("\nQUERY RESULT:")
            for row in response["results"]:
                print(row)


if __name__ == "__main__":
    main()