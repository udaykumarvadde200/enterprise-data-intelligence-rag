from app.sql.sql_pipeline import SQLPipeline


def main():
    pipeline = SQLPipeline()

    question = "Show customers from Bangalore."

    sql, results = pipeline.run(question)

    print("USER QUESTION:")
    print(question)

    print("\nGENERATED SQL:")
    print(sql)

    print("\nQUERY RESULT:")

    for row in results:
        print(row)


if __name__ == "__main__":
    main()