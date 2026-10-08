from app.db.connection import DatabaseConnection
from app.sql.ambiguity_checker import AmbiguityChecker


def main():
    db = DatabaseConnection()
    checker = AmbiguityChecker(db)

    print("John Smith:")
    print(checker.check_customer_name("John Smith"))

    print("\nRahul Sharma:")
    print(checker.check_customer_name("Rahul Sharma"))

    print("\nUnknown Person:")
    print(checker.check_customer_name("Unknown Person"))


if __name__ == "__main__":
    main()