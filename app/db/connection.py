import os

import psycopg
from dotenv import load_dotenv

load_dotenv()


class DatabaseConnection:
    def __init__(self):
        self.connection_string = (
            f"host={os.getenv('POSTGRES_HOST')} "
            f"port={os.getenv('POSTGRES_PORT')} "
            f"dbname={os.getenv('POSTGRES_DB')} "
            f"user={os.getenv('POSTGRES_USER')} "
            f"password={os.getenv('POSTGRES_PASSWORD')}"
        )

    def connect(self):
        return psycopg.connect(self.connection_string)

