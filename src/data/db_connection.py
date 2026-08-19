import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


database_url = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
)


engine = create_engine(database_url)


def test_connection():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))

            print("Database connection successful.")
            print(result.fetchone()[0])

    except Exception as error:
        print("Database connection failed.")
        print(error)


if __name__ == "__main__":
    test_connection()