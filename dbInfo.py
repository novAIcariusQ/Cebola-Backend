import os


class DB:
    ip = os.environ.get("CEBOLA_DB_HOST", "")
    username = os.environ.get("CEBOLA_DB_USER", "")
    password = os.environ.get("CEBOLA_DB_PASSWORD", "")
    dbname = os.environ.get("CEBOLA_DB_NAME", "")
