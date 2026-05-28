<<<<<<< HEAD

class DB:
    ip = ""
    username = ""
    password = ""
    dbname = ""
=======
import os


class DB:
    ip = os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", ""))
    username = os.getenv("POSTGRES_USER", os.getenv("DB_USER", ""))
    password = os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", ""))
    dbname = os.getenv("POSTGRES_DB", os.getenv("DB_NAME", ""))
    port = os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "5432"))
>>>>>>> d75227e (Add customer storefront API endpoints and Docker Compose with PostgreSQL)
