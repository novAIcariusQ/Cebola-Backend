<<<<<<< HEAD
# Backend for the thing
=======
# Cebola Backend API

FastAPI backend with PostgreSQL (Docker) or SQLite fallback.

## Run with Docker (PostgreSQL)

From the repository root:

```bash
docker compose up --build
```

API: http://localhost:3001  
Frontend (full stack): http://localhost:3000

Backend only:

```bash
cd Cebola-Backend-dev/Cebola-Backend-dev
docker compose up --build
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 3001
```

Set `POSTGRES_HOST`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` to use PostgreSQL; otherwise SQLite (`cebola.db`) is used.

Demo merchant login after seed: `merchant@example.com` / `password123`
>>>>>>> d75227e (Add customer storefront API endpoints and Docker Compose with PostgreSQL)
