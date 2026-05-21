# basket_pace_lab

Pre-match and live prediction of basketball totals (match / halves / quarters) using gradient boosting (CatBoost).

## Stack

| Layer | Tech |
|---|---|
| Language | Python 3.11+ |
| Database | PostgreSQL 16 |
| ORM | SQLAlchemy 2.0 async |
| Config | Pydantic v2 + pydantic-settings |
| ML | CatBoost, scikit-learn |
| HTTP | httpx |

## Quick start (Docker Compose)

### 1. Prepare environment

```bash
cp .env.example .env
# Edit .env if you need non-default credentials
```

### 2. Build and start

```bash
docker compose up --build
```

This spins up two containers:
- **postgres** — PostgreSQL 16 (port 5432, data persisted in `postgres_data` volume)
- **app** — Python app that runs `src/main.py` on startup

### 3. Verify tables were created

After `app` logs `System ready.`, connect to Postgres and check:

```bash
docker compose exec postgres psql -U postgres -d basket_pace -c "\dt"
```

Expected output:

```
        List of relations
 Schema |     Name      | Type  |  Owner
--------+---------------+-------+----------
 public | matches       | table | postgres
 public | quarter_stats | table | postgres
 public | teams         | table | postgres
```

### 4. Stop

```bash
docker compose down          # keep DB volume
docker compose down -v       # also delete DB volume
```

## Local development (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start Postgres separately, then:
cp .env.example .env  # set DB_HOST=localhost
python -m src.main
```

## Project structure

```
basket_pace_lab/
├── src/
│   ├── config.py              # Pydantic settings (reads from .env)
│   ├── main.py                # Entry point: init DB + health check
│   └── database/
│       ├── engine.py          # Async engine, session factory
│       ├── models.py          # SQLAlchemy 2.0 ORM models
│       └── crud.py            # Async CRUD: get_or_create_team, upsert_match, save_quarter_stats
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## OT isolation design

`QuarterStats.period_type` separates regulation quarters (`QUARTER`) from overtime periods (`OVERTIME`).
`Match` stores both `*_score_final` (includes OT) and `*_score_regulation` (Q1–Q4 only).
ML pipelines must use `*_score_regulation` as the totals target and filter out rows where `q4_includes_ot_points = True`.
