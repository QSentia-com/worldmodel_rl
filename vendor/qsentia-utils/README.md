# qsentia-utils

Shared QSentia utility library for lightweight contracts used across model containers, services, and jobs.

This repo should stay small. It should contain common model definitions and helper functions that are reused by multiple components. The database schema itself remains owned by `qsentia-db-migrations`.

## Important Boundary

`qsentia-utils` does **not** create or modify database tables.

The table definitions live in `qsentia-db-migrations`. That repo owns:

- `CREATE TABLE` statements
- indexes and constraints
- schema versioning
- migration CI that applies changes to RDS

This repo only owns runtime Python helpers. For example, `ModelOutputCreate` mirrors the `model_outputs` table so services can validate and serialize data before writing it. Creating a `ModelOutputCreate` object does not create the database table.

Think of it this way:

- `qsentia-db-migrations` = database truth / DDL / schema ownership
- `qsentia-utils` = app-side contracts / connection helpers / serialization helpers


## What Belongs Here

Good candidates:

- Shared Pydantic models for stable contracts, like `ModelRunCreate` and `ModelOutputCreate`
- Common enums, like `Environment` and `RunStatus`
- Small helper functions for converting models into database records
- Reusable output envelope definitions used by multiple models

Avoid adding:

- Database migration SQL
- Service-specific business logic
- Model-specific inference code
- Large clients or framework-heavy dependencies unless multiple services truly need them

## Current Contracts

The first version includes Python models for the shared tables maintained in `qsentia-db-migrations`:

- `model_runs`
- `model_outputs`

Example:

```python
from qsentia_utils.models import Environment, ModelOutputCreate, ModelRunCreate

run = ModelRunCreate(
    environment=Environment.DEV,
    producer_id="eth-futures-sentiment",
    producer_version="20260608",
    run_type="inference",
)

output = ModelOutputCreate(
    run_id=run.id,
    environment=Environment.DEV,
    producer_id="eth-futures-sentiment",
    producer_version="20260608",
    output_type="prediction",
    output={
        "signal": "LONG",
        "confidence": 0.72,
    },
)
```

## Relationship With Migrations

`qsentia-db-migrations` is the source of truth for actual table definitions, indexes, constraints, and schema versioning.

`qsentia-utils` provides runtime models that app code can import so each component does not redefine the same contracts again and again.

When a table changes:

1. Add a new SQL migration in `qsentia-db-migrations`.
2. Merge and apply the migration.
3. Update the matching models here only if application code needs the new fields.
4. Release a new version of this package.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Database Connection Helper

Components can use this library to create a PostgreSQL connection at runtime.

Install with the DB extra:

```bash
pip install "qsentia-utils[db]"
```

Runtime environment variables:

```bash
DATABASE_SECRET_ARN=arn:aws:secretsmanager:us-east-2:...
DATABASE_HOST=qsentia-dev-postgres.c9kswwwi611w.us-east-2.rds.amazonaws.com
DATABASE_NAME=qsentia
DATABASE_PORT=5432
AWS_REGION=us-east-2
```

Usage inside a model container or service:

```python
from qsentia_utils.db import open_connection

with open_connection() as connection:
    with connection.cursor() as cursor:
        cursor.execute("select now()")
        print(cursor.fetchone())
```

You can also build records from shared models:

```python
from qsentia_utils.db import model_output_record, open_connection
from qsentia_utils.models import ModelOutputCreate

output = ModelOutputCreate(
    run_id=run_id,
    environment="dev",
    producer_id="eth-futures-sentiment",
    output={"signal": "LONG", "confidence": 0.72},
)

record = model_output_record(output)
```

The library reads credentials in this order:

1. `DATABASE_SECRET_JSON`, useful for local tests
2. `DATABASE_SECRET_ARN`, fetched from AWS Secrets Manager
3. Direct env vars like `DATABASE_USER` and `DATABASE_PASSWORD`

Do not hardcode database passwords inside services. In AWS Batch, pass `DATABASE_SECRET_ARN`, `DATABASE_HOST`, `DATABASE_NAME`, `DATABASE_PORT`, and `AWS_REGION` through the job definition.
