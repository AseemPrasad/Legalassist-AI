# Docker Development Environment

## Prerequisites

- Docker Engine 24.0+
- Docker Compose v2.20+
- At least 4GB RAM allocated to Docker

## Quick Start

```bash
# Clone the repository
git clone https://github.com/AseemPrasad/Legalassist-AI.git
cd Legalassist-AI

# Copy environment configuration
cp .env.example .env

# Start all services
docker compose up -d

# View logs
docker compose logs -f
```

## Service Breakdown

| Service | Container Name | Port | Description |
|---------|---------------|------|-------------|
| **Streamlit UI** | `legalassist-ui` | 8501 | Main user interface |
| **FastAPI** | `legalassist-api` | 8000 | REST API and WebSocket server |
| **Celery Worker** | `legalassist-worker` | — | Background task processing |
| **Redis** | `legalassist-redis` | 6379 | Message broker and cache |
| **PostgreSQL** | `legalassist-db` | 5432 | Primary database |

## Environment Variables

Configure via `.env` file based on `.env.example`:

```
DATABASE_URL=postgresql://user:pass@db:5432/legalassist
REDIS_URL=redis://redis:6379/0
JWT_SECRET=your-secret-key-here
DEBUG=true
```

## Running Individual Services

```bash
# Start only the database
docker compose up -d db redis

# Start API with dependencies
docker compose up -d api

# Start UI with all dependencies
docker compose up -d ui
```

## Hot Reload

Volumes are mounted for source code to enable hot-reload:

```yaml
volumes:
  - ./api:/app/api        # FastAPI hot-reload
  - ./pages:/app/pages    # Streamlit hot-reload
  - ./core:/app/core      # Core modules
```

## Running Tests in Docker

```bash
# Run all tests
docker compose exec api pytest

# Run specific test file
docker compose exec api pytest tests/test_api_versioning.py -v

# Run with coverage
docker compose exec api pytest --cov=. --cov-report=term
```

## Database Management

```bash
# Access PostgreSQL
docker compose exec db psql -U user -d legalassist

# Run migrations
docker compose exec api alembic upgrade head

# Create migration
docker compose exec api alembic revision --autogenerate -m "description"
```

## Common Issues

### Port Already in Use
```bash
# Change host port in docker-compose.yml
services:
  ui:
    ports:
      - "8502:8501"  # Change 8502 to any available port
```

### Permission Denied on Volumes
```bash
# On Linux, ensure write permissions
sudo chown -R 1000:1000 ./data
```

### Docker Out of Disk Space
```bash
docker system prune -a --volumes
```

## Production Docker

For production deployment:

```bash
# Use production compose file
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With API-only stack
docker compose -f docker-compose-api.yml up -d
```

See `docs/deployment.md` for full production deployment guide.
