# Salesforce Lead Scoring Platform with MLflow & FastAPI

Complete production-ready machine learning platform for lead scoring (1-5 scale) with MLflow model registry and FastAPI microservice architecture.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Nginx (Port 80)                      │
│                    Reverse Proxy & Load Balancer             │
└─────────────┬───────────────────────┬───────────────────────┘
              │                       │
              │                       │
    ┌─────────▼─────────┐   ┌────────▼──────────┐
    │   FastAPI API     │   │   MLflow Server   │
    │   (Port 8000)     │   │   (Port 5000)     │
    │                   │   │                   │
    │ • Lead Scoring    │   │ • Model Registry  │
    │ • Model Loading   │   │ • Experiment      │
    │ • Predictions     │   │   Tracking        │
    └─────────┬─────────┘   └────────┬──────────┘
              │                      │
              │                      │
    ┌─────────▼──────────────────────▼──────────┐
    │                                            │
    │         DuckDB (Port 5432)                 │
    │         MLflow Metadata Store              │
    │                                            │
    └────────────────────────────────────────────┘
              │
              │
    ┌─────────▼──────────────────────┐
    │     MinIO (Port 9000/9001)     │
    │     S3-Compatible Storage      │
    │     MLflow Artifacts           │
    └────────────────────────────────┘
```

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 10GB disk space

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create project structure
mkdir salesforce-lead-scoring-platform
cd salesforce-lead-scoring-platform

# Create directories
git clone git@github.com:J0MS/reimagined-fishstick-salesforce-exec.git

# Copy files to appropriate directories
# - docker-compose.yml → root
# - mlflow-server/Dockerfile → mlflow-server/
# - fastapi-service/Dockerfile → fastapi-service/
# - fastapi-service/requirements.txt → fastapi-service/
# - fastapi-service/app/main.py → fastapi-service/app/
# - nginx/nginx.conf → nginx/
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configurations
nano .env
```

### 3. Start Services

```bash
# Build and start all services
make build
make up

# Or with docker-compose directly
docker-compose build && docker-compose up api mlflow-server
```

### 4. Verify Services

```bash
# Check service health
make health

# Or manually
curl http://localhost:8000/health
curl http://localhost:5000/health
```

Aditionally, check Swagger documentation on ```bash http://0.0.0.0:8000/docs```

## 📍 Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| FastAPI API | http://localhost:8000 | Lead scoring API |
| API Documentation | http://localhost:8000/docs | Interactive Swagger UI |
| MLflow UI | http://localhost:5000 | Model registry & experiments |
| MinIO Console | http://localhost:9001 | S3 storage management |
| Nginx (API) | http://localhost/api | Reverse proxy to FastAPI |
| Nginx (MLflow) | http://localhost/mlflow | Reverse proxy to MLflow |


### Making Predictions

```bash
# Single prediction
curl -X 'POST' \
  'http://0.0.0.0:8000/v1.0/compute' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "LEAD_ID": 0,
  "MARKET": "string",
  "LEAD_PARAMETERS": [{
                "email_opens": 10,
                "email_clicks": 5,
                "page_views": 25,
                "content_downloads": 3,
                "demo_requested": 1,
                "pricing_page_visits": 4,
                "case_study_views": 2,
                "days_since_last_activity": 2.5,
                "company_size": 500,
                "annual_revenue": 5000000,
                "is_decision_maker": 1,
                "job_level_score": 4,
                "session_duration_avg": 8.5,
                "pages_per_session": 5,
                "return_visitor": 1
            }]
}'
```

### Managing Models

```bash
# List registered models
make model-list

# Promote model to production
make model-promote

# Reload model in API (after promotion)
make model-reload
```

## 🛠️ Makefile Commands

```bash
make help           # Show all available commands
make build          # Build Docker images
make up             # Start all services
make down           # Stop all services
make restart        # Restart services
make logs           # View all logs
make logs-api       # View FastAPI logs
make logs-mlflow    # View MLflow logs
make clean          # Remove containers & volumes
make train          # Train new model
make test           # Run tests
make health         # Check service health
make ps             # Show running containers
make urls           # Display all service URLs
```

## 🔄 Model Deployment Workflow

### 1. Train Model
```bash
# Train and register model in MLflow
make train
```

### 2. View in MLflow UI
- Navigate to http://localhost:5000
- View experiments, runs, and metrics
- Check model in registry

### 3. Promote to Production
```bash
# Promote specific version to Production
make model-promote
# Enter version number when prompted
```

### 4. Reload in API
```bash
# API will automatically load new production model
make model-reload
```

### 5. Verify
```bash
# Check model info
curl http://localhost:8000/model/info
```

## 📊 Monitoring & Observability

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f mlflow-server
```

### Container Status
```bash
make ps
# Or
docker-compose ps
```

### Resource Usage
```bash
docker stats
```

## 🧪 Testing

### API Tests
```bash
# Run all tests
make test

# Run specific test
docker-compose exec fastapi-service pytest tests/test_api.py -v
```

### Manual Testing
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_lead.json
```

## 🔧 Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# MLflow
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MODEL_NAME=lead-scoring-xgboost
MODEL_STAGE=Production

# API
API_WORKERS=4
LOG_LEVEL=INFO

# Database
POSTGRES_PASSWORD=mlflow_password

# Storage
MINIO_ROOT_PASSWORD=minio_password
```

### Scaling FastAPI

Edit `docker-compose.yml`:

```yaml
fastapi-service:
  environment:
    API_WORKERS: 8  # Increase workers
  deploy:
    replicas: 3  # Multiple containers
```

## 🐛 Troubleshooting

### MLflow Server Not Starting

```bash
# Check PostgreSQL is healthy
docker-compose ps postgres

# Check logs
docker-compose logs mlflow-server

# Restart
docker-compose restart mlflow-server
```

### Model Not Loading in API

```bash
# Check MLflow connection
curl http://localhost:5000/health

# Check model exists
curl http://localhost:5000/api/2.0/mlflow/registered-models/get?name=lead-scoring-xgboost

# Force reload
make model-reload
```

### MinIO Connection Issues

```bash
# Check MinIO is running
docker-compose ps minio

# Verify bucket exists
docker-compose exec minio-client mc ls myminio

# Create bucket manually
docker-compose exec minio-client mc mb myminio/mlflow
```

### Database Connection Issues

```bash
# Check PostgreSQL
docker-compose exec postgres psql -U mlflow -d mlflow

# Run migrations
make db-migrate
```

## 🚀 Production Deployment

### Security Hardening

1. **Change Default Passwords**
```bash
# Update in .env
POSTGRES_PASSWORD=<strong-password>
MINIO_ROOT_PASSWORD=<strong-password>
```

2. **Enable HTTPS**
```bash
# Generate SSL certificates
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem

# Uncomment HTTPS server in nginx.conf
```

3. **Add API Authentication**
- Implement JWT tokens
- Add API key validation
- Rate limiting (already configured in Nginx)

### Resource Limits

Add to `docker-compose.yml`:

```yaml
services:
  fastapi-service:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

### Backup Strategy

```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U mlflow mlflow > backup.sql

# Backup MinIO
docker-compose exec minio-client mc mirror myminio/mlflow /backup/mlflow
```

## 📚 API Documentation

Full API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Current model info |
| `/model/reload` | POST | Reload model from registry |
| `/predict` | POST | Single lead prediction |
| `/predict/batch` | POST | Batch predictions |
| `/mlflow/experiments` | GET | List MLflow experiments |
| `/mlflow/models` | GET | List registered models |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Open an issue on GitHub
- Check MLflow docs: https://mlflow.org/docs/latest/
- Check FastAPI docs: https://fastapi.tiangolo.com/

## 🎯 Next Steps

- [ ] Add authentication & authorization
- [ ] Implement model A/B testing
- [ ] Add Prometheus metrics
- [ ] Set up CI/CD pipeline
- [ ] Add model drift detection
- [ ] Implement feature store
- [ ] Add Kubernetes deployment configs
