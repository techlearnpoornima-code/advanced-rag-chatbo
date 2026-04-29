# Deployment Skill

## Auto-triggered when:
- User mentions "deploy", "deployment", "production"
- Changes to docker/ files
- Updates to requirements.txt
- Discussion of cloud platforms (AWS, GCP, Azure)

## Knowledge Base

### Pre-deployment Checklist

```bash
# 1. Run tests
pytest tests/ -v

# 2. Check code style
black src/ app/ --check
ruff check src/ app/

# 3. Verify environment config
cat .env.example  # Ensure all required vars documented

# 4. Build Docker image
docker build -f docker/Dockerfile -t rag-chatbot:latest .

# 5. Test Docker image locally
docker run -p 8000:8000 --env-file .env rag-chatbot:latest

# 6. Check health endpoint
curl http://localhost:8000/api/v1/health
```

### Deployment Options

#### 1. Local/Development

```bash
# Virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or with gunicorn (production-like)
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

#### 2. Docker Compose

```bash
# Build and start
cd docker
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

#### 3. AWS ECS/Fargate

```bash
# Build and tag
docker build -t rag-chatbot:latest .
docker tag rag-chatbot:latest $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/rag-chatbot:latest

# Push to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker push $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/rag-chatbot:latest

# Create task definition (via AWS Console or CLI)
# Create ECS service with ALB
# Configure auto-scaling
```

#### 4. Google Cloud Run

```bash
# Build and submit
gcloud builds submit --tag gcr.io/$PROJECT_ID/rag-chatbot

# Deploy
gcloud run deploy rag-chatbot \
  --image gcr.io/$PROJECT_ID/rag-chatbot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --timeout 300 \
  --set-env-vars ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
```

#### 5. Azure Container Instances

```bash
# Login to ACR
az acr login --name $ACR_NAME

# Build and push
docker build -t rag-chatbot:latest .
docker tag rag-chatbot:latest $ACR_NAME.azurecr.io/rag-chatbot:latest
docker push $ACR_NAME.azurecr.io/rag-chatbot:latest

# Deploy
az container create \
  --resource-group $RG \
  --name rag-chatbot \
  --image $ACR_NAME.azurecr.io/rag-chatbot:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
```

### Environment Variables

Required for production:

```bash
# API Keys (REQUIRED)
ANTHROPIC_API_KEY=sk-ant-...

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=claude-sonnet-4-20250514
MAX_TOKENS=4096

# Performance
TOP_K=10
ENABLE_HYBRID_SEARCH=true
ENABLE_MULTI_INTENT=true
WORKERS=4

# Security
ALLOWED_ORIGINS=https://yourdomain.com
LOG_LEVEL=INFO

# Optional: Redis for caching
REDIS_URL=redis://redis:6379
```

### Health Checks

Configure health checks in your deployment:

```yaml
# Docker Compose example
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/ping"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Monitoring Setup

```python
# Prometheus metrics already available at /metrics

# Example Prometheus config:
scrape_configs:
  - job_name: 'rag-chatbot'
    static_configs:
      - targets: ['rag-chatbot:8000']
    metrics_path: '/metrics'
```

### Scaling Guidelines

| Load          | CPU  | RAM  | Workers | Vector DB |
|---------------|------|------|---------|-----------|
| Low (< 1K/d)  | 2    | 4GB  | 2-4     | 5GB       |
| Medium (1-10K)| 4    | 8GB  | 4-8     | 20GB      |
| High (10-100K)| 8    | 16GB | 8-16    | 50GB      |

### Rollback Procedure

```bash
# Docker Compose
docker-compose down
git checkout previous-tag
docker-compose up -d

# Cloud Run
gcloud run services update rag-chatbot \
  --image gcr.io/$PROJECT_ID/rag-chatbot:previous-version

# ECS
aws ecs update-service \
  --cluster rag-cluster \
  --service rag-service \
  --task-definition rag-chatbot:previous-revision
```

### Post-Deployment Verification

```bash
# 1. Check health
curl https://api.yourdomain.com/api/v1/health

# 2. Test chat endpoint
curl -X POST https://api.yourdomain.com/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FastAPI?"}'

# 3. Check metrics
curl https://api.yourdomain.com/api/v1/metrics

# 4. Monitor logs
# AWS: CloudWatch Logs
# GCP: Cloud Logging
# Azure: Application Insights

# 5. Load test (optional)
ab -n 1000 -c 10 https://api.yourdomain.com/api/v1/health
```

## Common Issues & Solutions

### Issue: Container fails to start
```bash
# Check logs
docker logs container-id

# Common causes:
# - Missing ANTHROPIC_API_KEY
# - Port already in use
# - Insufficient memory

# Solution:
docker run -p 8001:8000 --env-file .env rag-chatbot:latest
```

### Issue: High memory usage
```bash
# Reduce chunk size in .env
CHUNK_SIZE=256
MAX_CHUNKS_PER_DOC=25

# Reduce workers
WORKERS=2
```

### Issue: Slow responses
```bash
# Enable caching
ENABLE_CACHE=true

# Reduce retrieval
TOP_K=5
RERANK_TOP_K=3
```

## Deployment Guidance

When user asks about deployment:

1. **Clarify target environment**:
   - "Are you deploying locally, to cloud, or using Docker?"
   - "Which cloud provider? (AWS/GCP/Azure)"

2. **Check prerequisites**:
   - Docker installed?
   - Cloud CLI configured?
   - API keys available?

3. **Provide specific steps**:
   - Use code blocks with exact commands
   - Include verification steps
   - Mention common pitfalls

4. **Offer to help troubleshoot**:
   - "After running this, let me know if you hit any errors"
   - "Check the logs if it doesn't start"

5. **Security reminders**:
   - "Don't commit your .env file"
   - "Use secrets manager for API keys in production"
   - "Enable HTTPS for production deployments"
