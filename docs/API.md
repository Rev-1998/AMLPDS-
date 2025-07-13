# docs/API.md
# AMLPDS API Documentation

## Overview

The AMLPDS API provides RESTful endpoints for phishing email detection and analysis. All API endpoints require JWT authentication except for health checks.

## Authentication

### Login
```http
POST /auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "user": {
    "username": "string",
    "email": "string",
    "role": "string"
  }
}
```

### Register
```http
POST /auth/register
Content-Type: application/json

{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

### Refresh Token
```http
POST /auth/refresh
Authorization: Bearer <refresh_token>
```

## Prediction Endpoints

### Single Email Prediction
```http
POST /api/predict
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "email_text": "string",
  "urls": ["string"] // optional
}
```

**Response:**
```json
{
  "prediction": "phishing|legitimate",
  "confidence": 0.95,
  "risk_level": "CRITICAL|HIGH|MEDIUM|LOW",
  "explanation": {
    "summary": "string",
    "suspicious_keywords": [
      {
        "keyword": "string",
        "position": 0,
        "context": "string",
        "severity": 5
      }
    ],
    "recommendations": ["string"],
    "pattern_analysis": {
      "urgency_language": true,
      "action_required": true,
      "threat_language": false,
      "financial_language": true
    }
  },
  "features": {
    "text_length": 500,
    "urgency_score": 0.8,
    "financial_score": 0.6
  },
  "timestamp": "2025-07-13T12:00:00Z"
}
```

### Batch Prediction
```http
POST /api/predict/batch
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "emails": [
    {
      "id": "string",
      "text": "string"
    }
  ]
}
```

### File Analysis
```http
POST /api/analyze/file
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

file: <file>
```

## Reporting

### Report Phishing
```http
POST /api/report
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "email_text": "string",
  "reported_as": "phishing|legitimate",
  "notes": "string"
}
```

## System Information

### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy|unhealthy",
  "models_loaded": true,
  "timestamp": "2025-07-13T12:00:00Z"
}
```

### Statistics
```http
GET /api/stats
Authorization: Bearer <access_token>
```

### Model Information
```http
GET /api/model/info
Authorization: Bearer <access_token>
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "string",
  "message": "string"
}
```

### HTTP Status Codes

- `200 OK` - Request successful
- `201 Created` - Resource created
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required or failed
- `403 Forbidden` - Access denied
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

## Rate Limiting

Default rate limits:
- 200 requests per day per IP
- 50 requests per hour per IP
- 10 requests per minute per authenticated user

## Examples

### Python Example
```python
import requests

# Login
response = requests.post('https://api.amlpds.com/auth/login', json={
    'username': 'user',
    'password': 'password'
})
token = response.json()['access_token']

# Predict
headers = {'Authorization': f'Bearer {token}'}
response = requests.post('https://api.amlpds.com/api/predict', 
    headers=headers,
    json={'email_text': 'Urgent: Verify your account'})

result = response.json()
print(f"Prediction: {result['prediction']}")
```

### JavaScript Example
```javascript
// Login
const loginResponse = await fetch('/auth/login', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({username: 'user', password: 'password'})
});
const {access_token} = await loginResponse.json();

// Predict
const response = await fetch('/api/predict', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${access_token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({email_text: 'Test email'})
});

const result = await response.json();
console.log(`Prediction: ${result.prediction}`);
```

---

# docs/ARCHITECTURE.md
# AMLPDS System Architecture

## Overview

AMLPDS follows a modular, layered architecture designed for scalability, maintainability, and security.

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   HTML/CSS  │  │ JavaScript  │  │  Tailwind   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Web Server Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    Flask    │  │   Gunicorn  │  │    Nginx    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │     API     │  │    Auth     │  │   Security  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Service Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Preprocessor│  │  Classifier │  │ AI Assistant│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        ML Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │     BERT    │  │ TextAttack  │  │    SHAP     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  PostgreSQL │  │    Redis    │  │ File System │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend Layer
- **Technology**: HTML5, CSS3, JavaScript, Tailwind CSS
- **Responsibilities**:
  - User interface rendering
  - Form validation
  - API communication
  - Real-time updates

### 2. Web Server Layer
- **Flask**: Application framework
- **Gunicorn**: WSGI HTTP server
- **Nginx**: Reverse proxy and static file serving
- **Features**:
  - SSL/TLS termination
  - Load balancing
  - Request routing
  - Static asset caching

### 3. Application Layer
- **API Endpoints**: RESTful API design
- **Authentication**: JWT-based authentication
- **Security**: CSRF protection, rate limiting, input validation

### 4. Service Layer
- **Email Preprocessor**: Text cleaning and feature extraction
- **BERT Classifier**: Phishing detection model
- **AI Assistant**: Explainable AI using SHAP

### 5. ML Layer
- **BERT Model**: Pre-trained transformer for text classification
- **TextAttack**: Adversarial training framework
- **SHAP**: Model explainability

### 6. Data Layer
- **PostgreSQL**: User data and analysis history
- **Redis**: Session storage and caching
- **File System**: Model storage and logs

## Data Flow

1. **User Request** → Nginx → Gunicorn → Flask Application
2. **Authentication** → JWT validation → User identification
3. **Email Processing** → Preprocessing → Feature extraction
4. **Prediction** → BERT model → Confidence scoring
5. **Explanation** → SHAP analysis → Recommendation generation
6. **Response** → JSON formatting → Client

## Security Architecture

### Authentication Flow
```
User → Login → JWT Generation → Token Storage → API Access
                     ↓
                 Refresh Token → Token Renewal
```

### Security Layers
1. **Network Security**
   - HTTPS/TLS encryption
   - Firewall rules
   - DDoS protection

2. **Application Security**
   - Input validation
   - SQL injection prevention
   - XSS protection
   - CSRF tokens

3. **Authentication & Authorization**
   - JWT tokens
   - Role-based access control
   - Session management

4. **Data Security**
   - Encryption at rest
   - Secure key management
   - Audit logging

## Scalability Considerations

### Horizontal Scaling
- Multiple Flask workers
- Load balancer distribution
- Stateless application design

### Vertical Scaling
- GPU acceleration for ML models
- Memory optimization
- Database connection pooling

### Caching Strategy
- Redis for session data
- Model predictions caching
- Static asset CDN

## Deployment Architecture

### Development
```
Local Machine → Flask Dev Server → SQLite → Local Model
```

### Production
```
Docker Compose → Nginx → Gunicorn → PostgreSQL → Redis
       ↓
   Kubernetes → Auto-scaling → Load Balancing
```

## Monitoring & Logging

### Application Metrics
- Request rate
- Response time
- Error rate
- Model accuracy

### System Metrics
- CPU usage
- Memory usage
- Disk I/O
- Network throughput

### Logging
- Application logs → File system
- Access logs → Nginx
- Error logs → Centralized logging
- Audit logs → Database

## API Design Principles

1. **RESTful Design**
   - Resource-based URLs
   - HTTP methods (GET, POST, PUT, DELETE)
   - Stateless communication

2. **Versioning**
   - URL versioning (/api/v1/)
   - Backward compatibility

3. **Error Handling**
   - Consistent error format
   - Meaningful error messages
   - HTTP status codes

4. **Documentation**
   - OpenAPI/Swagger specification
   - Code examples
   - Rate limit information

---

# docs/DEPLOYMENT.md
# AMLPDS Deployment Guide

## Prerequisites

- Ubuntu 20.04+ or similar Linux distribution
- Docker and Docker Compose installed
- Domain name (for production)
- SSL certificate (Let's Encrypt recommended)

## Development Deployment

### 1. Local Development
```bash
# Clone repository
git clone https://github.com/yourusername/AMLPDS.git
cd AMLPDS

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run development server
./run_dev.sh
```

### 2. Docker Development
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:5000
```

## Production Deployment

### 1. Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo pip3 install docker-compose
```

### 2. SSL Certificate

```bash
# Obtain Let's Encrypt certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### 3. Environment Configuration

Create `.env.production`:
```env
FLASK_ENV=production
SECRET_KEY=your-production-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
DATABASE_URL=postgresql://user:password@localhost/amlpds
REDIS_URL=redis://localhost:6379/0
```

### 4. Database Setup

```bash
# Create PostgreSQL database
sudo -u postgres createdb amlpds
sudo -u postgres createuser -P amlpds_user

# Grant permissions
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE amlpds TO amlpds_user;"
```

### 5. Deploy with Docker

```bash
# Copy files to server
scp -r ./* user@server:/opt/amlpds/

# On server
cd /opt/amlpds

# Build and run
docker-compose -f docker-compose.yml up -d

# Check logs
docker-compose logs -f
```

### 6. Nginx Configuration

Create `/etc/nginx/sites-available/amlpds`:
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /opt/amlpds/static;
        expires 30d;
    }
}
```

### 7. Systemd Service

Create `/etc/systemd/system/amlpds.service`:
```ini
[Unit]
Description=AMLPDS Phishing Detection System
After=network.target

[Service]
Type=simple
User=amlpds
WorkingDirectory=/opt/amlpds
Environment="PATH=/opt/amlpds/venv/bin"
ExecStart=/opt/amlpds/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 src.web.app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

## Kubernetes Deployment

### 1. Create Deployment

`k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: amlpds
spec:
  replicas: 3
  selector:
    matchLabels:
      app: amlpds
  template:
    metadata:
      labels:
        app: amlpds
    spec:
      containers:
      - name: amlpds
        image: amlpds:latest
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### 2. Create Service

`k8s/service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: amlpds-service
spec:
  selector:
    app: amlpds
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
```

### 3. Deploy to Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods
kubectl get services
```

## Monitoring Setup

### 1. Install Prometheus

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/prometheus
```

### 2. Install Grafana

```bash
# Install Grafana
helm install grafana grafana/grafana

# Get admin password
kubectl get secret --namespace default grafana -o jsonpath="{.data.admin-password}" | base64 --decode
```

### 3. Application Metrics

Add to Flask app:
```python
from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)
```

## Backup Strategy

### 1. Database Backup

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup PostgreSQL
pg_dump -U amlpds_user amlpds > $BACKUP_DIR/amlpds_$DATE.sql

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz /opt/amlpds/data/models/

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/amlpds_$DATE.sql s3://your-backup-bucket/
```

### 2. Automated Backups

Add to crontab:
```bash
# Daily backups at 2 AM
0 2 * * * /opt/amlpds/scripts/backup.sh
```

## Performance Optimization

### 1. Model Optimization

```python
# Use ONNX for faster inference
import torch.onnx

# Convert model to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")
```

### 2. Caching

```python
# Redis caching
import redis
from functools import lru_cache

redis_client = redis.Redis()

@lru_cache(maxsize=1000)
def predict_with_cache(email_hash):
    # Check Redis cache
    cached = redis_client.get(email_hash)
    if cached:
        return json.loads(cached)
    
    # Compute prediction
    result = model.predict(email)
    
    # Cache result
    redis_client.setex(email_hash, 3600, json.dumps(result))
    return result
```

### 3. Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);

-- Analyze tables
ANALYZE users;
ANALYZE predictions;
```

## Troubleshooting

### Common Issues

1. **Model not loading**
   ```bash
   # Check model file exists
   ls -la data/models/
   
   # Check permissions
   chmod 644 data/models/bert_phishing_classifier.pt
   ```

2. **Out of memory**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=8
   
   # Use gradient checkpointing
   model.gradient_checkpointing_enable()
   ```

3. **Slow predictions**
   ```bash
   # Enable GPU
   export CUDA_VISIBLE_DEVICES=0
   
   # Use mixed precision
   from torch.cuda.amp import autocast
   ```

### Logs

```bash
# Application logs
tail -f logs/amlpds.log

# Docker logs
docker-compose logs -f web

# System logs
journalctl -u amlpds -f
```

## Security Hardening

### 1. Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. Fail2ban

```bash
# Install fail2ban
sudo apt install fail2ban

# Configure for Flask
echo '[amlpds]
enabled = true
port = http,https
filter = amlpds
logpath = /opt/amlpds/logs/access.log
maxretry = 5' | sudo tee /etc/fail2ban/jail.local
```

### 3. Security Headers

Add to Nginx:
```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
```

---

# CONTRIBUTING.md
# Contributing to AMLPDS

We welcome contributions to AMLPDS! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## How to Contribute

### 1. Fork the Repository

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/yourusername/AMLPDS.git
cd AMLPDS
git remote add upstream https://github.com/originalowner/AMLPDS.git
```

### 2. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b bugfix/issue-description
```

### 3. Make Changes

- Follow the coding standards
- Write tests for new features
- Update documentation
- Ensure all tests pass

### 4. Commit Changes

```bash
# Use meaningful commit messages
git commit -m "feat: add new phishing detection algorithm"

# Commit message format:
# type(scope): description
#
# Types: feat, fix, docs, style, refactor, test, chore
```

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## Development Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### 3. Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py::TestBERTClassifier
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with these additions:

- Maximum line length: 100 characters
- Use type hints where appropriate
- Document all public functions

```python
def predict_phishing(email_text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Predict if email is phishing.
    
    Args:
        email_text: The email content to analyze
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Dictionary containing prediction and confidence
        
    Raises:
        ValueError: If email_text is empty
    """
    if not email_text:
        raise ValueError("Email text cannot be empty")
    
    # Implementation
    return {"prediction": "phishing", "confidence": 0.95}
```

### JavaScript Style Guide

- Use ES6+ features
- Async/await over promises
- Meaningful variable names

```javascript
// Good
async function analyzeEmail(emailText) {
    try {
        const response = await api.predict(emailText);
        return response.data;
    } catch (error) {
        console.error('Analysis failed:', error);
        throw error;
    }
}

// Bad
function analyze(e) {
    return api.predict(e).then(r => r.data);
}
```

## Testing Guidelines

### 1. Unit Tests

```python
def test_predict_phishing_email():
    """Test phishing email detection"""
    classifier = BERTPhishingClassifier()
    
    phishing_email = "Urgent: Verify your account now!"
    result = classifier.predict(phishing_email)
    
    assert result['prediction'] == 'phishing'
    assert result['confidence'] > 0.8
```

### 2. Integration Tests

```python
def test_api_prediction_endpoint(client, auth_headers):
    """Test prediction API endpoint"""
    response = client.post('/api/predict',
                         headers=auth_headers,
                         json={'email_text': 'Test email'})
    
    assert response.status_code == 200
    assert 'prediction' in response.json
```

### 3. Performance Tests

```python
def test_prediction_performance(benchmark):
    """Test prediction speed"""
    classifier = BERTPhishingClassifier()
    email = "Test email content"
    
    result = benchmark(classifier.predict, email)
    assert result['prediction'] in ['phishing', 'legitimate']
```

## Documentation

### 1. Code Documentation

- Use docstrings for all public functions
- Include parameter descriptions
- Provide usage examples

### 2. API Documentation

- Update `docs/API.md` for new endpoints
- Include request/response examples
- Document error codes

### 3. User Documentation

- Update README.md for new features
- Add screenshots for UI changes
- Update installation instructions

## Pull Request Process

### 1. Before Submitting

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### 2. PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots here

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where necessary
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
```

### 3. Review Process

- PRs require at least one review
- Address all feedback
- Squash commits if requested
- Update PR based on feedback

## Reporting Issues

### 1. Bug Reports

Include:
- Python version
- OS information
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/logs

### 2. Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative solutions
- Additional context

## Release Process

### 1. Version Numbering

We use Semantic Versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### 2. Release Checklist

- [ ] Update version in `setup.py`
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Build and test Docker image
- [ ] Tag release in Git
- [ ] Create GitHub release
- [ ] Update documentation

## Getting Help

- Check existing issues
- Read the documentation
- Ask in discussions
- Contact maintainers

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to AMLPDS!
