# Quick Start - Docker Deployment

## 🚀 Deploy Locally with Docker

### Option 1: Quick Deploy (Recommended)
```powershell
.\deploy.ps1
```
Access at: http://localhost:8501

### Option 2: Docker Compose
```bash
docker-compose up -d
```

### Option 3: Manual Docker Commands
```bash
# Build
docker build -t algo-trading-system:latest .

# Run
docker run -d -p 8501:8501 --name algo-trading-system algo-trading-system:latest

# View logs
docker logs -f algo-trading-system
```

## ☁️ Deploy to Cloud

### Azure
```powershell
# Update variables in deploy-azure.ps1 first
.\deploy-azure.ps1
```

### Google Cloud
```powershell
# Update PROJECT_ID in deploy-gcp.ps1 first
.\deploy-gcp.ps1
```

## 📊 Verify Deployment

### Check Container Status
```bash
docker ps
```

### View Logs
```bash
docker logs -f algo-trading-system
```

### Access Application
- Local: http://localhost:8501
- Azure: http://<dns-name>.<region>.azurecontainer.io:8501
- GCP: Provided after deployment

## 🛑 Stop and Clean Up

```bash
# Stop container
docker stop algo-trading-system

# Remove container
docker rm algo-trading-system

# Remove image
docker rmi algo-trading-system:latest
```

## 📝 Notes

- First run downloads ~250MB DistilBERT model
- Data persists in ./data directory
- See DEPLOYMENT.md for detailed documentation
