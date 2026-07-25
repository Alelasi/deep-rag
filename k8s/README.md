# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying DeepRAG in a production environment.

## 📋 Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured
- Helm (optional, for dependencies)
- Ingress controller (nginx recommended)
- Cert-manager (for TLS)

## 📁 Files

- `deployment.yaml` - Main application deployment, service, configmap, secrets, PVC
- `ingress.yaml` - Ingress configuration for external access
- `hpa.yaml` - Horizontal Pod Autoscaler for auto-scaling
- `qdrant.yaml` - Qdrant vector database deployment
- `postgres.yaml` - PostgreSQL with pgvector deployment

## 🚀 Quick Start

### 1. Create Namespace

```bash
kubectl create namespace deep-rag
kubectl config set-context --current --namespace=deep-rag
```

### 2. Update Secrets

Edit `deployment.yaml` and update the secrets:

```yaml
stringData:
  anthropic-api-key: "your-actual-api-key"
  postgres-password: "your-secure-password"
```

### 3. Deploy Dependencies

```bash
# Deploy Qdrant
kubectl apply -f qdrant.yaml

# Deploy PostgreSQL
kubectl apply -f postgres.yaml

# Wait for dependencies to be ready
kubectl wait --for=condition=ready pod -l app=qdrant --timeout=300s
kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s
```

### 4. Deploy DeepRAG

```bash
# Deploy application
kubectl apply -f deployment.yaml

# Deploy ingress
kubectl apply -f ingress.yaml

# Deploy autoscaler
kubectl apply -f hpa.yaml
```

### 5. Verify Deployment

```bash
# Check pods
kubectl get pods

# Check services
kubectl get svc

# Check ingress
kubectl get ingress

# View logs
kubectl logs -l app=deep-rag --tail=100 -f
```

## 🔧 Configuration

### Environment Variables

Configure via ConfigMap in `deployment.yaml`:

```yaml
data:
  llm-backend: "anthropic"      # anthropic, openai, ollama
  vector-db: "qdrant"            # chromadb, qdrant, pgvector
  enable-agentic-rag: "true"
  enable-reranker: "false"
```

### Resource Limits

Adjust resources based on your workload:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Auto-scaling

Configure HPA in `hpa.yaml`:

```yaml
minReplicas: 2
maxReplicas: 10
metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

## 🔍 Monitoring

### Health Checks

- **Liveness**: `GET /health` - Basic health check
- **Readiness**: `GET /ready` - Dependency checks
- **Metrics**: `GET /metrics` - Prometheus metrics

### View Metrics

```bash
# Port-forward to access metrics
kubectl port-forward svc/deep-rag-service 8000:8000

# Access metrics
curl http://localhost:8000/metrics
```

## 📊 Scaling

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment deep-rag --replicas=5
```

### Auto-scaling

HPA automatically scales based on CPU/memory usage:

```bash
# Check HPA status
kubectl get hpa

# Describe HPA
kubectl describe hpa deep-rag-hpa
```

## 🔐 Security

### Secrets Management

**Production**: Use external secret managers:

```bash
# Using Sealed Secrets
kubeseal --format=yaml < secrets.yaml > sealed-secrets.yaml
kubectl apply -f sealed-secrets.yaml

# Using External Secrets Operator
kubectl apply -f external-secret.yaml
```

### Network Policies

Apply network policies to restrict traffic:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deep-rag-netpol
spec:
  podSelector:
    matchLabels:
      app: deep-rag
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
```

## 🔄 Updates

### Rolling Update

```bash
# Update image
kubectl set image deployment/deep-rag deep-rag=deep-rag:2.2.0

# Check rollout status
kubectl rollout status deployment/deep-rag

# Rollback if needed
kubectl rollout undo deployment/deep-rag
```

## 🐛 Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name>

# View logs
kubectl logs <pod-name>

# Check events
kubectl get events --sort-by='.lastTimestamp'
```

### Service Not Accessible

```bash
# Check service endpoints
kubectl get endpoints deep-rag-service

# Test service internally
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://deep-rag-service:8000/health
```

### Database Connection Issues

```bash
# Check Qdrant
kubectl exec -it <qdrant-pod> -- curl http://localhost:6333/health

# Check PostgreSQL
kubectl exec -it <postgres-pod> -- psql -U postgres -d deeprag -c "SELECT 1;"
```

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Charts](https://helm.sh/)
- [Ingress NGINX](https://kubernetes.github.io/ingress-nginx/)
- [Cert-manager](https://cert-manager.io/)

## 🙏 Support

For issues or questions, please open a GitHub issue.
