#!/bin/bash

# UPI Fraud Detection - Kubernetes Deployment Script
# This script deploys the fraud detection system to Kubernetes

set -e

# Configuration
NAMESPACE="fraud-detection"
IMAGE_NAME="fraud-detection"
IMAGE_TAG="latest"
REGISTRY="your-registry.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting UPI Fraud Detection Deployment${NC}"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl is not installed or not in PATH${NC}"
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Kubernetes cluster is accessible${NC}"

# Build Docker image
echo -e "${YELLOW}📦 Building Docker image...${NC}"
cd ../serving
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Tag and push to registry (uncomment if using remote registry)
# docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
# docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

cd ../infra

# Create namespace if it doesn't exist
echo -e "${YELLOW}🏗️  Creating namespace...${NC}"
kubectl apply -f k8s/namespace.yaml

# Apply configurations
echo -e "${YELLOW}⚙️  Applying configurations...${NC}"
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml

# Deploy infrastructure components
echo -e "${YELLOW}🗄️  Deploying Redis...${NC}"
kubectl apply -f k8s/redis-deployment.yaml

echo -e "${YELLOW}🐘 Deploying PostgreSQL...${NC}"
kubectl apply -f k8s/postgres-deployment.yaml

# Wait for infrastructure to be ready
echo -e "${YELLOW}⏳ Waiting for infrastructure to be ready...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment/redis -n ${NAMESPACE}
kubectl wait --for=condition=available --timeout=300s deployment/postgres -n ${NAMESPACE}

# Deploy the main application
echo -e "${YELLOW}🚀 Deploying Fraud Detection API...${NC}"
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Deploy ingress
echo -e "${YELLOW}🌐 Setting up ingress...${NC}"
kubectl apply -f k8s/ingress.yaml

# Deploy monitoring
echo -e "${YELLOW}📊 Setting up monitoring...${NC}"
kubectl apply -f k8s/monitoring.yaml

# Wait for deployment to be ready
echo -e "${YELLOW}⏳ Waiting for deployment to be ready...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment/fraud-detection-api -n ${NAMESPACE}

# Get service information
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${GREEN}📋 Service Information:${NC}"
kubectl get services -n ${NAMESPACE}
echo ""
kubectl get pods -n ${NAMESPACE}
echo ""

# Get ingress information
INGRESS_IP=$(kubectl get ingress fraud-detection-ingress -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -n "$INGRESS_IP" ]; then
    echo -e "${GREEN}🌐 API accessible at: http://${INGRESS_IP}${NC}"
    echo -e "${GREEN}📚 API Documentation: http://${INGRESS_IP}/docs${NC}"
    echo -e "${GREEN}❤️  Health Check: http://${INGRESS_IP}/health${NC}"
else
    echo -e "${YELLOW}⚠️  Ingress IP not yet assigned. Check with: kubectl get ingress -n ${NAMESPACE}${NC}"
fi

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
