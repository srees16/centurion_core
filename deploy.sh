#!/bin/bash

# Build and Run Script for AI Trading System

echo "=================================================="
echo "AI Trading Alert System - Docker Deployment"
echo "=================================================="

# Build the Docker image
echo "Building Docker image..."
docker build -t algo-trading-system:latest .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully!"
    
    # Stop and remove existing container if running
    echo "Stopping existing container (if any)..."
    docker stop algo-trading-system 2>/dev/null
    docker rm algo-trading-system 2>/dev/null
    
    # Run the container
    echo "Starting container..."
    docker run -d \
        --name algo-trading-system \
        -p 8501:8501 \
        -v $(pwd)/data:/app/data \
        algo-trading-system:latest
    
    if [ $? -eq 0 ]; then
        echo "✓ Container started successfully!"
        echo ""
        echo "=================================================="
        echo "Application is running at: http://localhost:8501"
        echo "=================================================="
        echo ""
        echo "Useful commands:"
        echo "  View logs:     docker logs -f algo-trading-system"
        echo "  Stop:          docker stop algo-trading-system"
        echo "  Restart:       docker restart algo-trading-system"
        echo "  Remove:        docker rm -f algo-trading-system"
    else
        echo "✗ Failed to start container"
        exit 1
    fi
else
    echo "✗ Failed to build Docker image"
    exit 1
fi
