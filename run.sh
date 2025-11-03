#!/bin/bash

# Object Detection Microservice Setup Script for Ubuntu
set -e

echo "🚀 Setting up Object Detection Microservice on Ubuntu..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/engine/install/ubuntu/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create necessary directories
echo "📁 Creating project structure..."
mkdir -p ai-service ui-service

# Make the script executable
chmod +x run.sh

echo "✅ Project structure created successfully!"

# Build and start services
echo "🐳 Building and starting Docker containers..."
docker compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
echo "🔍 Checking service status..."

# Check UI service
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ UI Service is running on http://localhost:5000"
else
    echo "❌ UI Service is not responding"
fi

# Check AI service
if curl -f http://localhost:5001/health > /dev/null 2>&1; then
    echo "✅ AI Service is running on http://localhost:5001"
else
    echo "❌ AI Service is not responding"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📊 Access your Object Detection Microservice:"
echo "   Web Interface: http://localhost:5000"
echo "   AI API: http://localhost:5001"
echo ""
echo "🛠️  Useful commands:"
echo "   docker-compose logs -f          # View logs"
echo "   docker-compose down             # Stop services"
echo "   docker-compose up -d            # Start services"
echo "   docker-compose restart          # Restart services"
echo ""
echo "📝 To test the API directly, run: python test_api.py"
