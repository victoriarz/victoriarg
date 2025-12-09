#!/bin/bash

# Start script for Saponify AI backend server
# This script starts the backend proxy server for local development

echo "🚀 Starting Saponify AI backend server..."
echo ""

# Check if .env file exists
if [ ! -f "server/.env" ]; then
    echo "❌ Error: server/.env file not found!"
    echo "Please create server/.env from server/.env.example and add your API key"
    exit 1
fi

# Navigate to server directory and start
cd server
echo "📦 Installing dependencies (if needed)..."
npm install --quiet

echo ""
echo "🎯 Starting server on port 3000..."
npm start
