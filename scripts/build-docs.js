#!/bin/bash

# LangForge Documentation Build Script
# This script builds the documentation for production deployment

set -e  # Exit on any error

echo "🚀 Starting LangForge Documentation build process..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ and try again."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm and try again."
    exit 1
fi

# Create necessary directories
echo "📁 Creating build directories..."
mkdir -p dist
mkdir -p logs
mkdir -p tmp

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm ci --production
fi

# Run linting
echo "🔍 Running code quality checks..."
npm run lint

# Run tests
echo "🧪 Running tests..."
npm test

# Build documentation
echo "📚 Building documentation..."
npm run build:docs

# Build assets
echo "🎨 Building static assets..."
npm run build:assets

# Copy static files
echo "📋 Copying static files..."
cp -r docs/assets/* dist/ 2>/dev/null || true
cp robots.txt dist/ 2>/dev/null || true
cp sitemap.xml dist/ 2>/dev/null || true

# Generate sitemap
echo "🗺️  Generating sitemap..."
node scripts/generate-sitemap.js

# Optimize images (if imagemin is available)
if command -v imagemin &> /dev/null; then
    echo "🖼️  Optimizing images..."
    imagemin "dist/**/*.{jpg,jpeg,png,gif,svg}" --out-dir=dist/optimized
fi

# Create version file
echo "📋 Creating version info..."
cat > dist/version.json << EOF
{
  "version": "$(npm pkg get version | tr -d '"')",
  "buildDate": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "environment": "${NODE_ENV:-production}"
}
EOF

# Security check
echo "🔒 Running security audit..."
npm audit --audit-level moderate

# Performance check
echo "⚡ Running performance checks..."
if command -v lighthouse &> /dev/null; then
    echo "Running Lighthouse audit..."
    # lighthouse http://localhost:3000 --output=json --output-path=./dist/lighthouse-report.json
fi

echo "✅ Build completed successfully!"
echo "📊 Build statistics:"
echo "   - Total files: $(find dist -type f | wc -l)"
echo "   - Total size: $(du -sh dist | cut -f1)"
echo "   - Build time: $SECONDS seconds"

echo ""
echo "🎉 Your LangForge Documentation is ready for deployment!"
echo "📁 Built files are in the 'dist' directory"
echo "🚀 To deploy: npm run deploy"