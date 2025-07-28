#!/bin/bash

# Build documentation script
# This script builds the documentation site for production deployment

set -e  # Exit on any error

echo "🏗️  Building LangForge Documentation..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed. Please install Node.js 16+ first.${NC}"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2)
REQUIRED_VERSION="16.0.0"

if ! node -e "process.exit(require('semver').gte('$NODE_VERSION', '$REQUIRED_VERSION') ? 0 : 1)" 2>/dev/null; then
    echo -e "${RED}❌ Node.js version $NODE_VERSION is not supported. Please use Node.js 16+ ${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Node.js version $NODE_VERSION detected${NC}"

# Install dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
npm ci --production=false

# Run linting
echo -e "${YELLOW}🔍 Running linting...${NC}"
npm run lint

# Run tests
echo -e "${YELLOW}🧪 Running tests...${NC}"
npm test

# Build documentation
echo -e "${YELLOW}📚 Building documentation...${NC}"
npm run docs:build

# Build server assets
echo -e "${YELLOW}⚡ Building server assets...${NC}"
npm run build:assets

# Generate sitemap
echo -e "${YELLOW}🗺️  Generating sitemap...${NC}"
node scripts/generate-sitemap.js

# Optimize images
echo -e "${YELLOW}🖼️  Optimizing images...${NC}"
if command -v imagemin &> /dev/null; then
    find docs -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" | xargs imagemin --out-dir=dist/images/
else
    echo -e "${YELLOW}⚠️  imagemin not found, skipping image optimization${NC}"
fi

# Create build info
echo -e "${YELLOW}📋 Creating build info...${NC}"
cat > dist/build-info.json << EOF
{
  "buildTime": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "gitCommit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "gitBranch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
  "nodeVersion": "$NODE_VERSION",
  "version": "$(npm pkg get version | tr -d '"')"
}
EOF

# Validate build
echo -e "${YELLOW}✅ Validating build...${NC}"
if [ ! -d "dist" ]; then
    echo -e "${RED}❌ Build failed: dist directory not found${NC}"
    exit 1
fi

if [ ! -f "dist/index.html" ]; then
    echo -e "${RED}❌ Build failed: index.html not found${NC}"
    exit 1
fi

# Calculate build size
BUILD_SIZE=$(du -sh dist | cut -f1)
echo -e "${GREEN}📊 Build size: $BUILD_SIZE${NC}"

echo -e "${GREEN}🎉 Build completed successfully!${NC}"
echo -e "${GREEN}📂 Output directory: ./dist${NC}"
echo -e "${GREEN}🚀 Ready for deployment!${NC}"

# Show deployment commands
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "  • Test locally: npm run serve"
echo "  • Deploy to production: npm run deploy"
echo "  • Deploy with Docker: docker build -t langforge-docs ."