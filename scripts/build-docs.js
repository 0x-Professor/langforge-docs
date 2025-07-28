const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

/**
 * LangForge Documentation Build Script
 * This script builds the documentation for production deployment
 */

console.log('🚀 Starting LangForge Documentation build process...');

// Utility function to execute commands
function execCommand(command, description) {
  console.log(`📋 ${description}...`);
  try {
    execSync(command, { stdio: 'inherit' });
    console.log(`✅ ${description} completed successfully`);
  } catch (error) {
    console.error(`❌ ${description} failed:`, error.message);
    process.exit(1);
  }
}

// Utility function to ensure directory exists
function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
    console.log(`📁 Created directory: ${dirPath}`);
  }
}

// Main build process
async function build() {
  try {
    // Create necessary directories
    console.log('📁 Creating build directories...');
    ensureDir('dist');
    ensureDir('logs');
    ensureDir('data');
    ensureDir('data/analytics');

    // Check if node_modules exists
    if (!fs.existsSync('node_modules')) {
      execCommand('npm ci', 'Installing dependencies');
    }

    // Build VitePress documentation
    console.log('📚 Building VitePress documentation...');
    if (fs.existsSync('.vitepress') || fs.existsSync('docs/.vitepress')) {
      execCommand('npm run docs:build', 'Building VitePress docs');
    } else {
      console.log('⚠️  VitePress not configured, creating static build...');
      
      // Copy docs to dist
      if (fs.existsSync('docs')) {
        execCommand('cp -r docs dist/ || xcopy docs dist\\ /e /i /y', 'Copying documentation files');
      }
    }

    // Copy static files
    console.log('📋 Copying static files...');
    const staticFiles = ['robots.txt', 'README.md'];
    
    staticFiles.forEach(file => {
      if (fs.existsSync(file)) {
        fs.copyFileSync(file, path.join('dist', file));
        console.log(`📄 Copied ${file}`);
      }
    });

    // Generate version file
    console.log('📋 Creating version info...');
    const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    
    let gitCommit = 'unknown';
    try {
      gitCommit = execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
    } catch (error) {
      console.log('⚠️  Git not available, using unknown commit');
    }

    const versionInfo = {
      version: packageJson.version,
      buildDate: new Date().toISOString(),
      commit: gitCommit,
      environment: process.env.NODE_ENV || 'production'
    };

    fs.writeFileSync('dist/version.json', JSON.stringify(versionInfo, null, 2));
    console.log('📄 Created version.json');

    // Create a simple index.html if it doesn't exist
    const indexPath = 'dist/index.html';
    if (!fs.existsSync(indexPath)) {
      console.log('📄 Creating index.html...');
      const indexHtml = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LangForge Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .nav { background: #f5f5f5; padding: 20px; border-radius: 8px; }
        .nav ul { list-style: none; padding: 0; }
        .nav li { margin: 10px 0; }
        .nav a { text-decoration: none; color: #333; }
        .nav a:hover { color: #007bff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 LangForge Documentation</h1>
            <p>The Complete Guide to Building Production-Ready LLM Applications</p>
        </div>
        <div class="nav">
            <h2>Documentation</h2>
            <ul>
                <li><a href="/docs/getting-started/">Getting Started</a></li>
                <li><a href="/docs/langchain.html">LangChain Guide</a></li>
                <li><a href="/docs/langsmith.html">LangSmith Guide</a></li>
                <li><a href="/docs/langgraph.html">LangGraph Guide</a></li>
                <li><a href="/docs/langserve.html">LangServe Guide</a></li>
                <li><a href="/docs/examples/">Examples</a></li>
                <li><a href="/docs/guides/">Guides</a></li>
            </ul>
            <h2>API</h2>
            <ul>
                <li><a href="/api/health">Health Check</a></li>
                <li><a href="/api/search?q=langchain">Search API</a></li>
            </ul>
        </div>
    </div>
</body>
</html>`;
      
      fs.writeFileSync(indexPath, indexHtml);
    }

    // Run security audit (non-blocking)
    console.log('🔒 Running security audit...');
    try {
      execSync('npm audit --audit-level moderate', { stdio: 'inherit' });
    } catch (error) {
      console.log('⚠️  Security audit found issues, but continuing build...');
    }

    // Build statistics
    console.log('✅ Build completed successfully!');
    console.log('📊 Build statistics:');
    
    try {
      const stats = execSync('find dist -type f | wc -l || dir dist /s /-c | find "File(s)"', { encoding: 'utf8' });
      console.log(`   - Files built: ${stats.trim()}`);
    } catch (error) {
      console.log('   - Build statistics unavailable');
    }

    console.log('');
    console.log('🎉 Your LangForge Documentation is ready!');
    console.log('📁 Built files are in the "dist" directory');
    console.log('🚀 To start the server: npm start');

  } catch (error) {
    console.error('❌ Build failed:', error.message);
    process.exit(1);
  }
}

// Run the build
build();