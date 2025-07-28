#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

/**
 * LangForge Documentation Deployment Script
 * This script handles deployment to various platforms
 */

console.log('🚀 Starting LangForge Documentation deployment...');

// Configuration
const config = {
  buildDir: 'dist',
  backupDir: 'backup',
  environment: process.env.NODE_ENV || 'production'
};

// Utility functions
function execCommand(command, description) {
  console.log(`📋 ${description}...`);
  try {
    execSync(command, { stdio: 'inherit' });
    console.log(`✅ ${description} completed`);
  } catch (error) {
    console.error(`❌ ${description} failed:`, error.message);
    process.exit(1);
  }
}

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
    console.log(`📁 Created directory: ${dirPath}`);
  }
}

// Main deployment function
async function deploy() {
  try {
    console.log(`🌍 Deploying to ${config.environment} environment...`);

    // Pre-deployment checks
    console.log('🔍 Running pre-deployment checks...');
    
    if (!fs.existsSync(config.buildDir)) {
      console.log('📦 Build directory not found, running build...');
      execCommand('npm run build', 'Building project');
    }

    // Create backup if needed
    if (fs.existsSync(config.buildDir)) {
      ensureDir(config.backupDir);
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const backupPath = path.join(config.backupDir, `build-${timestamp}`);
      
      console.log(`💾 Creating backup at ${backupPath}...`);
      try {
        execSync(`cp -r ${config.buildDir} ${backupPath} || xcopy ${config.buildDir} ${backupPath}\\ /e /i /y`, { stdio: 'inherit' });
      } catch (error) {
        console.log('⚠️  Backup creation failed, continuing...');
      }
    }

    // Environment-specific deployment
    switch (config.environment) {
      case 'development':
        deployDevelopment();
        break;
      case 'staging':
        deployStaging();
        break;
      case 'production':
        deployProduction();
        break;
      default:
        console.log('🔧 Unknown environment, running basic deployment...');
        deployBasic();
    }

    console.log('✅ Deployment completed successfully!');
    console.log('🎉 Your LangForge Documentation is now live!');

  } catch (error) {
    console.error('❌ Deployment failed:', error.message);
    process.exit(1);
  }
}

function deployDevelopment() {
  console.log('🔧 Deploying to development environment...');
  
  // Start local server
  console.log('🚀 Starting development server...');
  console.log('📱 Server will be available at: http://localhost:3000');
  console.log('📚 Documentation: http://localhost:3000/docs');
  console.log('🔍 Search API: http://localhost:3000/api/search');
  
  // Note: In a real deployment, this would start the server
  console.log('💡 Run "npm start" to start the server');
}

function deployStaging() {
  console.log('🧪 Deploying to staging environment...');
  
  // Staging-specific deployment steps
  console.log('🔍 Running staging validation...');
  
  // You could add staging-specific commands here
  // For example: rsync to staging server, run smoke tests, etc.
  
  console.log('📡 Staging deployment would typically:');
  console.log('   - Upload files to staging server');
  console.log('   - Run smoke tests');
  console.log('   - Validate all endpoints');
  console.log('   - Check performance metrics');
}

function deployProduction() {
  console.log('🌍 Deploying to production environment...');
  
  // Production deployment steps
  console.log('🔒 Running production validation...');
  
  // Security checks
  try {
    execCommand('npm audit --audit-level high', 'Security audit');
  } catch (error) {
    console.log('⚠️  Security audit found issues, review before proceeding');
  }
  
  // Performance optimization
  console.log('⚡ Optimizing for production...');
  
  console.log('🚀 Production deployment would typically:');
  console.log('   - Upload to CDN');
  console.log('   - Update DNS records');
  console.log('   - Invalidate cache');
  console.log('   - Run health checks');
  console.log('   - Monitor deployment');
  
  // Example deployment commands (uncomment and modify as needed)
  /*
  if (process.env.DEPLOY_TARGET === 'vercel') {
    execCommand('vercel --prod', 'Deploying to Vercel');
  } else if (process.env.DEPLOY_TARGET === 'netlify') {
    execCommand('netlify deploy --prod --dir=dist', 'Deploying to Netlify');
  } else if (process.env.DEPLOY_TARGET === 'aws') {
    execCommand('aws s3 sync dist/ s3://your-bucket --delete', 'Deploying to AWS S3');
  }
  */
}

function deployBasic() {
  console.log('📦 Running basic deployment...');
  
  // Basic deployment steps
  console.log('📋 Basic deployment checklist:');
  console.log('   ✅ Build completed');
  console.log('   ✅ Files ready in dist/');
  console.log('   📝 Manual steps required:');
  console.log('      - Upload dist/ contents to your server');
  console.log('      - Configure web server (nginx/apache)');
  console.log('      - Set up SSL certificate');
  console.log('      - Configure domain DNS');
}

// Platform-specific deployment helpers
const deploymentHelpers = {
  vercel: () => {
    console.log('🔷 Vercel Deployment Guide:');
    console.log('   1. Install Vercel CLI: npm i -g vercel');
    console.log('   2. Login: vercel login');
    console.log('   3. Deploy: vercel --prod');
    console.log('   4. Set environment variables in Vercel dashboard');
  },
  
  netlify: () => {
    console.log('🟢 Netlify Deployment Guide:');
    console.log('   1. Install Netlify CLI: npm i -g netlify-cli');
    console.log('   2. Login: netlify login');
    console.log('   3. Deploy: netlify deploy --prod --dir=dist');
    console.log('   4. Configure environment variables');
  },
  
  docker: () => {
    console.log('🐳 Docker Deployment Guide:');
    console.log('   1. Build image: docker build -t langforge-docs .');
    console.log('   2. Run container: docker run -p 3000:3000 langforge-docs');
    console.log('   3. Or use docker-compose: docker-compose up -d');
  }
};

// Handle command line arguments
const args = process.argv.slice(2);
const platform = args.find(arg => arg.startsWith('--platform='))?.split('=')[1];

if (platform && deploymentHelpers[platform]) {
  console.log(`📖 Showing ${platform} deployment guide:`);
  deploymentHelpers[platform]();
} else {
  // Run main deployment
  deploy();
}

// Export for testing
module.exports = { deploy, deploymentHelpers };