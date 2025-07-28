const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

/**
 * Deployment script for LangForge Documentation
 * Handles deployment to various environments
 */

const ENVIRONMENTS = {
  staging: {
    name: 'Staging',
    url: 'https://staging.langforge.dev',
    branch: 'develop'
  },
  production: {
    name: 'Production',
    url: 'https://langforge.dev',
    branch: 'main'
  }
};

async function deploy() {
  const environment = process.env.DEPLOY_ENV || 'production';
  const config = ENVIRONMENTS[environment];
  
  if (!config) {
    console.error(`❌ Unknown environment: ${environment}`);
    process.exit(1);
  }

  console.log(`🚀 Deploying to ${config.name}...`);
  console.log(`📍 URL: ${config.url}`);
  console.log(`🌿 Branch: ${config.branch}\n`);

  try {
    // Pre-deployment checks
    await runPreDeploymentChecks();
    
    // Build the application
    console.log('🏗️  Building application...');
    execSync('npm run build', { stdio: 'inherit' });
    
    // Run tests
    console.log('🧪 Running tests...');
    execSync('npm test', { stdio: 'inherit' });
    
    // Security audit
    console.log('🔒 Running security audit...');
    execSync('npm audit --audit-level high', { stdio: 'inherit' });
    
    // Deploy based on environment
    if (environment === 'production') {
      await deployToProduction();
    } else {
      await deployToStaging();
    }
    
    // Post-deployment verification
    await verifyDeployment(config.url);
    
    console.log(`\n✅ Deployment to ${config.name} completed successfully!`);
    console.log(`🌐 Live at: ${config.url}`);
    
  } catch (error) {
    console.error(`❌ Deployment failed:`, error.message);
    await rollbackIfNeeded();
    process.exit(1);
  }
}

async function runPreDeploymentChecks() {
  console.log('🔍 Running pre-deployment checks...');
  
  // Check if dist directory exists
  if (!fs.existsSync('./dist')) {
    throw new Error('Build output directory not found. Run npm run build first.');
  }
  
  // Check environment variables
  const requiredEnvVars = ['NODE_ENV'];
  for (const envVar of requiredEnvVars) {
    if (!process.env[envVar]) {
      console.warn(`⚠️  Warning: ${envVar} environment variable not set`);
    }
  }
  
  // Validate package.json
  const packageJson = JSON.parse(fs.readFileSync('./package.json', 'utf8'));
  if (!packageJson.version) {
    throw new Error('Package version not found in package.json');
  }
  
  console.log(`   ✓ Version: ${packageJson.version}`);
  console.log(`   ✓ Build output verified`);
}

async function deployToProduction() {
  console.log('🎯 Deploying to production...');
  
  // Example deployment commands - customize based on your hosting provider
  
  // For GitHub Pages
  if (process.env.DEPLOY_TO_GITHUB_PAGES === 'true') {
    execSync('gh-pages -d dist', { stdio: 'inherit' });
    return;
  }
  
  // For Docker deployment
  if (process.env.DEPLOY_WITH_DOCKER === 'true') {
    console.log('🐳 Building and pushing Docker image...');
    execSync('docker build -t langforge-docs:latest .', { stdio: 'inherit' });
    execSync(`docker tag langforge-docs:latest ${process.env.DOCKER_REGISTRY}/langforge-docs:latest`, { stdio: 'inherit' });
    execSync(`docker push ${process.env.DOCKER_REGISTRY}/langforge-docs:latest`, { stdio: 'inherit' });
    return;
  }
  
  // For SSH deployment
  if (process.env.SERVER_HOST) {
    console.log('📡 Deploying via SSH...');
    const deployCommands = [
      `rsync -avz --delete ./dist/ ${process.env.SERVER_USER}@${process.env.SERVER_HOST}:/var/www/langforge-docs/`,
      `ssh ${process.env.SERVER_USER}@${process.env.SERVER_HOST} "sudo systemctl reload nginx"`
    ];
    
    for (const command of deployCommands) {
      execSync(command, { stdio: 'inherit' });
    }
    return;
  }
  
  // For Vercel deployment
  if (process.env.DEPLOY_TO_VERCEL === 'true') {
    execSync('vercel --prod', { stdio: 'inherit' });
    return;
  }
  
  // For Netlify deployment
  if (process.env.DEPLOY_TO_NETLIFY === 'true') {
    execSync('netlify deploy --prod --dir=dist', { stdio: 'inherit' });
    return;
  }
  
  console.log('ℹ️  No specific deployment method configured. Manual deployment required.');
}

async function deployToStaging() {
  console.log('🧪 Deploying to staging...');
  
  // Similar to production but with staging-specific commands
  if (process.env.STAGING_SERVER_HOST) {
    execSync(`rsync -avz --delete ./dist/ ${process.env.SERVER_USER}@${process.env.STAGING_SERVER_HOST}:/var/www/staging-langforge-docs/`, { stdio: 'inherit' });
  } else {
    console.log('ℹ️  Staging deployment method not configured.');
  }
}

async function verifyDeployment(url) {
  console.log('🔬 Verifying deployment...');
  
  // Wait a moment for deployment to propagate
  await new Promise(resolve => setTimeout(resolve, 5000));
  
  try {
    // Check if the site is accessible
    const https = require('https');
    const http = require('http');
    const client = url.startsWith('https') ? https : http;
    
    const response = await new Promise((resolve, reject) => {
      const req = client.get(url, resolve);
      req.on('error', reject);
      req.setTimeout(10000, () => reject(new Error('Request timeout')));
    });
    
    if (response.statusCode === 200) {
      console.log('   ✓ Site is accessible');
    } else {
      throw new Error(`Site returned status code: ${response.statusCode}`);
    }
    
    // Check health endpoint if available
    try {
      const healthUrl = `${url}/api/health`;
      const healthResponse = await new Promise((resolve, reject) => {
        const req = client.get(healthUrl, resolve);
        req.on('error', reject);
        req.setTimeout(5000, () => reject(new Error('Health check timeout')));
      });
      
      if (healthResponse.statusCode === 200) {
        console.log('   ✓ Health check passed');
      }
    } catch (error) {
      console.log('   ⚠️  Health check not available (this is normal for static sites)');
    }
    
  } catch (error) {
    throw new Error(`Deployment verification failed: ${error.message}`);
  }
}

async function rollbackIfNeeded() {
  console.log('🔄 Initiating rollback procedures...');
  
  // Implement rollback logic based on your deployment method
  if (process.env.ENABLE_AUTO_ROLLBACK === 'true') {
    console.log('   Automatic rollback not implemented yet');
    console.log('   Please manually rollback if necessary');
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n⚠️  Deployment interrupted. Cleaning up...');
  process.exit(1);
});

process.on('SIGTERM', () => {
  console.log('\n⚠️  Deployment terminated. Cleaning up...');
  process.exit(1);
});

// Run deployment
if (require.main === module) {
  deploy().catch(error => {
    console.error('❌ Deployment script error:', error);
    process.exit(1);
  });
}

module.exports = { deploy };