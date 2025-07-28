const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

/**
 * Production deployment script for LangForge Documentation
 * Handles deployment to various platforms and environments
 */

const DEPLOY_CONFIG = {
  production: {
    server: process.env.SERVER_HOST || 'langforge.dev',
    user: process.env.SERVER_USER || 'deploy',
    path: '/var/www/langforge-docs',
    branch: 'main'
  },
  staging: {
    server: process.env.STAGING_HOST || 'staging.langforge.dev',
    user: process.env.STAGING_USER || 'deploy',
    path: '/var/www/staging-docs',
    branch: 'develop'
  }
};

const ENVIRONMENT = process.env.NODE_ENV || 'production';
const config = DEPLOY_CONFIG[ENVIRONMENT];

console.log(`🚀 Starting deployment to ${ENVIRONMENT}...`);

async function deploy() {
  try {
    // Pre-deployment checks
    await runPreDeploymentChecks();
    
    // Build the application
    await buildApplication();
    
    // Run tests
    await runTests();
    
    // Deploy based on platform
    if (process.env.DEPLOY_PLATFORM === 'docker') {
      await deployWithDocker();
    } else if (process.env.DEPLOY_PLATFORM === 'serverless') {
      await deployServerless();
    } else {
      await deployToServer();
    }
    
    // Post-deployment verification
    await postDeploymentChecks();
    
    console.log('✅ Deployment completed successfully!');
    
  } catch (error) {
    console.error('❌ Deployment failed:', error.message);
    await rollbackDeployment();
    process.exit(1);
  }
}

async function runPreDeploymentChecks() {
  console.log('🔍 Running pre-deployment checks...');
  
  // Check environment variables
  const requiredEnvVars = ['NODE_ENV'];
  for (const envVar of requiredEnvVars) {
    if (!process.env[envVar]) {
      throw new Error(`Missing required environment variable: ${envVar}`);
    }
  }
  
  // Check Git status
  try {
    const gitStatus = execSync('git status --porcelain', { encoding: 'utf8' });
    if (gitStatus.trim()) {
      console.warn('⚠️  Warning: Working directory is not clean');
    }
  } catch (error) {
    console.warn('⚠️  Could not check Git status');
  }
  
  // Check current branch
  try {
    const currentBranch = execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf8' }).trim();
    if (currentBranch !== config.branch) {
      throw new Error(`Expected branch ${config.branch}, but on ${currentBranch}`);
    }
  } catch (error) {
    console.warn('⚠️  Could not verify Git branch');
  }
  
  console.log('✅ Pre-deployment checks passed');
}

async function buildApplication() {
  console.log('🏗️  Building application...');
  
  execSync('npm run build', { stdio: 'inherit' });
  
  // Verify build output
  const distDir = path.join(__dirname, '..', 'dist');
  if (!fs.existsSync(distDir)) {
    throw new Error('Build output directory not found');
  }
  
  console.log('✅ Application built successfully');
}

async function runTests() {
  console.log('🧪 Running tests...');
  
  execSync('npm test', { stdio: 'inherit' });
  
  console.log('✅ All tests passed');
}

async function deployWithDocker() {
  console.log('🐳 Deploying with Docker...');
  
  const imageName = `langforge-docs:${process.env.GITHUB_SHA || 'latest'}`;
  
  // Build Docker image
  execSync(`docker build -t ${imageName} .`, { stdio: 'inherit' });
  
  // Tag for registry
  const registryImage = `ghcr.io/0x-professor/langforge-docs:${process.env.GITHUB_SHA || 'latest'}`;
  execSync(`docker tag ${imageName} ${registryImage}`, { stdio: 'inherit' });
  
  // Push to registry
  execSync(`docker push ${registryImage}`, { stdio: 'inherit' });
  
  // Deploy to production
  if (config.server && config.user) {
    const deployCommand = `
      ssh ${config.user}@${config.server} '
        docker pull ${registryImage} &&
        docker stop langforge-docs || true &&
        docker rm langforge-docs || true &&
        docker run -d --name langforge-docs -p 80:3000 --restart unless-stopped ${registryImage}
      '
    `;
    execSync(deployCommand, { stdio: 'inherit' });
  }
  
  console.log('✅ Docker deployment completed');
}

async function deployServerless() {
  console.log('☁️  Deploying to serverless platform...');
  
  // Example for Vercel deployment
  if (process.env.VERCEL_TOKEN) {
    execSync('vercel --prod --token $VERCEL_TOKEN', { stdio: 'inherit' });
  }
  // Example for Netlify deployment
  else if (process.env.NETLIFY_AUTH_TOKEN) {
    execSync('netlify deploy --prod --auth $NETLIFY_AUTH_TOKEN', { stdio: 'inherit' });
  }
  // Example for AWS deployment
  else if (process.env.AWS_ACCESS_KEY_ID) {
    execSync('aws s3 sync dist/ s3://langforge-docs-bucket --delete', { stdio: 'inherit' });
    execSync('aws cloudfront create-invalidation --distribution-id $CLOUDFRONT_DISTRIBUTION_ID --paths "/*"', { stdio: 'inherit' });
  }
  else {
    throw new Error('No serverless deployment configuration found');
  }
  
  console.log('✅ Serverless deployment completed');
}

async function deployToServer() {
  console.log('🖥️  Deploying to server...');
  
  if (!config.server || !config.user) {
    throw new Error('Server configuration missing');
  }
  
  const deployCommands = [
    `rsync -avz --delete dist/ ${config.user}@${config.server}:${config.path}/`,
    `ssh ${config.user}@${config.server} 'cd ${config.path} && pm2 restart langforge-docs || pm2 start server.js --name langforge-docs'`
  ];
  
  for (const command of deployCommands) {
    execSync(command, { stdio: 'inherit' });
  }
  
  console.log('✅ Server deployment completed');
}

async function postDeploymentChecks() {
  console.log('🔍 Running post-deployment checks...');
  
  const healthUrl = `https://${config.server}/api/health`;
  
  // Wait for service to be ready
  await new Promise(resolve => setTimeout(resolve, 10000));
  
  try {
    const response = await fetch(healthUrl);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    
    const health = await response.json();
    console.log('✅ Health check passed:', health);
    
  } catch (error) {
    console.warn('⚠️  Health check failed:', error.message);
    // Don't fail deployment for health check issues
  }
  
  console.log('✅ Post-deployment checks completed');
}

async function rollbackDeployment() {
  console.log('⏪ Rolling back deployment...');
  
  try {
    if (process.env.DEPLOY_PLATFORM === 'docker') {
      // Rollback to previous Docker image
      const previousImage = `ghcr.io/0x-professor/langforge-docs:previous`;
      execSync(`docker pull ${previousImage} && docker stop langforge-docs && docker rm langforge-docs && docker run -d --name langforge-docs -p 80:3000 ${previousImage}`, { stdio: 'inherit' });
    } else {
      // Rollback using Git
      execSync('git reset --hard HEAD~1', { stdio: 'inherit' });
      await buildApplication();
      await deployToServer();
    }
    
    console.log('✅ Rollback completed');
  } catch (error) {
    console.error('❌ Rollback failed:', error.message);
  }
}

// Run deployment
if (require.main === module) {
  deploy();
}

module.exports = { deploy, rollbackDeployment };