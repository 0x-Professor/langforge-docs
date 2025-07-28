#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🚀 Starting deployment process...');

// Configuration
const config = {
  server: process.env.SERVER_HOST || 'your-server.com',
  user: process.env.SERVER_USER || 'deploy',
  deployPath: process.env.DEPLOY_PATH || '/var/www/langforge-docs',
  branch: process.env.DEPLOY_BRANCH || 'main',
  backup: true,
  healthCheck: true
};

async function deploy() {
  try {
    console.log('📋 Deployment configuration:', config);

    // Pre-deployment checks
    await runPreDeploymentChecks();

    // Build the application
    await buildApplication();

    // Create deployment package
    await createDeploymentPackage();

    // Deploy to server
    await deployToServer();

    // Post-deployment verification
    await postDeploymentChecks();

    console.log('✅ Deployment completed successfully!');
  } catch (error) {
    console.error('❌ Deployment failed:', error.message);
    process.exit(1);
  }
}

async function runPreDeploymentChecks() {
  console.log('🔍 Running pre-deployment checks...');

  // Check if we're on the right branch
  const currentBranch = execSync('git branch --show-current', { encoding: 'utf8' }).trim();
  if (currentBranch !== config.branch) {
    throw new Error(`Not on deployment branch. Current: ${currentBranch}, Expected: ${config.branch}`);
  }

  // Check for uncommitted changes
  const status = execSync('git status --porcelain', { encoding: 'utf8' });
  if (status.trim()) {
    throw new Error('Uncommitted changes found. Please commit or stash changes before deploying.');
  }

  // Check if remote is up to date
  execSync('git fetch origin');
  const behind = execSync(`git rev-list --count HEAD..origin/${config.branch}`, { encoding: 'utf8' }).trim();
  if (parseInt(behind) > 0) {
    throw new Error(`Local branch is ${behind} commits behind origin. Please pull latest changes.`);
  }

  console.log('✓ Pre-deployment checks passed');
}

async function buildApplication() {
  console.log('🔨 Building application...');

  // Run tests
  execSync('npm test', { stdio: 'inherit' });

  // Run build
  execSync('npm run build', { stdio: 'inherit' });

  // Verify build output
  if (!fs.existsSync('dist/index.html')) {
    throw new Error('Build failed: dist/index.html not found');
  }

  console.log('✓ Application built successfully');
}

async function createDeploymentPackage() {
  console.log('📦 Creating deployment package...');

  const version = require('../package.json').version;
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const packageName = `langforge-docs-${version}-${timestamp}.tar.gz`;

  // Create package
  execSync(`tar -czf ${packageName} dist/ package.json server.js`, { stdio: 'inherit' });

  console.log(`✓ Deployment package created: ${packageName}`);
  return packageName;
}

async function deployToServer() {
  console.log('🚀 Deploying to server...');

  const packageName = fs.readdirSync('.').find(f => f.match(/langforge-docs-.*\.tar\.gz$/));
  
  if (!packageName) {
    throw new Error('Deployment package not found');
  }

  // Upload package
  execSync(`scp ${packageName} ${config.user}@${config.server}:/tmp/`, { stdio: 'inherit' });

  // Deploy on server
  const deployScript = `
    set -e
    cd ${config.deployPath}
    
    # Backup current deployment
    if [ -d "current" ] && [ "${config.backup}" = "true" ]; then
      echo "Creating backup..."
      cp -r current backup-$(date +%Y%m%d-%H%M%S)
      # Keep only last 5 backups
      ls -dt backup-* | tail -n +6 | xargs rm -rf
    fi
    
    # Extract new deployment
    echo "Extracting new deployment..."
    mkdir -p releases/$(date +%Y%m%d-%H%M%S)
    cd releases/$(date +%Y%m%d-%H%M%S)
    tar -xzf /tmp/${packageName}
    
    # Install dependencies
    echo "Installing dependencies..."
    npm ci --production
    
    # Update symlink
    echo "Updating symlink..."
    cd ${config.deployPath}
    rm -f current
    ln -sf releases/$(date +%Y%m%d-%H%M%S) current
    
    # Restart services
    echo "Restarting services..."
    sudo systemctl restart langforge-docs || pm2 restart langforge-docs || echo "Service restart failed"
    
    # Cleanup
    rm -f /tmp/${packageName}
    
    echo "Deployment completed!"
  `;

  execSync(`ssh ${config.user}@${config.server} "${deployScript}"`, { stdio: 'inherit' });

  console.log('✓ Deployed to server successfully');
}

async function postDeploymentChecks() {
  if (!config.healthCheck) {
    console.log('⏭️  Skipping health checks');
    return;
  }

  console.log('🔍 Running post-deployment checks...');

  // Wait for service to start
  await new Promise(resolve => setTimeout(resolve, 10000));

  // Health check
  try {
    const response = await fetch(`https://${config.server}/api/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    const data = await response.json();
    console.log('✓ Health check passed:', data);
  } catch (error) {
    console.warn('⚠️  Health check failed:', error.message);
    console.warn('Please verify the deployment manually');
  }

  console.log('✓ Post-deployment checks completed');
}

// Utility function for fetch (Node.js < 18)
function fetch(url) {
  return new Promise((resolve, reject) => {
    const https = require('https');
    const req = https.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        resolve({
          ok: res.statusCode >= 200 && res.statusCode < 300,
          status: res.statusCode,
          json: () => Promise.resolve(JSON.parse(data))
        });
      });
    });
    req.on('error', reject);
    req.setTimeout(10000, () => reject(new Error('Request timeout')));
  });
}

// Handle script interruption
process.on('SIGINT', () => {
  console.log('\n⚠️  Deployment interrupted by user');
  process.exit(1);
});

process.on('SIGTERM', () => {
  console.log('\n⚠️  Deployment terminated');
  process.exit(1);
});

// Run deployment
if (require.main === module) {
  deploy().catch(error => {
    console.error('💥 Deployment failed:', error);
    process.exit(1);
  });
}

module.exports = { deploy };