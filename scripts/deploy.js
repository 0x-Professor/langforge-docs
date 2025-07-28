#!/usr/bin/env node

/**
 * Production deployment script for LangForge Documentation
 * Handles deployment to various environments with safety checks
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
require('dotenv').config();

// Configuration
const config = {
  production: {
    host: process.env.PRODUCTION_HOST || 'langforge.dev',
    user: process.env.PRODUCTION_USER || 'deploy',
    path: '/var/www/langforge-docs',
    branch: 'main'
  },
  staging: {
    host: process.env.STAGING_HOST || 'staging.langforge.dev',
    user: process.env.STAGING_USER || 'deploy',
    path: '/var/www/staging-langforge-docs',
    branch: 'develop'
  }
};

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function error(message) {
  log(`❌ ${message}`, 'red');
  process.exit(1);
}

function success(message) {
  log(`✅ ${message}`, 'green');
}

function info(message) {
  log(`ℹ️  ${message}`, 'blue');
}

function warning(message) {
  log(`⚠️  ${message}`, 'yellow');
}

// Deployment functions
function preDeploymentChecks() {
  info('Running pre-deployment checks...');
  
  // Check if we're on the correct branch
  const currentBranch = execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf8' }).trim();
  const targetBranch = process.argv[2] === 'staging' ? 'develop' : 'main';
  
  if (currentBranch !== targetBranch) {
    error(`Wrong branch! Expected '${targetBranch}', got '${currentBranch}'`);
  }
  
  // Check if working directory is clean
  try {
    execSync('git diff --exit-code', { stdio: 'ignore' });
    execSync('git diff --cached --exit-code', { stdio: 'ignore' });
  } catch (e) {
    error('Working directory is not clean. Commit or stash changes first.');
  }
  
  // Check if we're up to date with remote
  try {
    execSync(`git fetch origin ${targetBranch}`, { stdio: 'ignore' });
    const localCommit = execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
    const remoteCommit = execSync(`git rev-parse origin/${targetBranch}`, { encoding: 'utf8' }).trim();
    
    if (localCommit !== remoteCommit) {
      error('Local branch is not up to date with remote. Pull latest changes first.');
    }
  } catch (e) {
    warning('Could not check remote status. Proceeding anyway...');
  }
  
  success('Pre-deployment checks passed');
}

function buildApplication() {
  info('Building application...');
  
  try {
    execSync('npm run build', { stdio: 'inherit' });
    success('Build completed successfully');
  } catch (e) {
    error('Build failed');
  }
}

function runTests() {
  info('Running tests...');
  
  try {
    execSync('npm test', { stdio: 'inherit' });
    success('All tests passed');
  } catch (e) {
    error('Tests failed');
  }
}

function createBackup(environment) {
  const env = config[environment];
  info(`Creating backup on ${env.host}...`);
  
  try {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupPath = `${env.path}-backup-${timestamp}`;
    
    execSync(`ssh ${env.user}@${env.host} "cp -r ${env.path} ${backupPath}"`, { stdio: 'inherit' });
    success(`Backup created at ${backupPath}`);
    
    return backupPath;
  } catch (e) {
    error('Failed to create backup');
  }
}

function deployToServer(environment) {
  const env = config[environment];
  info(`Deploying to ${environment} environment (${env.host})...`);
  
  try {
    // Upload files
    execSync(`rsync -avz --delete dist/ ${env.user}@${env.host}:${env.path}/`, { stdio: 'inherit' });
    
    // Restart services
    execSync(`ssh ${env.user}@${env.host} "systemctl restart langforge-docs"`, { stdio: 'inherit' });
    
    success(`Deployment to ${environment} completed`);
  } catch (e) {
    error(`Deployment to ${environment} failed`);
  }
}

function runHealthCheck(environment) {
  const env = config[environment];
  info(`Running health check on ${env.host}...`);
  
  try {
    const healthUrl = `https://${env.host}/api/health`;
    const response = execSync(`curl -f ${healthUrl}`, { encoding: 'utf8' });
    const health = JSON.parse(response);
    
    if (health.status === 'ok') {
      success('Health check passed');
      return true;
    } else {
      error('Health check failed: Invalid response');
    }
  } catch (e) {
    error('Health check failed: Could not reach server');
  }
}

function rollback(environment, backupPath) {
  const env = config[environment];
  warning(`Rolling back deployment on ${env.host}...`);
  
  try {
    execSync(`ssh ${env.user}@${env.host} "rm -rf ${env.path} && mv ${backupPath} ${env.path}"`, { stdio: 'inherit' });
    execSync(`ssh ${env.user}@${env.host} "systemctl restart langforge-docs"`, { stdio: 'inherit' });
    
    success('Rollback completed');
  } catch (e) {
    error('Rollback failed');
  }
}

function notifySlack(message) {
  const webhookUrl = process.env.SLACK_WEBHOOK_URL;
  if (!webhookUrl) return;
  
  try {
    const payload = JSON.stringify({
      text: `🚀 LangForge Docs Deployment: ${message}`,
      channel: '#deployments',
      username: 'Deploy Bot'
    });
    
    execSync(`curl -X POST -H 'Content-type: application/json' --data '${payload}' ${webhookUrl}`, { stdio: 'ignore' });
  } catch (e) {
    warning('Failed to send Slack notification');
  }
}

// Main deployment flow
async function deploy() {
  const environment = process.argv[2] || 'production';
  
  if (!config[environment]) {
    error(`Unknown environment: ${environment}`);
  }
  
  log(`🚀 Starting deployment to ${environment}...`, 'magenta');
  
  try {
    // Pre-deployment
    preDeploymentChecks();
    runTests();
    buildApplication();
    
    // Create backup
    const backupPath = createBackup(environment);
    
    // Deploy
    deployToServer(environment);
    
    // Post-deployment
    setTimeout(() => {
      if (runHealthCheck(environment)) {
        success(`🎉 Deployment to ${environment} completed successfully!`);
        notifySlack(`Deployment to ${environment} successful`);
      } else {
        rollback(environment, backupPath);
        notifySlack(`Deployment to ${environment} failed and was rolled back`);
      }
    }, 5000); // Wait 5 seconds for service to start
    
  } catch (e) {
    error(`Deployment failed: ${e.message}`);
  }
}

// Handle command line arguments
if (require.main === module) {
  if (process.argv.includes('--help') || process.argv.includes('-h')) {
    console.log(`
Usage: node scripts/deploy.js [environment]

Environments:
  production  Deploy to production server (default)
  staging     Deploy to staging server

Options:
  --help, -h  Show this help message

Examples:
  node scripts/deploy.js production
  node scripts/deploy.js staging
    `);
    process.exit(0);
  }
  
  deploy();
}