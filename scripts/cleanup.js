import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const PROJECT_ROOT = path.join(__dirname, '..');
const PRESERVE_DIRS = ['docs', '.git'];
const PRESERVE_FILES = ['README.md', 'LICENSE', 'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md'];

// Function to safely remove a directory
function removeDirectory(dirPath) {
  if (fs.existsSync(dirPath)) {
    console.log(`Removing directory: ${dirPath}`);
    fs.rmSync(dirPath, { recursive: true, force: true });
  }
}

// Function to safely remove a file
function removeFile(filePath) {
  if (fs.existsSync(filePath)) {
    console.log(`Removing file: ${filePath}`);
    fs.unlinkSync(filePath);
  }
}

// Function to clean the project
function cleanProject() {
  console.log('Starting project cleanup...');
  
  // List of directories and files to remove
  const removeDirs = [
    'node_modules',
    'public',
    'src',
    'scripts',
    '.next',
    'dist',
    '.vscode',
    '.github',
    '.gitignore',
    'bun.lockb',
    'components.json',
    'eslint.config.js',
    'index.html',
    'package-lock.json',
    'package.json',
    'postcss.config.js',
    'tailwind.config.ts',
    'tsconfig.app.json',
    'tsconfig.json',
    'tsconfig.node.json',
    'vite.config.ts'
  ];

  // Remove directories
  removeDirs.forEach(dir => {
    const fullPath = path.join(PROJECT_ROOT, dir);
    if (fs.existsSync(fullPath)) {
      if (fs.lstatSync(fullPath).isDirectory()) {
        removeDirectory(fullPath);
      } else {
        removeFile(fullPath);
      }
    }
  });

  console.log('\nCleanup complete!');
  console.log('Only markdown documentation remains in the /docs directory.');
}

// Run the cleanup
cleanProject();
