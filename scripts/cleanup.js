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

// List of file extensions to keep (markdown files)
const KEEP_EXTENSIONS = ['.md', '.markdown'];

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
  
  // List of directories to keep
  const keepDirs = ['docs', '.git'];
  
  // List of files to keep
  const keepFiles = ['README.md', 'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md', 'LICENSE'];
  
  // Get all files and directories in the project root
  const items = fs.readdirSync(PROJECT_ROOT, { withFileTypes: true });
  
  items.forEach(item => {
    const itemPath = path.join(PROJECT_ROOT, item.name);
    
    // Skip items we want to keep
    if (keepDirs.includes(item.name) || 
        (item.isFile() && keepFiles.includes(item.name)) ||
        (item.isFile() && KEEP_EXTENSIONS.includes(path.extname(item.name).toLowerCase()))) {
      console.log(`Keeping: ${itemPath}`);
      return;
    }
    
    // Remove the item
    try {
      if (item.isDirectory()) {
        console.log(`Removing directory: ${itemPath}`);
        fs.rmSync(itemPath, { recursive: true, force: true });
      } else {
        console.log(`Removing file: ${itemPath}`);
        fs.unlinkSync(itemPath);
      }
    } catch (error) {
      console.error(`Error removing ${itemPath}:`, error.message);
    }
  });
  
  // Clean up any remaining files in the root directory
  const removeFiles = [
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
  
  removeFiles.forEach(file => {
    const filePath = path.join(PROJECT_ROOT, file);
    if (fs.existsSync(filePath)) {
      try {
        console.log(`Removing file: ${filePath}`);
        fs.unlinkSync(filePath);
      } catch (error) {
        console.error(`Error removing ${filePath}:`, error.message);
      }
    }
  });

  console.log('\nCleanup complete!');
  console.log('Only markdown documentation remains in the project.');
}

// Run the cleanup
cleanProject();
