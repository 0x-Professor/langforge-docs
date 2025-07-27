import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const DOCS_DIR = path.join(__dirname, '..', 'docs');
const NEW_STRUCTURE = {
  'getting-started': [
    'introduction',
    'installation',
    'quickstart',
    'tutorial',
    'overview'
  ],
  'guides': [
    'concepts',
    'tutorials',
    'examples',
    'best-practices',
    'troubleshooting'
  ],
  'api': [
    'reference',
    'classes',
    'interfaces',
    'functions',
    'types'
  ],
  'examples': [
    'basic-usage',
    'advanced-usage',
    'integrations',
    'templates'
  ]
};

// Create the new directory structure
function createNewStructure() {
  console.log('Creating new documentation structure...');
  
  // Create main category directories
  Object.keys(NEW_STRUCTURE).forEach(category => {
    const categoryPath = path.join(DOCS_DIR, category);
    if (!fs.existsSync(categoryPath)) {
      fs.mkdirSync(categoryPath, { recursive: true });
      console.log(`Created directory: ${categoryPath}`);
    }
    
    // Create subdirectories for each category
    NEW_STRUCTURE[category].forEach(subdir => {
      const subdirPath = path.join(categoryPath, subdir);
      if (!fs.existsSync(subdirPath)) {
        fs.mkdirSync(subdirPath, { recursive: true });
        console.log(`Created directory: ${subdirPath}`);
      }
    });
  });
  
  console.log('New documentation structure created!');
}

// Function to move files to the new structure
function organizeFiles() {
  console.log('\nOrganizing documentation files...');
  
  // Get all markdown files in the docs directory
  const files = [];
  
  function walkDir(dir) {
    const items = fs.readdirSync(dir, { withFileTypes: true });
    
    items.forEach(item => {
      const fullPath = path.join(dir, item.name);
      
      if (item.isDirectory() && !['guides', 'api', 'examples', 'getting-started'].includes(item.name)) {
        walkDir(fullPath);
      } else if (item.isFile() && item.name.endsWith('.md')) {
        files.push(fullPath);
      }
    });
  }
  
  // Start walking from the docs directory
  walkDir(DOCS_DIR);
  
  console.log(`Found ${files.length} markdown files to organize.`);
  
  // Move files to appropriate directories based on their content
  files.forEach(file => {
    const content = fs.readFileSync(file, 'utf8').toLowerCase();
    const fileName = path.basename(file);
    let targetDir = '';
    
    // Determine the target directory based on file content and name
    if (fileName.toLowerCase().includes('introduction') || 
        fileName.toLowerCase().includes('getting-started') ||
        fileName.toLowerCase().includes('installation') ||
        fileName.toLowerCase().includes('quickstart')) {
      targetDir = path.join(DOCS_DIR, 'getting-started');
    } else if (fileName.toLowerCase().includes('example') || 
               content.includes('example') || 
               content.includes('usage')) {
      targetDir = path.join(DOCS_DIR, 'examples');
    } else if (fileName.toLowerCase().includes('api') || 
               content.includes('api') || 
               content.includes('reference') ||
               content.includes('interface') ||
               content.includes('class') ||
               content.includes('function')) {
      targetDir = path.join(DOCS_DIR, 'api');
    } else {
      targetDir = path.join(DOCS_DIR, 'guides');
    }
    
    // Create the target directory if it doesn't exist
    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true });
    }
    
    // Move the file
    const targetPath = path.join(targetDir, fileName);
    
    try {
      fs.renameSync(file, targetPath);
      console.log(`Moved: ${file} -> ${targetPath}`);
    } catch (error) {
      console.error(`Error moving ${file}:`, error.message);
    }
  });
  
  console.log('\nDocumentation files have been organized!');
}

// Main function
function main() {
  try {
    createNewStructure();
    organizeFiles();
    
    console.log('\nDocumentation organization complete!');
    console.log('The documentation has been reorganized into the following structure:');
    console.log('- getting-started/  # Introduction, installation, and quickstart guides');
    console.log('- guides/           # Concept guides and tutorials');
    console.log('- api/              # API reference and technical documentation');
    console.log('- examples/         # Code examples and templates');
  } catch (error) {
    console.error('Error organizing documentation:', error);
    process.exit(1);
  }
}

// Run the main function
main();
