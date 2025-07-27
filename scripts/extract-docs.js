import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuration
const SOURCE_DIR = path.join(__dirname, '..', 'src');
const OUTPUT_DIR = path.join(__dirname, '..', 'docs');
const IGNORE_DIRS = ['node_modules', '.next', '.git', 'dist'];

// Ensure output directory exists
async function ensureOutputDir() {
  if (!(await fs.stat(OUTPUT_DIR).catch(() => null))) {
    await fs.mkdir(OUTPUT_DIR, { recursive: true });
  }
}

// Function to extract text from JSX/TSX content
function extractTextFromJSX(content) {
  // Remove imports and exports
  let text = content
    .replace(/import\s+.*?\s+from\s+['"].*?['"];?/g, '')
    .replace(/export\s+.*?\s+from\s+['"].*?['"];?/g, '')
    .replace(/export\s+default\s+.*?;?/g, '');

  // Extract text from JSX
  const textMatches = text.match(/<[^>]*>([^<]*)<\/[^>]*>/g) || [];
  textMatches.forEach(match => {
    const textContent = match.replace(/<[^>]*>/g, '').trim();
    if (textContent) {
      text = text.replace(match, `\n${textContent}\n`);
    }
  });

  // Remove remaining JSX tags
  text = text.replace(/<[^>]*>/g, '');
  
  // Clean up multiple newlines
  text = text.replace(/\n{3,}/g, '\n\n');
  
  return text.trim();
}

// Function to process a file
function processFile(filePath, relativePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    let markdown = '';
    
    if (filePath.endsWith('.md')) {
      // For markdown files, just copy them
      markdown = content;
    } else if (filePath.endsWith('.tsx') || filePath.endsWith('.jsx')) {
      // For React components, extract text
      const componentName = path.basename(filePath, path.extname(filePath));
      markdown = `# ${componentName}\n\n${extractTextFromJSX(content)}`;
    }
    
    // Save the markdown file
    if (markdown) {
      const outputPath = path.join(OUTPUT_DIR, relativePath).replace(/\.(tsx|jsx)$/, '.md');
      const outputDir = path.dirname(outputPath);
      
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }
      
      fs.writeFileSync(outputPath, markdown);
      console.log(`Processed: ${filePath} -> ${outputPath}`);
    }
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
  }
}

// Function to walk directory
function walkDir(dir, callback) {
  const files = fs.readdirSync(dir);
  
  files.forEach(file => {
    if (IGNORE_DIRS.includes(file)) return;
    
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    
    if (stat.isDirectory()) {
      walkDir(filePath, callback);
    } else if (file.match(/\.(tsx|jsx|md)$/)) {
      const relativePath = path.relative(SOURCE_DIR, filePath);
      callback(filePath, relativePath);
    }
  });
}

// Main function
function main() {
  console.log('Starting documentation extraction...');
  
  // Process all relevant files
  walkDir(SOURCE_DIR, (filePath, relativePath) => {
    processFile(filePath, relativePath);
  });
  
  console.log('\nDocumentation extraction complete!');
  console.log(`Output directory: ${path.resolve(OUTPUT_DIR)}`);
}

// Run the main function
main();
