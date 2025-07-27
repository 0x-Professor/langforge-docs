import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const SOURCE_DIR = path.join(__dirname, '..', 'src');
const OUTPUT_DIR = path.join(__dirname, '..', 'docs');
const IGNORE_DIRS = ['node_modules', '.next', '.git', 'dist', '__tests__', 'test', 'tests'];

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Function to create a markdown file from a React component
function createMarkdownFromComponent(componentPath, outputPath) {
  try {
    const content = fs.readFileSync(componentPath, 'utf8');
    
    // Extract component name from file name
    const componentName = path.basename(componentPath, path.extname(componentPath));
    
    // Simple extraction of text content (this is a basic implementation)
    let markdownContent = `# ${componentName}\n\n`;
    
    // Extract text from JSX content (simplified)
    const textContent = content
      // Remove imports and exports
      .replace(/import\s+.*?\s+from\s+['"].*?['"];?/g, '')
      .replace(/export\s+.*?\s+from\s+['"].*?['"];?/g, '')
      .replace(/export\s+default\s+.*?;?/g, '')
      // Extract text from JSX
      .replace(/<[^>]*>([^<]*)<\/[^>]*>/g, (match, p1) => p1 ? `\n${p1.trim()}\n` : '')
      // Remove remaining JSX tags
      .replace(/<[^>]*>/g, '')
      // Clean up multiple newlines
      .replace(/\n{3,}/g, '\n\n')
      .trim();
    
    markdownContent += textContent;
    
    // Ensure output directory exists
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Write markdown file
    fs.writeFileSync(outputPath, markdownContent);
    console.log(`Created: ${outputPath}`);
    
  } catch (error) {
    console.error(`Error processing ${componentPath}:`, error.message);
  }
}

// Function to process a directory
function processDirectory(dir, outputBase) {
  try {
    const files = fs.readdirSync(dir);
    
    for (const file of files) {
      if (IGNORE_DIRS.includes(file)) continue;
      
      const filePath = path.join(dir, file);
      const stat = fs.statSync(filePath);
      
      if (stat.isDirectory()) {
        // Process subdirectories
        const newOutputBase = path.join(outputBase, file);
        processDirectory(filePath, newOutputBase);
      } else if (file.match(/\.(tsx|jsx|md)$/)) {
        // Process component files
        const relativePath = path.relative(SOURCE_DIR, filePath);
        const outputPath = path.join(
          OUTPUT_DIR,
          relativePath.replace(/\.(tsx|jsx)$/, '.md')
        );
        
        if (file.endsWith('.md')) {
          // Copy markdown files directly
          const content = fs.readFileSync(filePath, 'utf8');
          const outputDir = path.dirname(outputPath);
          
          if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
          }
          
          fs.writeFileSync(outputPath, content);
          console.log(`Copied: ${outputPath}`);
        } else {
          // Convert React components to markdown
          createMarkdownFromComponent(filePath, outputPath);
        }
      }
    }
  } catch (error) {
    console.error(`Error processing directory ${dir}:`, error.message);
  }
}

// Main function
function main() {
  console.log('Starting documentation generation...');
  
  // Process the source directory
  processDirectory(SOURCE_DIR, OUTPUT_DIR);
  
  console.log('\nDocumentation generation complete!');
  console.log(`Output directory: ${path.resolve(OUTPUT_DIR)}`);
}

// Run the main function
main();
