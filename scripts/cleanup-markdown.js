import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const DOCS_DIR = path.join(__dirname, '..', 'docs');

// Common patterns to clean up
const CLEANUP_PATTERNS = [
  // Remove JSX/TSX code blocks
  { 
    pattern: /```(jsx?|tsx?)[\s\S]*?```/g, 
    replacement: '' 
  },
  // Remove import/export statements
  { 
    pattern: /^import\s+.*?from\s+['"].*?['"];?\n?/gm, 
    replacement: '' 
  },
  { 
    pattern: /^export\s+(default\s+)?(const|let|var|function|class|interface|type|enum)\s+.*?\{/gm, 
    replacement: (match) => `# ${match.replace(/^export\s+(default\s+)?/, '')}`
  },
  // Clean up React component definitions
  { 
    pattern: /const\s+([A-Z][A-Za-z0-9]*)\s*=\s*\(\{[^}]*\}\)\s*=>\s*\{/g, 
    replacement: '## $1 Component\n\n### Props'
  },
  // Clean up TypeScript interfaces
  { 
    pattern: /interface\s+([A-Z][A-Za-z0-9]*)\s*\{([^}]*)\}/g, 
    replacement: '### $1\n\n```typescript\ninterface $1 {$2}\n```\n\n**Properties:**\n'
  },
  // Clean up prop types
  { 
    pattern: /\*\s*@param\s+\{([^}]*)\}\s+([^\s]+)(?:\s+-\s+)?([^\n]*)/g, 
    replacement: '- **$2** `$1` - $3\n'
  },
  // Remove empty lines at the start of the file
  { 
    pattern: /^\s*\n+/, 
    replacement: '' 
  },
  // Clean up multiple consecutive empty lines
  { 
    pattern: /\n{3,}/g, 
    replacement: '\n\n' 
  },
  // Clean up component closing tags
  { 
    pattern: /<\/[A-Z][A-Za-z0-9]*>\s*$/, 
    replacement: '' 
  }
];

// Function to clean markdown content
function cleanMarkdown(content) {
  let cleaned = content;
  
  // Apply all cleanup patterns
  for (const { pattern, replacement } of CLEANUP_PATTERNS) {
    cleaned = cleaned.replace(pattern, replacement);
  }
  
  return cleaned;
}

// Function to process all markdown files
function processMarkdownFiles(dir) {
  const items = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const item of items) {
    const fullPath = path.join(dir, item.name);
    
    if (item.isDirectory()) {
      // Recursively process subdirectories
      processMarkdownFiles(fullPath);
    } else if (item.name.endsWith('.md')) {
      try {
        // Read the file
        const content = fs.readFileSync(fullPath, 'utf8');
        
        // Clean the markdown
        const cleanedContent = cleanMarkdown(content);
        
        // Write the cleaned content back to the file
        fs.writeFileSync(fullPath, cleanedContent);
        console.log(`Cleaned: ${fullPath}`);
      } catch (error) {
        console.error(`Error processing ${fullPath}:`, error.message);
      }
    }
  }
}

// Main function
function main() {
  console.log('Starting markdown cleanup...');
  
  try {
    // Process all markdown files in the docs directory
    processMarkdownFiles(DOCS_DIR);
    
    console.log('\nMarkdown cleanup complete!');
    console.log('All documentation files have been cleaned and formatted.');
  } catch (error) {
    console.error('Error during markdown cleanup:', error);
    process.exit(1);
  }
}

// Run the main function
main();
