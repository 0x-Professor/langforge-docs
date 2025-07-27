const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const { marked } = require('marked');
const { renderToStaticMarkup } = require('react-dom/server');
const React = require('react');

// Create output directory if it doesn't exist
const outputDir = path.join(__dirname, '..', 'docs');
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Function to convert React component to markdown
async function convertComponentToMarkdown(componentPath, outputPath) {
  try {
    // Import the component dynamically
    const componentModule = require(componentPath);
    const componentName = Object.keys(componentModule).find(key => 
      key.endsWith('Section') || key.endsWith('Page')
    );
    
    if (!componentName) {
      console.warn(`No valid component found in ${componentPath}`);
      return;
    }
    
    const Component = componentModule[componentName];
    
    // Render component to static markup
    const html = renderToStaticMarkup(React.createElement(Component));
    
    // Convert HTML to markdown
    const markdown = marked(html);
    
    // Write to file
    fs.writeFileSync(outputPath, `# ${componentName}\n\n${markdown}`);
    console.log(`Successfully converted ${componentPath} to ${outputPath}`);
  } catch (error) {
    console.error(`Error converting ${componentPath}:`, error);
  }
}

// Function to process all components in a directory
async function processDirectory(directory, outputBaseDir) {
  const files = fs.readdirSync(directory);
  
  for (const file of files) {
    const filePath = path.join(directory, file);
    const stat = fs.statSync(filePath);
    
    if (stat.isDirectory()) {
      // Recursively process subdirectories
      const newOutputDir = path.join(outputBaseDir, file);
      if (!fs.existsSync(newOutputDir)) {
        fs.mkdirSync(newOutputDir, { recursive: true });
      }
      await processDirectory(filePath, newOutputDir);
    } else if (file.endsWith('.tsx') && !file.endsWith('.test.tsx') && !file.endsWith('.stories.tsx')) {
      // Process .tsx files (excluding test and story files)
      const outputFileName = file.replace(/\.tsx$/, '.md');
      const outputPath = path.join(outputBaseDir, outputFileName);
      await convertComponentToMarkdown(filePath, outputPath);
    } else if (file.endsWith('.md')) {
      // Copy existing markdown files
      const outputPath = path.join(outputBaseDir, file);
      fs.copyFileSync(filePath, outputPath);
      console.log(`Copied ${filePath} to ${outputPath}`);
    }
  }
}

// Main function
async function main() {
  try {
    // Process sections
    const sectionsDir = path.join(__dirname, '..', 'src', 'components', 'sections');
    await processDirectory(sectionsDir, path.join(outputDir, 'sections'));
    
    // Process docs
    const docsDir = path.join(__dirname, '..', 'src', 'pages', 'docs');
    await processDirectory(docsDir, path.join(outputDir, 'docs'));
    
    console.log('Conversion completed successfully!');
  } catch (error) {
    console.error('Error during conversion:', error);
    process.exit(1);
  }
}

main();
