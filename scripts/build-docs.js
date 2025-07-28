const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

/**
 * Build script for LangForge Documentation
 * Generates optimized static files for production deployment
 */

const BUILD_DIR = path.join(__dirname, '..', 'dist');
const DOCS_DIR = path.join(__dirname, '..', 'docs');

console.log('🏗️  Starting LangForge Documentation build...');

// Clean build directory
if (fs.existsSync(BUILD_DIR)) {
  fs.rmSync(BUILD_DIR, { recursive: true });
  console.log('🧹 Cleaned build directory');
}

// Create build directory
fs.mkdirSync(BUILD_DIR, { recursive: true });

// Build VitePress documentation
try {
  console.log('📚 Building documentation with VitePress...');
  execSync('npm run docs:build', { stdio: 'inherit' });
  console.log('✅ Documentation built successfully');
} catch (error) {
  console.error('❌ Documentation build failed:', error.message);
  process.exit(1);
}

// Generate sitemap
console.log('🗺️  Generating sitemap...');
generateSitemap();

// Generate search index
console.log('🔍 Generating search index...');
generateSearchIndex();

// Optimize assets
console.log('⚡ Optimizing assets...');
optimizeAssets();

// Generate manifest
console.log('📋 Generating manifest...');
generateManifest();

console.log('🎉 Build completed successfully!');
console.log(`📁 Output directory: ${BUILD_DIR}`);

function generateSitemap() {
  const pages = getAllPages(DOCS_DIR);
  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${pages.map(page => `  <url>
    <loc>https://langforge.dev${page}</loc>
    <lastmod>${new Date().toISOString().split('T')[0]}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>${page === '/' ? '1.0' : '0.8'}</priority>
  </url>`).join('\n')}
</urlset>`;

  fs.writeFileSync(path.join(BUILD_DIR, 'sitemap.xml'), sitemap);
  console.log('✅ Sitemap generated');
}

function generateSearchIndex() {
  const searchIndex = {};
  const files = getAllMarkdownFiles(DOCS_DIR);
  
  files.forEach(file => {
    const content = fs.readFileSync(file, 'utf8');
    const relativePath = path.relative(DOCS_DIR, file);
    const title = extractTitle(content);
    const excerpt = content.substring(0, 300).replace(/\n/g, ' ');
    
    searchIndex[relativePath] = {
      title,
      content: content.toLowerCase(),
      excerpt,
      path: `/${relativePath.replace(/\.md$/, '')}`
    };
  });
  
  fs.writeFileSync(
    path.join(BUILD_DIR, 'search-index.json'),
    JSON.stringify(searchIndex, null, 2)
  );
  console.log('✅ Search index generated');
}

function optimizeAssets() {
  // Copy static assets
  const assetsDir = path.join(__dirname, '..', 'assets');
  if (fs.existsSync(assetsDir)) {
    const targetDir = path.join(BUILD_DIR, 'assets');
    fs.mkdirSync(targetDir, { recursive: true });
    copyRecursive(assetsDir, targetDir);
    console.log('✅ Assets copied');
  }
}

function generateManifest() {
  const manifest = {
    name: 'LangForge Documentation',
    short_name: 'LangForge',
    description: 'The Complete Guide to Building Production-Ready LLM Applications',
    start_url: '/',
    display: 'standalone',
    background_color: '#ffffff',
    theme_color: '#3b82f6',
    icons: [
      {
        src: '/favicon-192x192.png',
        sizes: '192x192',
        type: 'image/png'
      },
      {
        src: '/favicon-512x512.png',
        sizes: '512x512',
        type: 'image/png'
      }
    ]
  };
  
  fs.writeFileSync(
    path.join(BUILD_DIR, 'manifest.json'),
    JSON.stringify(manifest, null, 2)
  );
  console.log('✅ Web manifest generated');
}

function getAllPages(dir) {
  const pages = ['/'];
  const files = getAllMarkdownFiles(dir);
  
  files.forEach(file => {
    const relativePath = path.relative(dir, file);
    if (relativePath !== 'README.md') {
      const pagePath = `/${relativePath.replace(/\.md$/, '').replace(/README$/, '')}`;
      pages.push(pagePath.replace(/\/$/, '') || '/');
    }
  });
  
  return [...new Set(pages)];
}

function getAllMarkdownFiles(dir) {
  const files = [];
  const items = fs.readdirSync(dir);
  
  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory() && !item.startsWith('.')) {
      files.push(...getAllMarkdownFiles(fullPath));
    } else if (item.endsWith('.md')) {
      files.push(fullPath);
    }
  }
  
  return files;
}

function extractTitle(content) {
  const match = content.match(/^#\s+(.+)$/m);
  return match ? match[1] : 'Untitled';
}

function copyRecursive(src, dest) {
  const stat = fs.statSync(src);
  if (stat.isDirectory()) {
    fs.mkdirSync(dest, { recursive: true });
    const items = fs.readdirSync(src);
    items.forEach(item => {
      copyRecursive(path.join(src, item), path.join(dest, item));
    });
  } else {
    fs.copyFileSync(src, dest);
  }
}