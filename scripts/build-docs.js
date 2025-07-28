const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

/**
 * Build script for LangForge Documentation
 * Builds static assets and prepares for deployment
 */

console.log('🏗️  Building LangForge Documentation...\n');

// Configuration
const config = {
  sourceDir: './docs',
  outputDir: './dist',
  assetsDir: './assets',
  publicDir: './public'
};

// Clean output directory
console.log('🧹 Cleaning output directory...');
if (fs.existsSync(config.outputDir)) {
  fs.rmSync(config.outputDir, { recursive: true });
}
fs.mkdirSync(config.outputDir, { recursive: true });

// Build VitePress documentation
console.log('📚 Building documentation with VitePress...');
try {
  execSync('npm run docs:build', { stdio: 'inherit' });
} catch (error) {
  console.error('❌ Documentation build failed:', error.message);
  process.exit(1);
}

// Copy static assets
console.log('📁 Copying static assets...');
if (fs.existsSync(config.assetsDir)) {
  copyRecursiveSync(config.assetsDir, path.join(config.outputDir, 'assets'));
}

if (fs.existsSync(config.publicDir)) {
  copyRecursiveSync(config.publicDir, config.outputDir);
}

// Generate sitemap
console.log('🗺️  Generating sitemap...');
generateSitemap();

// Generate robots.txt
console.log('🤖 Generating robots.txt...');
generateRobotsTxt();

// Optimize images
console.log('🖼️  Optimizing images...');
optimizeImages();

// Generate manifest
console.log('📋 Generating build manifest...');
generateBuildManifest();

console.log('\n✅ Build completed successfully!');
console.log(`📦 Output: ${config.outputDir}`);
console.log(`🌐 Ready for deployment\n`);

// Utility functions
function copyRecursiveSync(src, dest) {
  const exists = fs.existsSync(src);
  const stats = exists && fs.statSync(src);
  const isDirectory = exists && stats.isDirectory();
  
  if (isDirectory) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    fs.readdirSync(src).forEach(childItemName => {
      copyRecursiveSync(
        path.join(src, childItemName),
        path.join(dest, childItemName)
      );
    });
  } else {
    fs.copyFileSync(src, dest);
  }
}

function generateSitemap() {
  const baseUrl = process.env.BASE_URL || 'https://langforge.dev';
  const pages = getAllPages(config.sourceDir);
  
  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${pages.map(page => `  <url>
    <loc>${baseUrl}${page.url}</loc>
    <lastmod>${new Date().toISOString().split('T')[0]}</lastmod>
    <changefreq>${page.changefreq || 'weekly'}</changefreq>
    <priority>${page.priority || '0.8'}</priority>
  </url>`).join('\n')}
</urlset>`;
  
  fs.writeFileSync(path.join(config.outputDir, 'sitemap.xml'), sitemap);
}

function generateRobotsTxt() {
  const baseUrl = process.env.BASE_URL || 'https://langforge.dev';
  const robots = `User-agent: *
Allow: /

Sitemap: ${baseUrl}/sitemap.xml

# Disallow crawling of API endpoints
Disallow: /api/
Disallow: /_next/
Disallow: /admin/

# Allow crawling of documentation
Allow: /docs/
Allow: /examples/
Allow: /guides/`;

  fs.writeFileSync(path.join(config.outputDir, 'robots.txt'), robots);
}

function getAllPages(dir, baseUrl = '') {
  const pages = [];
  const items = fs.readdirSync(dir);
  
  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
      pages.push(...getAllPages(fullPath, `${baseUrl}/${item}`));
    } else if (item.endsWith('.md') && item !== 'README.md') {
      const url = `${baseUrl}/${item.replace('.md', '')}`;
      pages.push({
        url: url,
        priority: url === '' ? '1.0' : '0.8',
        changefreq: 'weekly'
      });
    }
  }
  
  return pages;
}

function optimizeImages() {
  // Simple image optimization - in production, use imagemin or similar
  const imageDir = path.join(config.outputDir, 'images');
  if (fs.existsSync(imageDir)) {
    console.log('   Images found, optimization recommended for production');
  }
}

function generateBuildManifest() {
  const manifest = {
    buildTime: new Date().toISOString(),
    version: process.env.npm_package_version || '1.0.0',
    environment: process.env.NODE_ENV || 'development',
    commit: process.env.GITHUB_SHA || 'local',
    branch: process.env.GITHUB_REF_NAME || 'main'
  };
  
  fs.writeFileSync(
    path.join(config.outputDir, 'build-manifest.json'),
    JSON.stringify(manifest, null, 2)
  );
}