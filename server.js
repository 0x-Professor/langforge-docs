const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const compression = require('compression');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || '0.0.0.0';

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
      fontSrc: ["'self'", "https://fonts.gstatic.com"],
      imgSrc: ["'self'", "data:", "https:"],
      scriptSrc: ["'self'", "'unsafe-inline'", "https://www.googletagmanager.com"],
      connectSrc: ["'self'", "https://api.github.com"]
    }
  }
}));

// CORS configuration
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
  credentials: true
}));

// Compression middleware
app.use(compression());

// Logging
app.use(morgan('combined'));

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const { RateLimiterMemory } = require('rate-limiter-flexible');
const rateLimiter = new RateLimiterMemory({
  keyPrefix: 'middleware',
  points: process.env.RATE_LIMIT_MAX_REQUESTS || 100,
  duration: process.env.RATE_LIMIT_WINDOW_MS || 900, // 15 minutes
});

app.use(async (req, res, next) => {
  try {
    await rateLimiter.consume(req.ip);
    next();
  } catch (rejRes) {
    res.status(429).send('Too Many Requests');
  }
});

// Serve static files
app.use(express.static(path.join(__dirname, 'dist')));
app.use('/docs', express.static(path.join(__dirname, 'docs')));

// API Routes
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    version: process.env.npm_package_version || '1.0.0',
    environment: process.env.NODE_ENV || 'development'
  });
});

app.get('/api/sitemap', (req, res) => {
  const sitemap = generateSitemap();
  res.set('Content-Type', 'application/xml');
  res.send(sitemap);
});

// Search endpoint
app.get('/api/search', (req, res) => {
  const query = req.query.q;
  if (!query) {
    return res.status(400).json({ error: 'Query parameter required' });
  }
  
  // Implement search logic here
  const results = searchDocumentation(query);
  res.json({ results });
});

// Analytics endpoint
app.post('/api/analytics', (req, res) => {
  const { event, page, data } = req.body;
  
  // Log analytics data (implement your analytics service)
  console.log('Analytics:', { event, page, data, timestamp: new Date().toISOString() });
  
  res.json({ status: 'recorded' });
});

// Feedback endpoint
app.post('/api/feedback', (req, res) => {
  const { page, rating, comment, email } = req.body;
  
  // Store feedback (implement your feedback storage)
  console.log('Feedback:', { page, rating, comment, email, timestamp: new Date().toISOString() });
  
  res.json({ status: 'received', message: 'Thank you for your feedback!' });
});

// Catch-all handler for SPA
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Error:', error);
  res.status(500).json({
    error: 'Internal Server Error',
    message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
  });
});

// Utility functions
function generateSitemap() {
  const baseUrl = process.env.BASE_URL || 'https://langforge.dev';
  const pages = [
    '',
    '/docs',
    '/docs/getting-started',
    '/docs/langchain',
    '/docs/langsmith',
    '/docs/langgraph',
    '/docs/langserve',
    '/docs/examples',
    '/docs/guides'
  ];
  
  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${pages.map(page => `  <url>
    <loc>${baseUrl}${page}</loc>
    <lastmod>${new Date().toISOString().split('T')[0]}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>${page === '' ? '1.0' : '0.8'}</priority>
  </url>`).join('\n')}
</urlset>`;
  
  return sitemap;
}

function searchDocumentation(query) {
  // Simple file-based search implementation
  // In production, use Algolia, Elasticsearch, or similar
  const results = [];
  const docsPath = path.join(__dirname, 'docs');
  
  try {
    const files = getAllMarkdownFiles(docsPath);
    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');
      if (content.toLowerCase().includes(query.toLowerCase())) {
        const relativePath = path.relative(docsPath, file);
        const title = extractTitle(content);
        results.push({
          title,
          path: `/docs/${relativePath.replace(/\.md$/, '')}`,
          excerpt: extractExcerpt(content, query)
        });
      }
    }
  } catch (error) {
    console.error('Search error:', error);
  }
  
  return results.slice(0, 10); // Limit results
}

function getAllMarkdownFiles(dir) {
  const files = [];
  const items = fs.readdirSync(dir);
  
  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
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

function extractExcerpt(content, query) {
  const lines = content.split('\n');
  for (const line of lines) {
    if (line.toLowerCase().includes(query.toLowerCase())) {
      return line.substring(0, 150) + '...';
    }
  }
  return content.substring(0, 150) + '...';
}

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  process.exit(0);
});

app.listen(PORT, HOST, () => {
  console.log(`🚀 LangForge Documentation Server running at http://${HOST}:${PORT}`);
  console.log(`📚 Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`🔒 Security headers enabled`);
  console.log(`⚡ Compression enabled`);
  console.log(`🛡️  Rate limiting active`);
});