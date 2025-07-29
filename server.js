const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const compression = require('compression');
const morgan = require('morgan');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

// Import professional services
const SearchService = require('./services/SearchService');
const AnalyticsService = require('./services/AnalyticsService');

const app = express();
const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || '0.0.0.0';

// Initialize services
const searchService = new SearchService();
const analyticsService = new AnalyticsService();

// Initialize search service on startup
searchService.initialize().catch(error => {
  console.error('Failed to initialize search service:', error);
});

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ['self'],
      styleSrc: ['self', 'unsafe-inline', 'https://fonts.googleapis.com'],
      fontSrc: ['self', 'https://fonts.gstatic.com'],
      imgSrc: ['self', 'data:', 'https:'],
      scriptSrc: ['self', 'unsafe-inline', 'https://www.googletagmanager.com'],
      connectSrc: ['self', 'https://api.github.com']
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

// Session tracking middleware
app.use((req, res, next) => {
  // Generate or retrieve session ID
  let sessionId = req.headers['x-session-id'] || req.query.sessionId;
  if (!sessionId) {
    sessionId = uuidv4();
    res.setHeader('x-session-id', sessionId);
  }
  
  req.sessionId = sessionId;
  
  // Track page view for HTML requests
  if (req.method === 'GET' && req.accepts('html')) {
    analyticsService.trackPageView({
      sessionId: req.sessionId,
      page: req.path,
      referrer: req.get('Referrer'),
      userAgent: req.get('User-Agent'),
      ip: req.ip,
      country: req.get('CF-IPCountry') // Cloudflare header
    });
  }
  
  next();
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

// Enhanced search endpoint with professional service
app.get('/api/search', async (req, res) => {
  try {
    const { q: query, category, limit = 10, suggestions = false } = req.query;
    
    if (!query) {
      return res.status(400).json({ error: 'Query parameter required' });
    }

    // Track search
    analyticsService.trackSearch({
      sessionId: req.sessionId,
      query,
      page: req.get('Referer') || 'unknown'
    });

    if (suggestions === 'true') {
      const suggestions = searchService.getSuggestions(query, parseInt(limit));
      return res.json({ suggestions });
    }

    const results = searchService.search(query, {
      limit: parseInt(limit),
      category,
      includeContent: true
    });

    // Track search results
    analyticsService.trackSearch({
      sessionId: req.sessionId,
      query,
      results: results.length,
      page: req.get('Referer') || 'unknown'
    });

    res.json({ 
      query,
      results,
      total: results.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({ error: 'Search service unavailable' });
  }
});

// Enhanced analytics endpoint
app.post('/api/analytics', (req, res) => {
  try {
    const { event, page, data = {} } = req.body;
    
    if (!event) {
      return res.status(400).json({ error: 'Event type required' });
    }

    // Add session context
    const eventData = {
      ...data,
      sessionId: req.sessionId,
      page: page || req.get('Referer'),
      userAgent: req.get('User-Agent'),
      ip: req.ip
    };

    analyticsService.trackEvent(event, eventData);
    
    res.json({ status: 'recorded', timestamp: new Date().toISOString() });
  } catch (error) {
    console.error('Analytics error:', error);
    res.status(500).json({ error: 'Analytics service unavailable' });
  }
});

// Enhanced feedback endpoint
app.post('/api/feedback', (req, res) => {
  try {
    const { page, rating, comment, email, category } = req.body;
    
    if (!rating) {
      return res.status(400).json({ error: 'Rating is required' });
    }

    analyticsService.trackFeedback({
      sessionId: req.sessionId,
      page: page || req.get('Referer'),
      rating: parseInt(rating),
      comment,
      email,
      category
    });
    
    res.json({ 
      status: 'received', 
      message: 'Thank you for your feedback!',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Feedback error:', error);
    res.status(500).json({ error: 'Feedback service unavailable' });
  }
});

// New analytics dashboard endpoint (for internal use)
app.get('/api/analytics/dashboard', async (req, res) => {
  try {
    // Basic authentication check (implement proper auth in production)
    const authHeader = req.headers.authorization;
    if (!authHeader || authHeader !== `Bearer ${process.env.ADMIN_TOKEN}`) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const days = parseInt(req.query.days) || 30;
    const dashboardData = await analyticsService.getDashboardData(days);
    
    res.json(dashboardData);
  } catch (error) {
    console.error('Dashboard error:', error);
    res.status(500).json({ error: 'Dashboard service unavailable' });
  }
});

// Search suggestions endpoint
app.get('/api/search/suggestions', (req, res) => {
  try {
    const { q: query, limit = 5 } = req.query;
    
    if (!query || query.length < 2) {
      return res.json({ suggestions: [] });
    }

    const suggestions = searchService.getSuggestions(query, parseInt(limit));
    res.json({ suggestions });
  } catch (error) {
    console.error('Suggestions error:', error);
    res.json({ suggestions: [] });
  }
});

// Popular searches endpoint
app.get('/api/search/popular', (req, res) => {
  try {
    const popular = searchService.getPopularSearches();
    res.json({ popular });
  } catch (error) {
    console.error('Popular searches error:', error);
    res.json({ popular: [] });
  }
});

// Search index rebuild endpoint (for admin use)
app.post('/api/search/rebuild', async (req, res) => {
  try {
    // Basic authentication check
    const authHeader = req.headers.authorization;
    if (!authHeader || authHeader !== `Bearer ${process.env.ADMIN_TOKEN}`) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    await searchService.rebuildIndex();
    res.json({ status: 'rebuilt', timestamp: new Date().toISOString() });
  } catch (error) {
    console.error('Rebuild error:', error);
    res.status(500).json({ error: 'Failed to rebuild search index' });
  }
});

// Catch-all handler for SPA
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// Error handling middleware
app.use((error, req, res, next) => {
  if (next) {
    // Use next parameter to avoid unused variable warning
    next();
  }
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