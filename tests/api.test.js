const request = require('supertest');
const app = require('../server');

describe('API Endpoints', () => {
  describe('GET /api/health', () => {
    it('should return health status', async () => {
      const response = await request(app)
        .get('/api/health')
        .expect(200);
      
      expect(response.body).toHaveProperty('status', 'ok');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('version');
    });
  });

  describe('GET /api/search', () => {
    it('should return search results', async () => {
      const response = await request(app)
        .get('/api/search?q=langchain')
        .expect(200);
      
      expect(response.body).toHaveProperty('results');
      expect(Array.isArray(response.body.results)).toBe(true);
    });

    it('should return 400 for missing query', async () => {
      await request(app)
        .get('/api/search')
        .expect(400);
    });
  });

  describe('POST /api/feedback', () => {
    it('should accept feedback', async () => {
      const feedback = {
        page: '/docs/langchain',
        rating: 5,
        comment: 'Great documentation!',
        email: 'test@example.com'
      };

      const response = await request(app)
        .post('/api/feedback')
        .send(feedback)
        .expect(200);
      
      expect(response.body).toHaveProperty('status', 'received');
    });
  });

  describe('GET /api/sitemap', () => {
    it('should return XML sitemap', async () => {
      const response = await request(app)
        .get('/api/sitemap')
        .expect(200);
      
      expect(response.headers['content-type']).toBe('application/xml; charset=utf-8');
      expect(response.text).toContain('<?xml version="1.0" encoding="UTF-8"?>');
    });
  });
});