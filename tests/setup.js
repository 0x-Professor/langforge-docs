// Test setup file
global.console = {
  ...console,
  // Uncomment to ignore console.log during tests
  // log: jest.fn(),
  debug: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};

// Setup test environment variables
process.env.NODE_ENV = 'test';
process.env.PORT = '3001';

// Mock external services during tests
jest.mock('rate-limiter-flexible', () => ({
  RateLimiterMemory: jest.fn().mockImplementation(() => ({
    consume: jest.fn().mockResolvedValue(true)
  }))
}));

// Global test utilities
global.testUtils = {
  delay: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
  mockRequest: (overrides = {}) => ({
    ip: '127.0.0.1',
    headers: {},
    query: {},
    body: {},
    ...overrides
  }),
  mockResponse: () => {
    const res = {};
    res.status = jest.fn().mockReturnValue(res);
    res.json = jest.fn().mockReturnValue(res);
    res.send = jest.fn().mockReturnValue(res);
    res.set = jest.fn().mockReturnValue(res);
    return res;
  }
};