<div align="center">

# 🔧 Troubleshooting Guide

Common issues and solutions for LangForge Documentation and LangChain applications.

## 🚨 Common Issues

### 1. **API Rate Limits**

**Problem**: Getting rate limit errors from OpenAI/other providers

**Solutions**:
```python
from langchain.llms import OpenAI
import time
import backoff

# Exponential backoff retry
@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    max_time=60
)
def safe_llm_call(prompt):
    return OpenAI()(prompt)

# Rate limiting with delays
class RateLimitedLLM:
    def __init__(self, requests_per_minute=60):
        self.rpm = requests_per_minute
        self.last_request = 0
    
    def call(self, prompt):
        # Ensure minimum time between requests
        time_since_last = time.time() - self.last_request
        min_interval = 60.0 / self.rpm
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request = time.time()
        return OpenAI()(prompt)
```

### 2. **Memory Issues**

**Problem**: Application running out of memory

**Solutions**:
```python
# Clear memory periodically
import gc

def cleanup_memory():
    gc.collect()
    
# Limit conversation history
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # Keep only last 5 exchanges

# Use memory-efficient vector stores
from langchain.vectorstores import FAISS

# For large datasets, use disk-based storage
vectorstore = FAISS.from_documents(
    documents, 
    embeddings,
    allow_dangerous_deserialization=True
)
vectorstore.save_local("./vectorstore")
```

### 3. **Slow Response Times**

**Problem**: Application is too slow

**Solutions**:
```python
# Enable caching
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
set_llm_cache(InMemoryCache())

# Use async operations
import asyncio

async def fast_processing(prompts):
    tasks = [llm.agenerate([prompt]) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return results

# Optimize model choice
fast_llm = OpenAI(model_name="gpt-3.5-turbo", max_tokens=100)
```

## 🐛 Debugging Techniques

### Enable Debug Logging
```python
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langchain")

# Add custom logging to your chains
from langchain.callbacks import StdOutCallbackHandler

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    callbacks=[StdOutCallbackHandler()]
)
```

### Environment Variables
```bash
# Debug environment variables
export LANGCHAIN_VERBOSE=true
export LANGCHAIN_DEBUG=true
export OPENAI_LOG_LEVEL=debug
```

## 📊 Performance Monitoring

### Health Check Implementation
```python
from fastapi import FastAPI
import psutil
import time

app = FastAPI()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent(),
        "version": "1.0.0"
    }
```

### Error Tracking
```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1
)
```

## 🔍 Common Error Messages

### "Token limit exceeded"
- **Cause**: Input + output tokens exceed model limit
- **Solution**: Truncate input or use a model with higher limits

### "API key not found"
- **Cause**: Missing or invalid API key
- **Solution**: Check environment variables and key validity

### "Rate limit exceeded"
- **Cause**: Too many requests per minute
- **Solution**: Implement rate limiting and retry logic

### "Context length exceeded"
- **Cause**: Conversation history too long
- **Solution**: Use conversation summary or truncate history

## 🛠️ Development Tips

### 1. **Local Development Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run development server
npm run dev
```

### 2. **Testing Configuration**
```javascript
// jest.config.js
module.exports = {
  testEnvironment: 'node',
  setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  }
};
```

### 3. **Docker Development**
```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "3000:3000"
    volumes:
      - .:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
    command: npm run dev
```

## 🚀 Production Deployment

### Common Deployment Issues

**Issue**: "Module not found" errors
```bash
# Solution: Ensure all dependencies are installed
npm ci --only=production
```

**Issue**: Port already in use
```bash
# Solution: Change port or kill existing process
export PORT=3001
# Or
pkill -f "node.*server.js"
```

**Issue**: Permission denied
```bash
# Solution: Run with proper permissions
sudo chown -R $USER:$USER /app
chmod +x scripts/deploy.sh
```

### Environment-Specific Configuration
```javascript
// config/production.js
module.exports = {
  port: process.env.PORT || 3000,
  host: process.env.HOST || '0.0.0.0',
  nodeEnv: 'production',
  security: {
    cors: {
      origin: process.env.CORS_ORIGIN || 'https://langforge.dev'
    },
    rateLimit: {
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100
    }
  }
};
```

## 📞 Getting Help

### Support Channels
- **Documentation**: Check this guide and [FAQ](faq.md)
- **GitHub Issues**: [Report bugs](https://github.com/0x-Professor/langforge-docs/issues)
- **Discussions**: [Community help](https://github.com/0x-Professor/langforge-docs/discussions)
- **Email**: support@langforge.dev

### Before Reporting Issues
1. Check the [FAQ](faq.md)
2. Search existing GitHub issues
3. Try the latest version
4. Provide minimal reproduction steps
5. Include error logs and environment details

### Issue Template
```markdown
**Bug Description**
A clear description of what the bug is.

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Windows 10, macOS 12.1]
- Node.js version: [e.g., 18.17.0]
- npm version: [e.g., 9.6.7]
- Browser: [e.g., Chrome 91.0]

**Additional Context**
Any other context about the problem.
```