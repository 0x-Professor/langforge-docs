# LangServe Documentation

## Introduction
LangServe helps deploy LangChain chains as REST APIs. It provides:
- Automatic API endpoint generation
- Built-in documentation (OpenAPI/Swagger)
- Input/output validation
- Authentication and rate limiting
- Streaming support
- Batch processing capabilities

## Key Features

### 1. Automatic API Generation
- Convert any LangChain chain to a REST API
- Support for streaming responses
- Batch processing endpoints
- Async support for better performance

### 2. Built-in Documentation
- Interactive API docs with Swagger UI
- Automatic schema validation
- Example requests/responses
- Type hints integration

### 3. Deployment Ready
- FastAPI backend with high performance
- Container support (Dockerfile included)
- Horizontal scaling capabilities
- Health checks and monitoring

## Installation

```bash
pip install "langserve[all]"
```

## Quick Start

### Basic Setup

```python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Create a chain
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt | model

# Add the chain route
add_routes(
    app,
    chain,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

### Multiple Endpoints

```python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

app = FastAPI()

# Create different chains
joke_chain = (
    ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)

poem_chain = (
    ChatPromptTemplate.from_template("Write a poem about {topic}")
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)

# Add multiple routes
add_routes(app, joke_chain, path="/joke")
add_routes(app, poem_chain, path="/poem")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

## Advanced Features

### Streaming Responses

```python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

app = FastAPI()

# Chain that supports streaming
chain = (
    ChatPromptTemplate.from_template("Tell me a story about {topic}")
    | ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
)

add_routes(
    app,
    chain,
    path="/story",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
)
```

### Custom Input/Output Types

```python
from typing import List
from pydantic import BaseModel, Field
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

# Define custom types
class JokeRequest(BaseModel):
    topic: str = Field(description="The topic for the joke")
    style: str = Field(default="funny", description="Style of the joke")

class JokeResponse(BaseModel):
    joke: str = Field(description="The generated joke")
    topic: str = Field(description="The topic used")

app = FastAPI()

# Create a chain with custom types
def format_joke_input(request: JokeRequest) -> dict:
    return {"topic": request.topic, "style": request.style}

def format_joke_output(response: str) -> JokeResponse:
    return JokeResponse(joke=response, topic="custom")

chain = (
    RunnableLambda(format_joke_input)
    | ChatPromptTemplate.from_template("Tell me a {style} joke about {topic}")
    | ChatOpenAI(model="gpt-3.5-turbo")
    | RunnableLambda(format_joke_output)
)

add_routes(
    app,
    chain.with_types(input_type=JokeRequest, output_type=JokeResponse),
    path="/custom-joke",
)
```

### Authentication and Middleware

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import os

app = FastAPI()
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    if credentials.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Protected chain
protected_chain = (
    ChatPromptTemplate.from_template("Answer this question: {question}")
    | ChatOpenAI(model="gpt-4")
)

add_routes(
    app,
    protected_chain,
    path="/protected",
    dependencies=[Depends(verify_token)],
)
```

## Usage Examples

### Making Requests

#### Standard Request
```bash
curl -X POST "http://localhost:8000/joke/invoke" \
     -H "Content-Type: application/json" \
     -d '{"input": {"topic": "programming"}}'
```

#### Streaming Request
```bash
curl -X POST "http://localhost:8000/story/stream" \
     -H "Content-Type: application/json" \
     -d '{"input": {"topic": "dragons"}}' \
     --no-buffer
```

#### Batch Request
```bash
curl -X POST "http://localhost:8000/joke/batch" \
     -H "Content-Type: application/json" \
     -d '{"inputs": [
       {"topic": "programming"}, 
       {"topic": "data science"}
     ]}'
```

### Python Client

```python
from langserve import RemoteRunnable

# Connect to the remote chain
remote_chain = RemoteRunnable("http://localhost:8000/joke/")

# Use it like a local chain
result = remote_chain.invoke({"topic": "programming"})
print(result)

# Streaming
for chunk in remote_chain.stream({"topic": "programming"}):
    print(chunk, end="", flush=True)

# Batch processing
results = remote_chain.batch([
    {"topic": "programming"},
    {"topic": "data science"}
])
print(results)
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/docs || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration

```python
import os
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langserve import add_routes

# Environment-based configuration
app = FastAPI(
    title=os.getenv("APP_TITLE", "LangChain Server"),
    version=os.getenv("APP_VERSION", "1.0.0"),
)

# Configure model
model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    temperature=float(os.getenv("MODEL_TEMPERATURE", "0.7")),
    max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": os.getenv("APP_VERSION", "1.0.0")}
```

### Production Considerations

#### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/joke/invoke")
@limiter.limit("5/minute")
async def rate_limited_joke(request: Request):
    # Your endpoint logic here
    pass
```

#### Monitoring and Logging
```python
import logging
from fastapi import FastAPI, Request
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Request {request_id}: {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Request {request_id} completed in {process_time:.4f}s")
    
    return response
```

## Best Practices

### 1. Chain Design for APIs
- Keep chains stateless when possible
- Use proper input validation
- Handle errors gracefully
- Implement timeouts

### 2. Performance Optimization
- Use async/await for I/O operations
- Implement caching for expensive operations
- Monitor memory usage
- Use connection pooling

### 3. Security
- Always validate inputs
- Use authentication for production
- Implement rate limiting
- Don't expose sensitive information in error messages

### 4. Monitoring
- Log all requests and responses
- Monitor response times
- Track error rates
- Set up alerts for critical issues

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill the process
   kill -9 <PID>
   ```

2. **CORS Issues**
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

3. **Large Payload Issues**
   ```python
   # Increase payload size limit
   app = FastAPI()
   app.router.route_class = CustomAPIRoute  # Custom route with larger limits
   ```

## API Endpoints

LangServe automatically generates these endpoints for each chain:

- `POST /your-chain/invoke` - Single invocation
- `POST /your-chain/batch` - Batch processing
- `POST /your-chain/stream` - Streaming responses
- `GET /your-chain/playground` - Interactive playground
- `GET /docs` - OpenAPI documentation
- `GET /redoc` - Alternative documentation

## Common Use Cases
- Serving LLM applications as APIs
- Building microservices with LangChain
- Creating backend services for web/mobile apps
- Integrating with existing infrastructure
- Rapid prototyping and testing
- Multi-tenant AI services
