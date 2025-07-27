# LangServe Documentation

## Introduction
LangServe helps deploy LangChain chains as REST APIs. It provides:
- Automatic API endpoint generation
- Built-in documentation (OpenAPI/Swagger)
- Input/output validation
- Authentication and rate limiting

## Key Features

### 1. Automatic API Generation
- Convert any LangChain chain to a REST API
- Support for streaming responses
- Batch processing endpoints

### 2. Built-in Documentation
- Interactive API docs
- Schema validation
- Example requests/responses

### 3. Deployment Ready
- FastAPI backend
- Container support (Dfile included)
- Horizontal scaling

## Quick Start

1. Install LangServe:
```bash
pip install langserve
```

2. Create a simple API:
```python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes

app = FastAPI()

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
model = ChatOpenAI()

add_routes(
    app,
    prompt | model,
    path="/joke"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

3. Run the server:
```bash
python app.py
```

4. Make requests:
```bash
curl -X POST http://localhost:8000/joke/invoke \
  -H "Content-Type: application/json" \
  -d '{"input":{"topic":"programming"}}'
```

## Common Use Cases
- Serving LLM applications as APIs
- Building microservices with LangChain
- Creating backend services for web/mobile apps
- Integrating with existing infrastructure
