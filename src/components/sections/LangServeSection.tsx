import { DocSection, FeatureCard, QuickStart } from '@/components/DocSection';
import { CodeBlock } from '@/components/CodeBlock';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Database, Zap, Globe, Settings, Code, Shield } from 'lucide-react';

export const LangServeSection = () => {
  const basicSetupCode = `from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

# Create FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# Create a simple chain
model = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
parser = StrOutputParser()

chain = prompt | model | parser

# Add the chain route
add_routes(
    app,
    chain,
    path="/joke",
)

# Add a more complex chain
qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the following question:

Question: {question}
Context: {context}

Answer:
""")

qa_chain = qa_prompt | model | parser

add_routes(
    app,
    qa_chain,
    path="/qa",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)`;

  const advancedCode = `from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI(
    title="Advanced LangChain API",
    version="1.0",
    description="Advanced LangChain server with RAG and custom endpoints",
)

# Initialize components
model = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

# RAG Chain Setup
class DocumentStore:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
    
    def add_documents(self, texts: List[str]):
        """Add documents to the vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Split texts
        splits = []
        for text in texts:
            splits.extend(text_splitter.split_text(text))
        
        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_texts(
                splits, 
                embeddings,
                persist_directory="./chroma_db"
            )
        else:
            self.vectorstore.add_texts(splits)
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever()
        )
    
    def query(self, question: str) -> str:
        """Query the document store."""
        if self.qa_chain is None:
            raise HTTPException(status_code=400, detail="No documents loaded")
        
        return self.qa_chain.invoke({"query": question})["result"]

# Initialize document store
doc_store = DocumentStore()

# Pydantic models for API
class DocumentRequest(BaseModel):
    texts: List[str]

class QueryRequest(BaseModel):
    question: str

class SummarizeRequest(BaseModel):
    text: str
    max_length: Optional[int] = 150

# Custom endpoints
@app.post("/documents/add")
async def add_documents(request: DocumentRequest):
    """Add documents to the RAG system."""
    try:
        doc_store.add_documents(request.texts)
        return {"message": f"Added {len(request.texts)} documents successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/query")
async def query_documents(request: QueryRequest):
    """Query the document store."""
    try:
        answer = doc_store.query(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Summarization chain
summarize_prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in {max_length} words or less:\\n\\n{text}"
)
summarize_chain = summarize_prompt | model | StrOutputParser()

# Add chain routes
add_routes(
    app,
    summarize_chain,
    path="/summarize",
    input_type=SummarizeRequest,
)

# Add health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "langchain-server"}

# WebSocket support for streaming
from langserve import add_routes
from langchain_core.runnables import RunnableLambda

def streaming_response(inputs):
    """Generate streaming response."""
    for chunk in model.stream(inputs["message"]):
        yield chunk.content

streaming_chain = RunnableLambda(streaming_response)

add_routes(
    app,
    streaming_chain,
    path="/stream",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)`;

  const clientCode = `import requests
import json
from typing import Dict, Any

class LangServeClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def invoke_chain(self, path: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a chain endpoint."""
        url = f"{self.base_url}{path}/invoke"
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"input": inputs}
        )
        
        response.raise_for_status()
        return response.json()
    
    def stream_chain(self, path: str, inputs: Dict[str, Any]):
        """Stream responses from a chain."""
        url = f"{self.base_url}{path}/stream"
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"input": inputs},
            stream=True
        )
        
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    yield data
                except json.JSONDecodeError:
                    continue
    
    def add_documents(self, texts: list) -> Dict[str, Any]:
        """Add documents to the RAG system."""
        url = f"{self.base_url}/documents/add"
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"texts": texts}
        )
        
        response.raise_for_status()
        return response.json()
    
    def query_documents(self, question: str) -> str:
        """Query the document store."""
        url = f"{self.base_url}/documents/query"
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"question": question}
        )
        
        response.raise_for_status()
        return response.json()["answer"]

# Usage examples
def main():
    client = LangServeClient()
    
    # Test basic chain
    joke_response = client.invoke_chain("/joke", {"topic": "programming"})
    print(f"Joke: {joke_response['output']}")
    
    # Add documents
    documents = [
        "LangChain is a framework for developing applications powered by language models.",
        "LangServe makes it easy to deploy LangChain runnables and chains as a REST API.",
        "FastAPI is used as the underlying web framework for LangServe applications."
    ]
    
    add_result = client.add_documents(documents)
    print(f"Added documents: {add_result}")
    
    # Query documents
    answer = client.query_documents("What is LangServe?")
    print(f"Answer: {answer}")
    
    # Test summarization
    summary_response = client.invoke_chain("/summarize", {
        "text": "Long text to summarize...",
        "max_length": 50
    })
    print(f"Summary: {summary_response['output']}")
    
    # Test streaming
    print("Streaming response:")
    for chunk in client.stream_chain("/stream", {"message": "Tell me about AI"}):
        if 'output' in chunk:
            print(chunk['output'], end='', flush=True)
    print()

if __name__ == "__main__":
    main()`;

  const deploymentCode = `# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker-compose.yml
version: '3.8'

services:
  langserve-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=\${LANGCHAIN_API_KEY}
    volumes:
      - ./chroma_db:/app/chroma_db
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - langserve-app
    restart: unless-stopped

# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream langserve {
        server langserve-app:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        
        location / {
            proxy_pass http://langserve;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}

# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langserve-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langserve
  template:
    metadata:
      labels:
        app: langserve
    spec:
      containers:
      - name: langserve
        image: your-registry/langserve:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: langserve-secrets
              key: openai-api-key
---
apiVersion: v1
kind: Service
metadata:
  name: langserve-service
spec:
  selector:
    app: langserve
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer`;

  return (
    <DocSection
      id="langserve"
      title="LangServe - API Deployment"
      description="Deploy LangChain runnables and chains as production-ready REST APIs with automatic OpenAPI documentation."
      badges={["Deployment", "REST API", "FastAPI"]}
      externalLinks={[
        { title: "LangServe Docs", url: "https://python.langchain.com/docs/langserve" },
        { title: "FastAPI Docs", url: "https://fastapi.tiangolo.com/" },
        { title: "GitHub", url: "https://github.com/langchain-ai/langserve" }
      ]}
    >
      <div className="space-y-8">
        {/* Key Features */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <FeatureCard
              icon={<Zap className="w-6 h-6" />}
              title="Automatic API Generation"
              description="Automatically generate REST APIs from LangChain runnables with FastAPI."
              features={[
                "Auto-generated endpoints",
                "OpenAPI documentation",
                "Type validation",
                "Error handling"
              ]}
            />
            <FeatureCard
              icon={<Globe className="w-6 h-6" />}
              title="Production Ready"
              description="Built on FastAPI with production-grade features and performance."
              features={[
                "High performance",
                "Async support",
                "Authentication",
                "Rate limiting"
              ]}
            />
            <FeatureCard
              icon={<Database className="w-6 h-6" />}
              title="Streaming Support"
              description="Real-time streaming responses for better user experience."
              features={[
                "Server-sent events",
                "WebSocket support",
                "Chunked responses",
                "Progress tracking"
              ]}
            />
            <FeatureCard
              icon={<Code className="w-6 h-6" />}
              title="Client SDKs"
              description="Auto-generated client libraries for multiple programming languages."
              features={[
                "Python client",
                "JavaScript SDK",
                "TypeScript support",
                "OpenAPI clients"
              ]}
            />
            <FeatureCard
              icon={<Settings className="w-6 h-6" />}
              title="Customizable"
              description="Flexible configuration and customization options for advanced use cases."
              features={[
                "Custom middleware",
                "Request/response hooks",
                "Authentication plugins",
                "Monitoring integration"
              ]}
            />
            <FeatureCard
              icon={<Shield className="w-6 h-6" />}
              title="Security"
              description="Built-in security features for production deployments."
              features={[
                "CORS support",
                "API key authentication",
                "Rate limiting",
                "Input validation"
              ]}
            />
          </div>
        </div>

        {/* Code Examples */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Implementation Examples</h2>
          <Tabs defaultValue="basic" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="basic">Basic Setup</TabsTrigger>
              <TabsTrigger value="advanced">Advanced Features</TabsTrigger>
              <TabsTrigger value="client">Client Usage</TabsTrigger>
              <TabsTrigger value="deployment">Deployment</TabsTrigger>
            </TabsList>
            
            <TabsContent value="basic" className="space-y-4">
              <CodeBlock
                title="Basic LangServe Server"
                language="python"
                code={basicSetupCode}
              />
            </TabsContent>
            
            <TabsContent value="advanced" className="space-y-4">
              <CodeBlock
                title="Advanced Server with RAG"
                language="python"
                code={advancedCode}
              />
            </TabsContent>
            
            <TabsContent value="client" className="space-y-4">
              <CodeBlock
                title="Python Client Implementation"
                language="python"
                code={clientCode}
              />
            </TabsContent>
            
            <TabsContent value="deployment" className="space-y-4">
              <CodeBlock
                title="Production Deployment Configuration"
                language="yaml"
                code={deploymentCode}
              />
            </TabsContent>
          </Tabs>
        </div>

        {/* API Endpoints */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Automatic API Endpoints</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Standard Endpoints</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium">POST /invoke</h4>
                    <p className="text-sm text-muted-foreground">Synchronous chain invocation</p>
                  </div>
                  <div>
                    <h4 className="font-medium">POST /batch</h4>
                    <p className="text-sm text-muted-foreground">Batch processing multiple inputs</p>
                  </div>
                  <div>
                    <h4 className="font-medium">POST /stream</h4>
                    <p className="text-sm text-muted-foreground">Streaming responses</p>
                  </div>
                  <div>
                    <h4 className="font-medium">GET /docs</h4>
                    <p className="text-sm text-muted-foreground">Interactive API documentation</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Optional Endpoints</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium">POST /feedback</h4>
                    <p className="text-sm text-muted-foreground">Collect user feedback</p>
                  </div>
                  <div>
                    <h4 className="font-medium">GET /trace_link</h4>
                    <p className="text-sm text-muted-foreground">LangSmith trace links</p>
                  </div>
                  <div>
                    <h4 className="font-medium">GET /config_schema</h4>
                    <p className="text-sm text-muted-foreground">Configuration schema</p>
                  </div>
                  <div>
                    <h4 className="font-medium">GET /input_schema</h4>
                    <p className="text-sm text-muted-foreground">Input validation schema</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Quick Start Guide */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Quick Start</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <QuickStart
              title="Deploy Your First API"
              description="Get a LangServe API running in minutes."
              steps={[
                "Install LangServe: pip install langserve[all]",
                "Create your chain with LangChain components",
                "Add routes to FastAPI app with add_routes()",
                "Run with uvicorn and access /docs for documentation"
              ]}
            />
            <QuickStart
              title="Production Deployment"
              description="Deploy to production with proper configuration."
              steps={[
                "Configure environment variables and secrets",
                "Set up reverse proxy with nginx or cloud load balancer",
                "Enable monitoring and logging",
                "Deploy with Docker or Kubernetes for scalability"
              ]}
            />
          </div>
        </div>

        {/* Best Practices */}
        <Card className="shadow-card border-l-4 border-l-primary">
          <CardHeader>
            <CardTitle>LangServe Best Practices</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="font-medium">Development</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Use Pydantic models for request/response validation</li>
                  <li>• Implement proper error handling and logging</li>
                  <li>• Add health check endpoints</li>
                  <li>• Document your APIs thoroughly</li>
                </ul>
              </div>
              <div className="space-y-3">
                <h4 className="font-medium">Production</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Enable authentication and rate limiting</li>
                  <li>• Use HTTPS and proper CORS configuration</li>
                  <li>• Monitor performance and costs</li>
                  <li>• Implement graceful shutdown handling</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DocSection>
  );
};