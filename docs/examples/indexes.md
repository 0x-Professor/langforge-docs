# Indexes and Document Processing

## Overview

Indexes in LangChain provide a way to structure and organize documents for efficient retrieval and search. They are a crucial component for building retrieval-augmented generation (RAG) applications and other document-based workflows.

> **Note:** The typical workflow involves: loading documents, splitting them into chunks, generating embeddings, storing them in a vector database, and then retrieving relevant documents based on similarity search.

## Document Loaders

Document loaders help you load data from various sources into Document objects that LangChain can process.

### Text Files

```python
from langchain_community.document_loaders import TextLoader

# Load a text file
loader = TextLoader("document.txt", encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} documents")
print(f"First document: {documents[0].page_content[:100]}...")
```

### PDF Files

```python
from langchain_community.document_loaders import PyPDFLoader

# Load a PDF file
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Each page becomes a separate document
for i, doc in enumerate(documents):
    print(f"Page {i+1}: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

### Web Pages

```python
from langchain_community.document_loaders import WebBaseLoader
import requests

# Load web pages
loader = WebBaseLoader([
    "https://python.langchain.com/docs/get_started/introduction",
    "https://python.langchain.com/docs/get_started/quickstart"
])

documents = loader.load()

for doc in documents:
    print(f"URL: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:200]}...")
```

### CSV Files

```python
from langchain_community.document_loaders import CSVLoader

# Load CSV data
loader = CSVLoader(
    file_path="data.csv",
    csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['name', 'age', 'city']
    }
)

documents = loader.load()

for doc in documents:
    print(f"Row: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### Directory Loader

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load all text files from a directory
loader = DirectoryLoader(
    path="./documents/",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf8'}
)

documents = loader.load()
print(f"Loaded {len(documents)} documents from directory")
```

## Text Splitters

Since language models have limited context windows, documents often need to be split into smaller chunks.

### Recursive Character Text Splitter

The most commonly used splitter that tries to split on natural boundaries:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,           # Maximum chunk size
    chunk_overlap=200,         # Overlap between chunks
    length_function=len,       # Function to measure length
    separators=["\n\n", "\n", " ", ""]  # Split on these in order
)

# Split documents
split_docs = splitter.split_documents(documents)

print(f"Original documents: {len(documents)}")
print(f"Split into chunks: {len(split_docs)}")

# Examine a chunk
print(f"Sample chunk: {split_docs[0].page_content}")
print(f"Chunk metadata: {split_docs[0].metadata}")
```

### Character Text Splitter

A simpler splitter that splits on a specific character:

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",          # Split on double newlines
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

split_docs = splitter.split_documents(documents)
```

### Token-Based Splitter

Split based on token count rather than character count:

```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=500,    # 500 tokens per chunk
    chunk_overlap=50   # 50 token overlap
)

split_docs = splitter.split_documents(documents)
```

### Custom Text Splitter

Create your own splitter for specific needs:

```python
from langchain.text_splitter import TextSplitter
from typing import List

class ParagraphSplitter(TextSplitter):
    """Split text by paragraphs while respecting size limits."""
    
    def split_text(self, text: str) -> List[str]:
        # Split by paragraphs
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk_size, start new chunk
            if len(current_chunk) + len(paragraph) > self._chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

# Use custom splitter
custom_splitter = ParagraphSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = custom_splitter.split_documents(documents)
```

## Vector Stores

Vector stores are databases that store document embeddings and enable efficient similarity search.

### FAISS (Facebook AI Similarity Search)

Great for local development and small to medium datasets:

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store from documents
vectorstore = FAISS.from_documents(
    documents=split_docs,
    embedding=embeddings
)

# Save the vector store locally
vectorstore.save_local("faiss_index")

# Load the vector store later
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Search for similar documents
query = "What is machine learning?"
similar_docs = vectorstore.similarity_search(query, k=3)

for doc in similar_docs:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

### Chroma

Open-source embedding database with persistent storage:

```python
from langchain_community.vectorstores import Chroma

# Create Chroma vector store
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Directory to persist the database
)

# The database is automatically persisted
# Later, load the existing database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

### Pinecone

Managed vector database for production applications:

```python
from langchain_community.vectorstores import Pinecone
import pinecone

# Initialize Pinecone
pinecone.init(
    api_key="your-pinecone-api-key",
    environment="your-pinecone-environment"
)

# Create index if it doesn't exist
index_name = "langchain-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine"
    )

# Create vector store
vectorstore = Pinecone.from_documents(
    documents=split_docs,
    embedding=embeddings,
    index_name=index_name
)
```

### Weaviate

Open-source vector search engine:

```python
from langchain_community.vectorstores import Weaviate
import weaviate

# Connect to Weaviate instance
client = weaviate.Client(url="http://localhost:8080")

# Create vector store
vectorstore = Weaviate.from_documents(
    documents=split_docs,
    embedding=embeddings,
    client=client,
    by_text=False
)
```

## Retrievers

Retrievers provide a unified interface for document retrieval with various search strategies.

### Basic Similarity Retriever

```python
# Create a basic retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Return top 5 most similar documents
)

# Retrieve relevant documents
query = "How do I use LangChain for document processing?"
docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(docs):
    print(f"Document {i+1}: {doc.page_content[:100]}...")
```

### MMR (Maximal Marginal Relevance) Retriever

Balances similarity and diversity in results:

```python
# MMR retriever for diverse results
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,           # Number of documents to return
        "fetch_k": 20,    # Number of documents to fetch before MMR
        "lambda_mult": 0.7  # Diversity parameter (0=max diversity, 1=max similarity)
    }
)

docs = mmr_retriever.get_relevant_documents(query)
```

### Similarity Score Threshold Retriever

Only return documents above a certain similarity threshold:

```python
# Threshold retriever
threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,  # Only return docs with similarity > 0.8
        "k": 5
    }
)

docs = threshold_retriever.get_relevant_documents(query)
```

### Custom Retriever

Create your own retriever for specific needs:

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class HybridRetriever(BaseRetriever):
    """Combines vector similarity with keyword matching."""
    
    def __init__(self, vectorstore, keyword_weight=0.3):
        self.vectorstore = vectorstore
        self.keyword_weight = keyword_weight
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Get vector similarity results
        vector_docs = self.vectorstore.similarity_search(query, k=10)
        
        # Simple keyword scoring
        query_words = set(query.lower().split())
        
        scored_docs = []
        for doc in vector_docs:
            # Count keyword matches
            doc_words = set(doc.page_content.lower().split())
            keyword_score = len(query_words.intersection(doc_words)) / len(query_words)
            
            # Combine with vector similarity (simplified)
            combined_score = (1 - self.keyword_weight) + self.keyword_weight * keyword_score
            scored_docs.append((doc, combined_score))
        
        # Sort by combined score and return top results
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:5]]

# Use custom retriever
hybrid_retriever = HybridRetriever(vectorstore)
docs = hybrid_retriever.get_relevant_documents(query)
```

## Embeddings

Embeddings convert text into numerical vectors that capture semantic meaning.

### OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # or "text-embedding-3-large"
    openai_api_key="your-openai-api-key"
)

# Generate embeddings for documents
texts = ["This is a sample text.", "Another example sentence."]
doc_embeddings = embeddings.embed_documents(texts)

# Generate embedding for a query
query = "What is the sample about?"
query_embedding = embeddings.embed_query(query)

print(f"Document embeddings shape: {len(doc_embeddings)} x {len(doc_embeddings[0])}")
print(f"Query embedding shape: {len(query_embedding)}")
```

### HuggingFace Embeddings

Run embeddings locally without API calls:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# Use sentence-transformers model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # or 'cuda' for GPU
    encode_kwargs={'normalize_embeddings': True}
)

# Generate embeddings
texts = ["This is a sample text.", "Another example sentence."]
doc_embeddings = embeddings.embed_documents(texts)

print(f"Model: {embeddings.model_name}")
print(f"Embedding dimension: {len(doc_embeddings[0])}")
```

### Custom Embeddings

Create your own embedding class:

```python
from langchain_core.embeddings import Embeddings
from typing import List
import numpy as np

class SimpleWordEmbeddings(Embeddings):
    """Simple word-based embeddings for demonstration."""
    
    def __init__(self, dimension: int = 100):
        self.dimension = dimension
        self.word_vectors = {}
    
    def _get_word_vector(self, word: str) -> np.ndarray:
        """Get or create a vector for a word."""
        if word not in self.word_vectors:
            # Create a simple hash-based vector
            hash_val = hash(word) % (2**32)
            np.random.seed(hash_val)
            self.word_vectors[word] = np.random.normal(0, 1, self.dimension)
        return self.word_vectors[word]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            words = text.lower().split()
            if words:
                # Average word vectors
                vectors = [self._get_word_vector(word) for word in words]
                doc_vector = np.mean(vectors, axis=0)
            else:
                doc_vector = np.zeros(self.dimension)
            embeddings.append(doc_vector.tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query string."""
        return self.embed_documents([text])[0]

# Use custom embeddings
custom_embeddings = SimpleWordEmbeddings(dimension=128)
texts = ["machine learning is powerful", "artificial intelligence is the future"]
embeddings = custom_embeddings.embed_documents(texts)

print(f"Custom embedding dimension: {len(embeddings[0])}")
```

## Retrieval-Augmented Generation (RAG)

Combine document retrieval with language model generation:

### Basic RAG Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Create RAG prompt
template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | StrOutputParser()
)

# Ask questions
questions = [
    "What is LangChain used for?",
    "How do I split documents?",
    "What are vector stores?"
]

for question in questions:
    answer = rag_chain.invoke(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### RAG with Source Attribution

```python
from langchain_core.runnables import RunnableParallel

# Modified RAG chain that returns sources
rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain)

def rag_with_sources(question: str):
    result = rag_chain_with_source.invoke(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print("\nSources:")
    
    for i, doc in enumerate(result['context']):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        print(f"{i+1}. {source} (page {page})")
        print(f"   Excerpt: {doc.page_content[:100]}...")
    print()

# Use RAG with sources
rag_with_sources("How do embeddings work in LangChain?")
```

## Best Practices

### 1. Choose the Right Chunk Size

```python
def find_optimal_chunk_size(documents, queries, chunk_sizes=[500, 1000, 1500, 2000]):
    """Test different chunk sizes to find the optimal one."""
    results = {}
    
    for chunk_size in chunk_sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 10  # 10% overlap
        )
        
        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Test retrieval quality (simplified)
        relevance_scores = []
        for query in queries:
            docs = retriever.get_relevant_documents(query)
            # In practice, you'd use more sophisticated metrics
            avg_length = sum(len(doc.page_content) for doc in docs) / len(docs)
            relevance_scores.append(avg_length)
        
        results[chunk_size] = {
            'num_chunks': len(chunks),
            'avg_relevance': sum(relevance_scores) / len(relevance_scores)
        }
    
    return results

# Test different chunk sizes
# results = find_optimal_chunk_size(documents, test_queries)
```

### 2. Implement Metadata Filtering

```python
# Add metadata to documents for filtering
from langchain_core.documents import Document

documents_with_metadata = [
    Document(
        page_content="Content about machine learning...",
        metadata={"topic": "ML", "difficulty": "beginner", "date": "2024-01-01"}
    ),
    Document(
        page_content="Advanced deep learning concepts...",
        metadata={"topic": "DL", "difficulty": "advanced", "date": "2024-01-15"}
    )
]

# Create vector store with metadata
vectorstore = FAISS.from_documents(documents_with_metadata, embeddings)

# Filter by metadata during search
def search_with_filter(query: str, filter_dict: dict):
    # This is a simplified example - actual implementation depends on vector store
    all_docs = vectorstore.similarity_search(query, k=20)
    
    filtered_docs = []
    for doc in all_docs:
        match = True
        for key, value in filter_dict.items():
            if doc.metadata.get(key) != value:
                match = False
                break
        if match:
            filtered_docs.append(doc)
    
    return filtered_docs[:5]  # Return top 5

# Search for beginner ML content
results = search_with_filter("machine learning basics", {"difficulty": "beginner"})
```

### 3. Monitor Retrieval Quality

```python
import json
from datetime import datetime

class RetrievalLogger:
    """Log retrieval performance for monitoring."""
    
    def __init__(self, log_file="retrieval_logs.json"):
        self.log_file = log_file
    
    def log_retrieval(self, query: str, retrieved_docs: List[Document], 
                     user_feedback: str = None):
        """Log a retrieval event."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_docs_retrieved": len(retrieved_docs),
            "doc_sources": [doc.metadata.get('source', 'unknown') 
                           for doc in retrieved_docs],
            "avg_doc_length": sum(len(doc.page_content) for doc in retrieved_docs) / len(retrieved_docs),
            "user_feedback": user_feedback
        }
        
        # Append to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Logging error: {e}")
    
    def analyze_logs(self):
        """Analyze retrieval performance from logs."""
        try:
            with open(self.log_file, 'r') as f:
                logs = [json.loads(line) for line in f]
            
            total_queries = len(logs)
            avg_docs_per_query = sum(log['num_docs_retrieved'] for log in logs) / total_queries
            
            print(f"Total queries: {total_queries}")
            print(f"Average docs per query: {avg_docs_per_query:.2f}")
            
            # More analysis...
            
        except Exception as e:
            print(f"Analysis error: {e}")

# Use retrieval logger
logger = RetrievalLogger()

def monitored_retrieval(query: str):
    docs = retriever.get_relevant_documents(query)
    logger.log_retrieval(query, docs)
    return docs
```

### 4. Cache Embeddings

```python
import hashlib
import pickle
import os

class EmbeddingCache:
    """Cache embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir="embedding_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        content = f"{text}_{model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_embedding(self, text: str, model_name: str, embeddings_func):
        """Get embedding from cache or compute and cache it."""
        cache_key = self._get_cache_key(text, model_name)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache load error: {e}")
        
        # Compute embedding
        embedding = embeddings_func(text)
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            print(f"Cache save error: {e}")
        
        return embedding

# Use embedding cache
cache = EmbeddingCache()

class CachedEmbeddings:
    def __init__(self, base_embeddings):
        self.base_embeddings = base_embeddings
        self.cache = EmbeddingCache()
        self.model_name = getattr(base_embeddings, 'model', 'unknown')
    
    def embed_documents(self, texts):
        return [self.cache.get_embedding(
            text, self.model_name, 
            lambda t: self.base_embeddings.embed_query(t)
        ) for text in texts]
    
    def embed_query(self, text):
        return self.cache.get_embedding(
            text, self.model_name,
            lambda t: self.base_embeddings.embed_query(t)
        )

# Wrap your embeddings with caching
cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
```

This comprehensive guide covers all aspects of working with indexes and document processing in LangChain, from basic document loading to advanced RAG implementations and performance optimization techniques.