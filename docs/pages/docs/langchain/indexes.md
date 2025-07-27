# indexes

function IndexesDocumentation() {
  const toc = [
    { id: 'overview', title: 'Overview', level: 2 },
    { id: 'document-loaders', title: 'Document Loaders', level: 2 },
    { id: 'text-splitters', title: 'Text Splitters', level: 2 },
    { id: 'vector-stores', title: 'Vector Stores', level: 2 },
    { id: 'retrievers', title: 'Retrievers', level: 2 },
    { id: 'embeddings', title: 'Embeddings', level: 2 },
    { id: 'best-practices', title: 'Best Practices', level: 2 },
  ];

  const documentLoaderExample = `from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader
)

# Load a text file
loader = TextLoader("document.txt")
documents = loader.load()

# Load a PDF file
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Load web pages
loader = WebBaseLoader(["https://example.com"])
documents = loader.load()

# Load CSV data
loader = CSVLoader("data.csv")
documents = loader.load()`;

  const textSplitterExample = `from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

# Recursive splitter (recommended for most cases)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Character splitter (simpler but less sophisticated)
character_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# Split documents
split_docs = recursive_splitter.split_documents(documents)`;

  const vectorStoreExample = `from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create a FAISS vector store from documents
vectorstore = FAISS.from_documents(
    documents=split_docs,
    embedding=embeddings
)

# Save the vector store locally
vectorstore.save_local("faiss_index")

# Load the vector store later
vectorstore = FAISS.load_local("faiss_index", embeddings)`;

  const retrieverExample = `# Create a retriever from the vector store
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr", "similarity_score_threshold"
    search_kwargs={"k": 5}    # number of documents to retrieve
)

# Retrieve relevant documents
query = "What is LangChain?"
docs = retriever.get_relevant_documents(query)

# Use in a chain
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain({"query": query})`;

  const embeddingExample = `from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Using OpenAI embeddings (requires API key)
openai_embeddings = OpenAIEmbeddings()

# Using HuggingFace embeddings (runs locally)
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embeddings for a text
text = "This is a sample text to embed."

# Get embedding vector
embedding_vector = openai_embeddings.embed_query(text)
print(f"Embedding dimension: {len(embedding_vector)}")
print(f"First 5 values: {embedding_vector[:5]}")`;

  const ragExample = `from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

# Create a retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Ask a question
query = "What is LangChain used for?"
result = qa_chain({"query": query})

print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']} (page {doc.metadata.get('page', 'N/A')})")`;

  return (
    
      
        
Overview

        
Indexes in LangChain provide a way to structure and organize documents for efficient 
          retrieval and search. They are a crucial component for building retrieval-augmented 
          generation (RAG) applications and other document-based workflows.

        
        
          
The typical workflow for working with indexes involves: loading documents, splitting them 
            into chunks, generating embeddings, storing them in a vector database, and then retrieving 
            relevant documents based on similarity search.

        

      
        
Document Loaders

        
Document loaders help you load data from various sources into Document objects that LangChain 
          can process. LangChain provides loaders for many different file formats and data sources.

        
        

      
        
Text Splitters

        
Since language models have a limited context window, you'll often need to split your documents 
          into smaller chunks. Text splitters help you do this in a way that preserves the semantic 
          meaning of the text.

        
        

      
        
Vector Stores

        
Vector stores are databases that store document embeddings and allow for efficient similarity 
          search. LangChain provides a unified interface to work with various vector store implementations.

        
        
        
        
          
Supported Vector Stores

          
            
FAISS
: Facebook AI Similarity Search, efficient for small to medium datasets
            
Pinecone
: Managed vector database with high performance at scale
            
Chroma
: Open-source embedding database
            
Weaviate
: Open-source vector search engine
            
Milvus
: Vector database for scalable similarity search
            
Qdrant
: Vector similarity search engine with extended filtering support

        

      
        
Retrievers

        
Retrievers are interfaces that return documents given an unstructured query. They are a key 
          component in retrieval-augmented generation (RAG) applications.

        
        
        
        
          
Retrieval Strategies

          
            
Similarity Search
: Returns documents most similar to the query
            
MMR (Maximal Marginal Relevance)
: Balances similarity and diversity
            
Similarity Score Threshold
: Only returns documents above a similarity threshold
            
Self-Query
: Handles metadata filtering
            
Contextual Compression
: Reduces document size before returning

        

      
        
Embeddings

        
Embeddings are numerical representations of text that capture semantic meaning. They are used 
          to convert text into vectors that can be compared for similarity.

        
        
        
        
          
Supported Embedding Models

          
            
OpenAI
: text-embedding-ada-002 and other models
            
HuggingFace
: All sentence-transformers models
            
Cohere
: Cohere's embedding models
            
Google
: Google's Universal Sentence Encoder and other models
            
Custom
: Bring your own embedding model

        

      
        
Best Practices

        
        
          
            
1. Choose the Right Chunk Size

            
The optimal chunk size depends on your use case. Smaller chunks (200-500 tokens) work well for 
              question answering, while larger chunks (1000-2000 tokens) are better for summarization.

          
          
          
            
2. Use Appropriate Overlap

            
When splitting documents, include some overlap between chunks (10-20% of chunk size) to 
              prevent losing context at chunk boundaries.

          
          
          
            
3. Select the Right Embedding Model

            
Choose an embedding model that matches your domain and requirements. Consider factors like 
              model size, performance, and language support.

          
          
          
            
4. Implement Hybrid Search

            
Combine semantic search with keyword-based search for better results. This is particularly 
              useful for queries that require both semantic understanding and specific keyword matching.

          
          
          
            
5. Monitor and Evaluate

            
Track the quality of your retrievals and make adjustments as needed. Consider metrics like 
              precision, recall, and user feedback to evaluate performance.

          
          
          
            
6. Cache Embeddings

            
Store computed embeddings to avoid redundant computations, especially for static documents. 
              This can significantly improve performance and reduce costs.

          

      

  );
}