import { DocLayout } from '@/components/docs/DocLayout';
import { CodeBlock } from '@/components/CodeBlock';
import { Callout } from '@/components/docs/DocHeader';

export default function IndexesDocumentation() {
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
    <DocLayout 
      title="LangChain Indexes" 
      description="Learn how to work with document indexes, embeddings, and vector stores in LangChain for efficient information retrieval."
      toc={toc}
    >
      <section id="overview" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Overview</h2>
        <p className="mb-4">
          Indexes in LangChain provide a way to structure and organize documents for efficient 
          retrieval and search. They are a crucial component for building retrieval-augmented 
          generation (RAG) applications and other document-based workflows.
        </p>
        
        <Callout type="tip">
          <p>
            The typical workflow for working with indexes involves: loading documents, splitting them 
            into chunks, generating embeddings, storing them in a vector database, and then retrieving 
            relevant documents based on similarity search.
          </p>
        </Callout>
      </section>

      <section id="document-loaders" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Document Loaders</h2>
        <p className="mb-4">
          Document loaders help you load data from various sources into Document objects that LangChain 
          can process. LangChain provides loaders for many different file formats and data sources.
        </p>
        
        <CodeBlock 
          code={documentLoaderExample} 
          language="python" 
          title="Loading Documents from Different Sources"
        />
      </section>

      <section id="text-splitters" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Text Splitters</h2>
        <p className="mb-4">
          Since language models have a limited context window, you'll often need to split your documents 
          into smaller chunks. Text splitters help you do this in a way that preserves the semantic 
          meaning of the text.
        </p>
        
        <CodeBlock 
          code={textSplitterExample} 
          language="python" 
          title="Splitting Documents into Chunks"
        />
      </section>

      <section id="vector-stores" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Vector Stores</h2>
        <p className="mb-4">
          Vector stores are databases that store document embeddings and allow for efficient similarity 
          search. LangChain provides a unified interface to work with various vector store implementations.
        </p>
        
        <CodeBlock 
          code={vectorStoreExample} 
          language="python" 
          title="Working with Vector Stores"
        />
        
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2">Supported Vector Stores</h3>
          <ul className="list-disc pl-6 space-y-1">
            <li><strong>FAISS</strong>: Facebook AI Similarity Search, efficient for small to medium datasets</li>
            <li><strong>Pinecone</strong>: Managed vector database with high performance at scale</li>
            <li><strong>Chroma</strong>: Open-source embedding database</li>
            <li><strong>Weaviate</strong>: Open-source vector search engine</li>
            <li><strong>Milvus</strong>: Vector database for scalable similarity search</li>
            <li><strong>Qdrant</strong>: Vector similarity search engine with extended filtering support</li>
          </ul>
        </div>
      </section>

      <section id="retrievers" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Retrievers</h2>
        <p className="mb-4">
          Retrievers are interfaces that return documents given an unstructured query. They are a key 
          component in retrieval-augmented generation (RAG) applications.
        </p>
        
        <CodeBlock 
          code={retrieverExample} 
          language="python" 
          title="Using Retrievers for Document Search"
        />
        
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Retrieval Strategies</h3>
          <ul className="list-disc pl-6 space-y-1">
            <li><strong>Similarity Search</strong>: Returns documents most similar to the query</li>
            <li><strong>MMR (Maximal Marginal Relevance)</strong>: Balances similarity and diversity</li>
            <li><strong>Similarity Score Threshold</strong>: Only returns documents above a similarity threshold</li>
            <li><strong>Self-Query</strong>: Handles metadata filtering</li>
            <li><strong>Contextual Compression</strong>: Reduces document size before returning</li>
          </ul>
        </div>
      </section>

      <section id="embeddings" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Embeddings</h2>
        <p className="mb-4">
          Embeddings are numerical representations of text that capture semantic meaning. They are used 
          to convert text into vectors that can be compared for similarity.
        </p>
        
        <CodeBlock 
          code={embeddingExample} 
          language="python" 
          title="Working with Embeddings"
        />
        
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Supported Embedding Models</h3>
          <ul className="list-disc pl-6 space-y-1">
            <li><strong>OpenAI</strong>: text-embedding-ada-002 and other models</li>
            <li><strong>HuggingFace</strong>: All sentence-transformers models</li>
            <li><strong>Cohere</strong>: Cohere's embedding models</li>
            <li><strong>Google</strong>: Google's Universal Sentence Encoder and other models</li>
            <li><strong>Custom</strong>: Bring your own embedding model</li>
          </ul>
        </div>
      </section>

      <section id="best-practices" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Best Practices</h2>
        
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold mb-2">1. Choose the Right Chunk Size</h3>
            <p className="text-muted-foreground">
              The optimal chunk size depends on your use case. Smaller chunks (200-500 tokens) work well for 
              question answering, while larger chunks (1000-2000 tokens) are better for summarization.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">2. Use Appropriate Overlap</h3>
            <p className="text-muted-foreground">
              When splitting documents, include some overlap between chunks (10-20% of chunk size) to 
              prevent losing context at chunk boundaries.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">3. Select the Right Embedding Model</h3>
            <p className="text-muted-foreground">
              Choose an embedding model that matches your domain and requirements. Consider factors like 
              model size, performance, and language support.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">4. Implement Hybrid Search</h3>
            <p className="text-muted-foreground">
              Combine semantic search with keyword-based search for better results. This is particularly 
              useful for queries that require both semantic understanding and specific keyword matching.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">5. Monitor and Evaluate</h3>
            <p className="text-muted-foreground">
              Track the quality of your retrievals and make adjustments as needed. Consider metrics like 
              precision, recall, and user feedback to evaluate performance.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">6. Cache Embeddings</h3>
            <p className="text-muted-foreground">
              Store computed embeddings to avoid redundant computations, especially for static documents. 
              This can significantly improve performance and reduce costs.
            </p>
          </div>
        </div>
      </section>
    </DocLayout>
  );
}
