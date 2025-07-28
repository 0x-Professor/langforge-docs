# 🚀 Advanced Usage Examples

**Master Complex LangChain Patterns and Production-Ready Implementations**

---

## 🎯 Overview

This section covers advanced LangChain patterns for building production-ready applications. These examples demonstrate sophisticated integrations, custom implementations, and best practices for complex use cases.

### 🏗️ **Advanced Topics Covered**

- **🤖 Modern Agents** - ReAct, OpenAI Functions, and Custom Agents
- **🛠️ Custom Tools** - Building sophisticated tool integrations
- **📚 RAG Systems** - Retrieval-Augmented Generation pipelines
- **🧠 Advanced Memory** - Context management and conversation strategies
- **🔄 Async Processing** - High-performance async implementations
- **📊 Production Patterns** - Error handling, monitoring, and scaling

---

## 🤖 Modern Agents

### **OpenAI Functions Agent (Recommended)**

Modern agent using OpenAI's function calling:

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import asyncio

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Mock weather data - integrate with real weather API
    weather_data = {
        "New York": "Sunny, 75°F",
        "London": "Cloudy, 60°F", 
        "Tokyo": "Rainy, 68°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")

@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 18.0) -> str:
    """Calculate tip amount for a restaurant bill."""
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    return f"Tip: ${tip:.2f}, Total: ${total:.2f}"

async def advanced_agent_example():
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Create tools
    search = DuckDuckGoSearchRun()
    tools = [get_current_weather, calculate_tip, search]
    
    # Create prompt with system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant with access to various tools.
        Always be precise with calculations and provide context for your responses.
        When searching, summarize the key findings clearly."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        max_iterations=3,
        return_intermediate_steps=True
    )
    
    # Test complex queries
    queries = [
        "What's the weather in Tokyo and calculate a 20% tip on a $87.50 bill?",
        "Search for the latest developments in AI and tell me about them",
        "What's the weather in London and help me calculate tip for a $125 dinner"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        try:
            result = await agent_executor.ainvoke({"input": query})
            print(f"✅ Result: {result['output']}")
            print(f"🔧 Steps taken: {len(result.get('intermediate_steps', []))}")
        except Exception as e:
            print(f"❌ Error: {e}")

# Run the example
asyncio.run(advanced_agent_example())
```

### **Custom ReAct Agent with Enhanced Reasoning**

Build a custom agent with sophisticated reasoning patterns:

```python
from langchain.agents import Tool, AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any
import json

class EnhancedReActAgent:
    def __init__(self, tools: List[Tool], llm=None):
        self.tools = tools
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        self.setup_agent()
    
    def setup_agent(self):
        # Enhanced ReAct prompt with better reasoning structure
        template = """You are a problem-solving assistant that follows a structured thinking process.

Available tools:
{tools}

Use this EXACT format for your responses:

Question: {input}

Thought: I need to break this problem down and determine what tools I need to use.
Action: [tool_name]
Action Input: [specific input for the tool]
Observation: [tool result will appear here]
Thought: Based on this observation, I need to [next step reasoning]
Action: [next tool if needed]
Action Input: [input for next tool]
Observation: [next tool result]
Thought: Now I have enough information to provide a complete answer.
Final Answer: [comprehensive answer combining all observations]

Begin!

Question: {input}
{agent_scratchpad}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
            }
        )
        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5
        )
    
    def run(self, query: str) -> str:
        return self.agent_executor.run(query)

# Example usage
def enhanced_react_example():
    # Create sophisticated tools
    tools = [
        Tool(
            name="Research",
            func=lambda q: f"Research results for '{q}': [Mock research data about {q}]",
            description="Research information on any topic"
        ),
        Tool(
            name="Calculate",
            func=lambda expr: str(eval(expr)) if expr.replace('+', '').replace('-', '').replace('*', '').replace('/', '').replace('.', '').replace(' ', '').isdigit() else "Invalid expression",
            description="Perform mathematical calculations"
        ),
        Tool(
            name="Analyze",
            func=lambda data: f"Analysis of '{data}': Key insights and patterns identified.",
            description="Analyze data and extract insights"
        )
    ]
    
    agent = EnhancedReActAgent(tools)
    
    # Test complex reasoning
    query = "Research market trends for electric vehicles, calculate the growth rate if sales increased from 2.1 million to 3.8 million, and analyze what this means for the industry"
    
    result = agent.run(query)
    print(f"🧠 Enhanced ReAct Result: {result}")

enhanced_react_example()
```

---

## 🛠️ Advanced Custom Tools

### **Database Integration Tool**

Create tools that interact with databases:

```python
from langchain.tools import BaseTool
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel, Field
import sqlite3
import pandas as pd
from contextlib import contextmanager

class DatabaseQueryInput(BaseModel):
    query: str = Field(..., description="SQL query to execute")
    database_path: str = Field(default="./example.db", description="Path to SQLite database")

class DatabaseTool(BaseTool):
    name = "database_query"
    description = """Execute SQL queries on a database. 
    Useful for retrieving, analyzing, or manipulating structured data.
    Always use proper SQL syntax and be careful with data modifications."""
    args_schema: Type[BaseModel] = DatabaseQueryInput
    
    @contextmanager
    def get_db_connection(self, db_path: str):
        """Safe database connection context manager."""
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def _run(self, query: str, database_path: str = "./example.db") -> str:
        """Execute database query safely."""
        try:
            # Validate query (basic safety check)
            dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER']
            if any(keyword in query.upper() for keyword in dangerous_keywords):
                return "Error: Potentially dangerous query detected. Only SELECT queries are allowed."
            
            with self.get_db_connection(database_path) as conn:
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    return "Query executed successfully but returned no results."
                
                # Format results nicely
                result = f"Query Results ({len(df)} rows):\n"
                result += df.to_string(index=False, max_rows=10)
                
                if len(df) > 10:
                    result += f"\n... (showing first 10 of {len(df)} rows)"
                
                return result
                
        except Exception as e:
            return f"Database error: {str(e)}"
    
    async def _arun(self, query: str, database_path: str = "./example.db") -> str:
        """Async version - delegate to sync for simplicity."""
        return self._run(query, database_path)

# Setup example database
def setup_example_database():
    conn = sqlite3.connect("./example.db")
    
    # Create sample tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            product TEXT,
            quantity INTEGER,
            price REAL,
            date TEXT,
            region TEXT
        )
    """)
    
    # Insert sample data
    sample_data = [
        (1, "Widget A", 100, 19.99, "2024-01-15", "North"),
        (2, "Widget B", 75, 24.99, "2024-01-16", "South"),
        (3, "Widget A", 120, 19.99, "2024-01-17", "East"),
        (4, "Widget C", 90, 29.99, "2024-01-18", "West"),
        (5, "Widget B", 110, 24.99, "2024-01-19", "North")
    ]
    
    conn.executemany(
        "INSERT OR REPLACE INTO sales (id, product, quantity, price, date, region) VALUES (?, ?, ?, ?, ?, ?)",
        sample_data
    )
    
    conn.commit()
    conn.close()

# Example usage
def database_tool_example():
    setup_example_database()
    
    db_tool = DatabaseTool()
    
    # Test queries
    queries = [
        "SELECT * FROM sales",
        "SELECT product, SUM(quantity) as total_quantity, SUM(quantity * price) as revenue FROM sales GROUP BY product",
        "SELECT region, AVG(price) as avg_price FROM sales GROUP BY region"
    ]
    
    for query in queries:
        print(f"\n📊 Query: {query}")
        result = db_tool.run({"query": query})
        print(f"Results:\n{result}")

database_tool_example()
```

### **Web Scraping Tool with Error Handling**

Advanced web scraping with robust error handling:

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
import logging

class WebScrapingInput(BaseModel):
    url: str = Field(..., description="URL to scrape")
    selector: Optional[str] = Field(None, description="CSS selector for specific content")
    max_length: int = Field(2000, description="Maximum length of returned content")

class WebScrapingTool(BaseTool):
    name = "web_scraper"
    description = """Scrape content from web pages safely. 
    Can extract specific elements using CSS selectors.
    Handles errors gracefully and respects rate limits."""
    args_schema: Type[BaseModel] = WebScrapingInput
    
    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; LangChain WebScraper/1.0)'
        })
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _run(self, url: str, selector: Optional[str] = None, max_length: int = 2000) -> str:
        """Scrape web content safely."""
        try:
            # Validate URL
            if not self._is_valid_url(url):
                return f"Error: Invalid URL format: {url}"
            
            # Rate limiting
            time.sleep(1)
            
            # Make request with timeout
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract specific content if selector provided
            if selector:
                elements = soup.select(selector)
                if not elements:
                    return f"No elements found for selector: {selector}"
                content = "\n".join([elem.get_text(strip=True) for elem in elements])
            else:
                # Get main content
                content = soup.get_text(separator='\n', strip=True)
            
            # Clean and limit content
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            clean_content = '\n'.join(lines)
            
            if len(clean_content) > max_length:
                clean_content = clean_content[:max_length] + "... (content truncated)"
            
            return f"Content from {url}:\n{clean_content}"
            
        except requests.exceptions.RequestException as e:
            return f"Network error accessing {url}: {str(e)}"
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"
    
    async def _arun(self, url: str, selector: Optional[str] = None, max_length: int = 2000) -> str:
        """Async version."""
        return self._run(url, selector, max_length)

# Example usage
def web_scraping_example():
    scraper = WebScrapingTool()
    
    # Test different scraping scenarios
    examples = [
        {
            "url": "https://httpbin.org/html",
            "selector": None,
            "description": "Basic HTML scraping"
        },
        {
            "url": "https://httpbin.org/html", 
            "selector": "h1",
            "description": "Specific element selection"
        }
    ]
    
    for example in examples:
        print(f"\n🌐 {example['description']}")
        print(f"URL: {example['url']}")
        if example['selector']:
            print(f"Selector: {example['selector']}")
        
        result = scraper.run({
            "url": example['url'],
            "selector": example['selector']
        })
        print(f"Result:\n{result}")

web_scraping_example()
```

---

## 📚 Production RAG Systems

### **Advanced RAG with Multiple Retrievers**

Sophisticated RAG implementation with multiple retrieval strategies:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from typing import List, Dict, Any
import asyncio

class AdvancedRAGSystem:
    def __init__(self, documents: List[str]):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.setup_retrievers(documents)
        self.setup_chain()
    
    def setup_retrievers(self, documents: List[str]):
        """Setup multiple retrieval strategies."""
        # Process documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        docs = [Document(page_content=doc) for doc in documents]
        splits = text_splitter.split_documents(docs)
        
        # Vector retriever
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        vector_retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": 4, "fetch_k": 8}
        )
        
        # BM25 retriever (keyword-based)
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 4
        
        # Ensemble retriever combining both
        self.retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # Favor vector search slightly
        )
    
    def setup_chain(self):
        """Setup the RAG chain with advanced prompting."""
        template = """You are an expert assistant that answers questions based on provided context.

Context Information:
{context}

Instructions:
1. Answer the question using ONLY the information provided in the context
2. If the context doesn't contain enough information, clearly state this
3. Provide specific quotes or references when possible
4. Structure your answer clearly with key points
5. If there are conflicting information in the context, acknowledge this

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the chain with parallel processing
        self.chain = (
            RunnableParallel({
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            })
            | prompt
            | self.llm
        )
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents with metadata."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            formatted.append(f"[Source {i}]:\n{content}")
        return "\n\n".join(formatted)
    
    async def aquery(self, question: str) -> Dict[str, Any]:
        """Async query with detailed response."""
        # Get retrieved documents for transparency
        retrieved_docs = await self.retriever.ainvoke(question)
        
        # Generate answer
        response = await self.chain.ainvoke(question)
        
        return {
            "answer": response.content,
            "source_count": len(retrieved_docs),
            "sources": [doc.page_content[:200] + "..." for doc in retrieved_docs]
        }
    
    def query(self, question: str) -> Dict[str, Any]:
        """Synchronous query."""
        retrieved_docs = self.retriever.invoke(question)
        response = self.chain.invoke(question)
        
        return {
            "answer": response.content,
            "source_count": len(retrieved_docs),
            "sources": [doc.page_content[:200] + "..." for doc in retrieved_docs]
        }

# Example usage
async def advanced_rag_example():
    # Sample knowledge base
    documents = [
        """
        LangChain is a framework for developing applications powered by language models.
        It provides modular components that can be used to build complex applications.
        Key components include LLMs, prompts, chains, agents, and memory systems.
        LangChain supports multiple LLM providers including OpenAI, Anthropic, and others.
        """,
        """
        Vector databases are specialized databases designed to store and search high-dimensional vectors.
        They use algorithms like approximate nearest neighbor (ANN) to find similar vectors quickly.
        Popular vector databases include Pinecone, Weaviate, Qdrant, and FAISS.
        Vector databases are essential for semantic search and retrieval-augmented generation (RAG).
        """,
        """
        Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge.
        RAG systems retrieve relevant information from a knowledge base and use it to generate more accurate responses.
        The process involves embedding documents, storing them in a vector database, and retrieving relevant chunks.
        RAG helps reduce hallucinations and keeps LLM responses grounded in factual information.
        """,
        """
        Prompt engineering is the practice of designing effective prompts for language models.
        Good prompts are clear, specific, and provide necessary context and constraints.
        Techniques include few-shot learning, chain-of-thought prompting, and role-based prompts.
        Prompt templates in LangChain help standardize and reuse effective prompt patterns.
        """
    ]
    
    # Create RAG system
    rag_system = AdvancedRAGSystem(documents)
    
    # Test questions
    questions = [
        "What is LangChain and what are its key components?",
        "How do vector databases work and why are they important for RAG?",
        "What are some best practices for prompt engineering?",
        "How does RAG help with LLM hallucinations?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        result = await rag_system.aquery(question)
        print(f"✅ Answer: {result['answer']}")
        print(f"📚 Sources used: {result['source_count']}")

# Run the example
asyncio.run(advanced_rag_example())
```

---

## 🧠 Advanced Memory Patterns

### **Hierarchical Memory System**

Sophisticated memory management for long conversations:

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict, List, Any
import json
from datetime import datetime

class HierarchicalMemory:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Different memory layers
        self.working_memory = ConversationBufferMemory(
            memory_key="recent_history",
            return_messages=True,
            k=10  # Keep last 10 exchanges
        )
        
        self.summary_memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="conversation_summary"
        )
        
        # Long-term facts and preferences
        self.factual_memory: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {}
    
    def add_conversation(self, human_input: str, ai_output: str):
        """Add conversation to all memory layers."""
        # Add to working memory
        self.working_memory.save_context(
            {"input": human_input},
            {"output": ai_output}
        )
        
        # Add to summary memory
        self.summary_memory.save_context(
            {"input": human_input},
            {"output": ai_output}
        )
        
        # Extract and store facts/preferences
        self._extract_facts(human_input, ai_output)
    
    def _extract_facts(self, human_input: str, ai_output: str):
        """Extract facts and preferences from conversation."""
        extraction_prompt = f"""
        Extract any personal facts, preferences, or important information from this conversation:
        
        Human: {human_input}
        AI: {ai_output}
        
        Return a JSON object with:
        - "facts": List of factual statements about the user
        - "preferences": List of user preferences or opinions
        - "context": Any contextual information that might be useful later
        
        If no significant information is found, return empty lists.
        """
        
        try:
            response = self.llm.invoke(extraction_prompt)
            extracted = json.loads(response.content)
            
            # Store facts
            timestamp = datetime.now().isoformat()
            for fact in extracted.get("facts", []):
                self.factual_memory[fact] = timestamp
            
            # Store preferences
            for pref in extracted.get("preferences", []):
                self.user_preferences[pref] = timestamp
                
        except (json.JSONDecodeError, Exception):
            # Gracefully handle extraction failures
            pass
    
    def get_memory_context(self) -> Dict[str, Any]:
        """Get comprehensive memory context."""
        return {
            "recent_conversations": self.working_memory.load_memory_variables({})["recent_history"],
            "conversation_summary": self.summary_memory.load_memory_variables({})["conversation_summary"],
            "known_facts": list(self.factual_memory.keys())[-5:],  # Recent facts
            "user_preferences": list(self.user_preferences.keys())[-3:]  # Recent preferences
        }
    
    def search_memory(self, query: str) -> str:
        """Search through all memory layers."""
        # Simple keyword matching - could be enhanced with embeddings
        query_lower = query.lower()
        relevant_info = []
        
        # Search facts
        for fact in self.factual_memory:
            if any(word in fact.lower() for word in query_lower.split()):
                relevant_info.append(f"Fact: {fact}")
        
        # Search preferences
        for pref in self.user_preferences:
            if any(word in pref.lower() for word in query_lower.split()):
                relevant_info.append(f"Preference: {pref}")
        
        return "\n".join(relevant_info) if relevant_info else "No relevant information found in memory."

# Example usage
def hierarchical_memory_example():
    memory_system = HierarchicalMemory()
    
    # Simulate a long conversation
    conversations = [
        ("Hi, I'm Sarah and I'm a data scientist", "Hello Sarah! Nice to meet you. Data science is a fascinating field. What kind of projects do you work on?"),
        ("I mostly work on machine learning models for healthcare", "That sounds very impactful! Healthcare applications of ML can really make a difference. Do you focus on any particular area of healthcare?"),
        ("Yes, I specialize in medical imaging analysis", "Medical imaging is crucial for diagnosis. Are you working with specific types of imaging like MRI, CT scans, or X-rays?"),
        ("Mainly chest X-rays for pneumonia detection", "That's excellent work! Early detection of pneumonia can save lives. Have you deployed any models in clinical settings?"),
        ("I prefer Python over R for my work", "Python is indeed very popular in the data science community, especially with libraries like scikit-learn and TensorFlow."),
        ("I'm planning to learn more about LangChain", "Great choice! LangChain is excellent for building AI applications. Given your healthcare background, you might find RAG systems particularly useful."),
        ("What was my name again?", "Your name is Sarah."),
        ("What do I do for work?", "You're a data scientist who specializes in medical imaging analysis, particularly working with chest X-rays for pneumonia detection."),
        ("What programming language do I prefer?", "You prefer Python over R for your data science work.")
    ]
    
    # Process conversations
    for human_input, ai_output in conversations:
        memory_system.add_conversation(human_input, ai_output)
        print(f"👤 Human: {human_input}")
        print(f"🤖 AI: {ai_output}\n")
    
    # Test memory retrieval
    print("🧠 Memory System Test:")
    context = memory_system.get_memory_context()
    print(f"📝 Conversation Summary: {context['conversation_summary']}")
    print(f"📊 Known Facts: {context['known_facts']}")
    print(f"❤️ User Preferences: {context['user_preferences']}")
    
    # Test memory search
    print(f"\n🔍 Search 'healthcare': {memory_system.search_memory('healthcare')}")
    print(f"🔍 Search 'Python': {memory_system.search_memory('Python')}")

hierarchical_memory_example()
```

---

## 🔄 Async Processing Patterns

### **High-Performance Async Chain Processing**

Efficient async processing for production applications:

```python
import asyncio
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
import time
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    input_data: str
    result: str
    processing_time: float
    success: bool
    error: Optional[str] = None

class AsyncBatchProcessor:
    def __init__(self, max_concurrent: int = 5):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single(self, prompt_template: str, input_data: str) -> ProcessingResult:
        """Process a single item with error handling and timing."""
        start_time = time.time()
        
        async with self.semaphore:  # Limit concurrent requests
            try:
                prompt = ChatPromptTemplate.from_template(prompt_template)
                chain = prompt | self.llm
                
                result = await chain.ainvoke({"input": input_data})
                processing_time = time.time() - start_time
                
                return ProcessingResult(
                    input_data=input_data,
                    result=result.content,
                    processing_time=processing_time,
                    success=True
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                return ProcessingResult(
                    input_data=input_data,
                    result="",
                    processing_time=processing_time,
                    success=False,
                    error=str(e)
                )
    
    async def batch_process(self, prompt_template: str, inputs: List[str]) -> List[ProcessingResult]:
        """Process multiple inputs concurrently."""
        tasks = [
            self.process_single(prompt_template, input_data)
            for input_data in inputs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that weren't caught
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    input_data=inputs[i],
                    result="",
                    processing_time=0,
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def print_batch_summary(self, results: List[ProcessingResult]):
        """Print a summary of batch processing results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_time = sum(r.processing_time for r in results)
        avg_time = total_time / len(results) if results else 0
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"✅ Successful: {len(successful)}/{len(results)}")
        print(f"❌ Failed: {len(failed)}/{len(results)}")
        print(f"⏱️ Average processing time: {avg_time:.2f}s")
        print(f"🚀 Total time: {total_time:.2f}s")
        
        if failed:
            print(f"\n❌ Failed items:")
            for result in failed:
                print(f"  - {result.input_data[:50]}... Error: {result.error}")

# Example usage
async def async_processing_example():
    processor = AsyncBatchProcessor(max_concurrent=3)
    
    # Example 1: Sentiment analysis
    sentiment_prompt = """
    Analyze the sentiment of the following text and classify it as positive, negative, or neutral.
    Provide a brief explanation for your classification.
    
    Text: {input}
    
    Sentiment analysis:
    """
    
    sample_texts = [
        "I absolutely love this new restaurant! The food was amazing and the service was excellent.",
        "The movie was terrible. I wasted my money and two hours of my life.",
        "The weather is cloudy today. It might rain later.",
        "This book changed my perspective on life in such a positive way.",
        "The customer service was disappointing and unhelpful.",
        "The presentation was informative and well-structured.",
        "I'm feeling anxious about the upcoming exam.",
        "The concert last night was absolutely incredible!"
    ]
    
    print("🎭 Processing sentiment analysis batch...")
    sentiment_results = await processor.batch_process(sentiment_prompt, sample_texts)
    processor.print_batch_summary(sentiment_results)
    
    # Show some results
    print("\n📝 Sample Results:")
    for i, result in enumerate(sentiment_results[:3]):
        if result.success:
            print(f"\n{i+1}. Input: {result.input_data[:60]}...")
            print(f"   Output: {result.result[:100]}...")
            print(f"   Time: {result.processing_time:.2f}s")
    
    # Example 2: Content generation
    content_prompt = """
    Create a compelling social media post about {input}.
    Make it engaging, include relevant hashtags, and keep it under 280 characters.
    
    Social media post:
    """
    
    topics = [
        "sustainable living tips",
        "morning coffee routine",
        "weekend hiking adventure",
        "new AI technology trends",
        "healthy meal prep ideas"
    ]
    
    print("\n\n📱 Processing social media content batch...")
    content_results = await processor.batch_process(content_prompt, topics)
    processor.print_batch_summary(content_results)
    
    # Show results
    print("\n📝 Generated Social Media Posts:")
    for i, result in enumerate(content_results):
        if result.success:
            print(f"\n{i+1}. Topic: {result.input_data}")
            print(f"   Post: {result.result}")

# Run the async example
asyncio.run(async_processing_example())
```

---

## 📊 Production Monitoring and Error Handling

### **Comprehensive Monitoring System**

Production-ready monitoring and error handling:

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from contextlib import contextmanager
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
import time

@dataclass
class MetricData:
    timestamp: datetime
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorData:
    timestamp: datetime
    error_type: str
    error_message: str
    context: Dict[str, Any] = field(default_factory=dict)

class ProductionMonitor(BaseCallbackHandler):
    def __init__(self):
        self.metrics: List[MetricData] = []
        self.errors: List[ErrorData] = []
        self.active_requests: Dict[str, float] = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Track LLM request start."""
        request_id = kwargs.get('run_id', str(time.time()))
        self.active_requests[str(request_id)] = time.time()
        
        self.metrics.append(MetricData(
            timestamp=datetime.now(),
            metric_name="llm_request_started",
            value=1,
            metadata={"request_id": str(request_id), "prompt_count": len(prompts)}
        ))
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Track LLM request completion."""
        request_id = kwargs.get('run_id', str(time.time()))
        start_time = self.active_requests.pop(str(request_id), time.time())
        duration = time.time() - start_time
        
        # Track metrics
        self.metrics.extend([
            MetricData(
                timestamp=datetime.now(),
                metric_name="llm_request_completed",
                value=1,
                metadata={"request_id": str(request_id), "duration": duration}
            ),
            MetricData(
                timestamp=datetime.now(),
                metric_name="llm_response_time",
                value=duration,
                metadata={"request_id": str(request_id)}
            )
        ])
        
        # Track token usage if available
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            if token_usage:
                self.metrics.append(MetricData(
                    timestamp=datetime.now(),
                    metric_name="token_usage",
                    value=token_usage.get('total_tokens', 0),
                    metadata=token_usage
                ))
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Track LLM errors."""
        request_id = kwargs.get('run_id', str(time.time()))
        
        self.errors.append(ErrorData(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            context={"request_id": str(request_id)}
        ))
        
        self.metrics.append(MetricData(
            timestamp=datetime.now(),
            metric_name="llm_error",
            value=1,
            metadata={"error_type": type(error).__name__}
        ))
        
        self.logger.error(f"LLM Error: {error}")
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        recent_errors = [e for e in self.errors if e.timestamp > cutoff_time]
        
        # Calculate aggregations
        response_times = [m.value for m in recent_metrics if m.metric_name == "llm_response_time"]
        total_requests = len([m for m in recent_metrics if m.metric_name == "llm_request_completed"])
        total_errors = len(recent_errors)
        
        summary = {
            "time_period_hours": hours,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_requests, 1),
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0
        }
        
        # Error breakdown
        error_types = {}
        for error in recent_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        summary["error_breakdown"] = error_types
        
        return summary
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operations."""
        start_time = time.time()
        
        self.metrics.append(MetricData(
            timestamp=datetime.now(),
            metric_name=f"{operation_name}_started",
            value=1
        ))
        
        try:
            yield
            duration = time.time() - start_time
            self.metrics.extend([
                MetricData(
                    timestamp=datetime.now(),
                    metric_name=f"{operation_name}_completed",
                    value=1,
                    metadata={"duration": duration}
                ),
                MetricData(
                    timestamp=datetime.now(),
                    metric_name=f"{operation_name}_duration",
                    value=duration
                )
            ])
        except Exception as e:
            duration = time.time() - start_time
            self.errors.append(ErrorData(
                timestamp=datetime.now(),
                error_type=type(e).__name__,
                error_message=str(e),
                context={"operation": operation_name, "duration": duration}
            ))
            self.metrics.append(MetricData(
                timestamp=datetime.now(),
                metric_name=f"{operation_name}_error",
                value=1,
                metadata={"error_type": type(e).__name__}
            ))
            raise

# Example usage
def production_monitoring_example():
    monitor = ProductionMonitor()
    
    # Simulate some operations
    operations = ["document_processing", "embedding_generation", "vector_search"]
    
    for operation in operations:
        try:
            with monitor.monitor_operation(operation):
                # Simulate operation
                time.sleep(0.1)
                if operation == "vector_search":
                    # Simulate occasional error
                    import random
                    if random.random() < 0.3:
                        raise Exception("Simulated vector search timeout")
        except Exception as e:
            print(f"❌ Operation {operation} failed: {e}")
    
    # Get metrics summary
    summary = monitor.get_metrics_summary(hours=1)
    
    print("📊 Production Metrics Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Print recent errors
    if monitor.errors:
        print("\n❌ Recent Errors:")
        for error in monitor.errors[-5:]:  # Last 5 errors
            print(f"  {error.timestamp}: {error.error_type} - {error.error_message}")

production_monitoring_example()
```

---

This comprehensive advanced usage guide covers production-ready patterns that you can use to build robust LangChain applications. Each example includes proper error handling, async support, and monitoring capabilities essential for production deployments.

## 🔗 Related Documentation

- [Basic Usage Examples](../basic-usage/) - Start here if you're new to LangChain
- [Best Practices Guide](../../guides/best-practices/) - Production deployment guidelines
- [Troubleshooting](../../troubleshooting.md) - Common issues and solutions
