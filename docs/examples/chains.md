# Chains

## Overview

Chains in LangChain are a way to combine multiple components together to create more complex applications. They allow you to create sequences of operations that can be executed in order, with the output of one operation becoming the input to the next.

> **Note:** The modern approach in LangChain uses LCEL (LangChain Expression Language) for better composability, streaming support, and async handling.

## Modern Chain Creation with LCEL

LangChain Expression Language (LCEL) is the recommended way to create chains in modern LangChain applications.

### Basic Chain with LCEL

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Define the prompt template
prompt = PromptTemplate.from_template(
    "What is a good name for a company that makes {product}?"
)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

# Create the chain using LCEL
chain = prompt | llm | StrOutputParser()

# Run the chain
result = chain.invoke({"product": "colorful socks"})
print(result)
```

```typescript
// TypeScript equivalent
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";

async function basicChainLCEL() {
  // Define the prompt template
  const prompt = PromptTemplate.fromTemplate(
    "What is a good name for a company that makes {product}?"
  );

  // Initialize the LLM
  const llm = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0.9,
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  // Create the chain using LCEL
  const chain = prompt.pipe(llm).pipe(new StringOutputParser());

  try {
    // Run the chain
    const result = await chain.invoke({ product: "colorful socks" });
    console.log(result);
    return result;
  } catch (error) {
    console.error("Chain execution failed:", error);
    throw error;
  }
}

basicChainLCEL().catch(console.error);
```

### Sequential Chains with LCEL

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Initialize components
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
output_parser = StrOutputParser()

# Define prompts
name_prompt = PromptTemplate.from_template(
    "You are a naming consultant. What is a good name for a {company_type} company?"
)

slogan_prompt = PromptTemplate.from_template(
    "Create a catchy slogan for this company: {company_name}"
)

# Create sequential chain
name_chain = name_prompt | llm | output_parser

# Chain the output of the first chain into the second
full_chain = (
    {"company_name": name_chain} 
    | slogan_prompt 
    | llm 
    | output_parser
)

# Run the chain
result = full_chain.invoke({"company_type": "eco-friendly clothing"})
print(result)
```

```typescript
// TypeScript equivalent
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";

async function sequentialChainsLCEL() {
  // Initialize components
  const llm = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0.9,
    openAIApiKey: process.env.OPENAI_API_KEY,
  });
  const outputParser = new StringOutputParser();

  // Define prompts
  const namePrompt = PromptTemplate.fromTemplate(
    "You are a naming consultant. What is a good name for a {company_type} company?"
  );

  const sloganPrompt = PromptTemplate.fromTemplate(
    "Create a catchy slogan for this company: {company_name}"
  );

  // Create sequential chain
  const nameChain = namePrompt.pipe(llm).pipe(outputParser);

  // Chain the output of the first chain into the second
  const fullChain = RunnableSequence.from([
    {
      company_name: nameChain,
      company_type: new RunnablePassthrough(),
    },
    sloganPrompt,
    llm,
    outputParser,
  ]);

  try {
    // Run the chain
    const result = await fullChain.invoke({ company_type: "eco-friendly clothing" });
    console.log(result);
    return result;
  } catch (error) {
    console.error("Sequential chain execution failed:", error);
    throw error;
  }
}

sequentialChainsLCEL().catch(console.error);
```

### Parallel Chains

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create parallel chains
name_chain = (
    PromptTemplate.from_template("Generate a company name for: {product}")
    | llm
    | StrOutputParser()
)

description_chain = (
    PromptTemplate.from_template("Write a brief description for a company that makes: {product}")
    | llm
    | StrOutputParser()
)

tagline_chain = (
    PromptTemplate.from_template("Create a tagline for a company that makes: {product}")
    | llm
    | StrOutputParser()
)

# Run chains in parallel
parallel_chain = RunnableParallel(
    name=name_chain,
    description=description_chain,
    tagline=tagline_chain
)

result = parallel_chain.invoke({"product": "sustainable water bottles"})
print(result)
# Output: {'name': '...', 'description': '...', 'tagline': '...'}
```

```typescript
// TypeScript equivalent
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableParallel } from "@langchain/core/runnables";

async function parallelChains() {
  const llm = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  // Create parallel chains
  const nameChain = PromptTemplate.fromTemplate(
    "Generate a company name for: {product}"
  ).pipe(llm).pipe(new StringOutputParser());

  const descriptionChain = PromptTemplate.fromTemplate(
    "Write a brief description for a company that makes: {product}"
  ).pipe(llm).pipe(new StringOutputParser());

  const taglineChain = PromptTemplate.fromTemplate(
    "Create a tagline for a company that makes: {product}"
  ).pipe(llm).pipe(new StringOutputParser());

  // Run chains in parallel
  const parallelChain = RunnableParallel.from({
    name: nameChain,
    description: descriptionChain,
    tagline: taglineChain,
  });

  try {
    const result = await parallelChain.invoke({ product: "sustainable water bottles" });
    console.log(result);
    // Output: { name: '...', description: '...', tagline: '...' }
    return result;
  } catch (error) {
    console.error("Parallel chain execution failed:", error);
    throw error;
  }
}

parallelChains().catch(console.error);
```

## Conditional Chains

Create chains that route to different paths based on input:

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

# Define specialized chains
physics_chain = (
    PromptTemplate.from_template(
        "You are a physics professor. Answer this physics question: {question}"
    )
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

math_chain = (
    PromptTemplate.from_template(
        "You are a mathematician. Answer this math question: {question}"
    )
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

general_chain = (
    PromptTemplate.from_template(
        "Answer this general question: {question}"
    )
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

# Create conditional routing
def is_physics_question(x):
    return any(word in x["question"].lower() for word in ["physics", "force", "energy", "quantum", "particle"])

def is_math_question(x):
    return any(word in x["question"].lower() for word in ["calculate", "equation", "solve", "math", "algebra"])

# Create the routing chain
router_chain = RunnableBranch(
    (is_physics_question, physics_chain),
    (is_math_question, math_chain),
    general_chain,  # default
)

# Test the router
physics_result = router_chain.invoke({"question": "What is the speed of light?"})
math_result = router_chain.invoke({"question": "What is 25 * 4?"})
general_result = router_chain.invoke({"question": "What is the capital of France?"})

print(f"Physics: {physics_result}")
print(f"Math: {math_result}")
print(f"General: {general_result}")
```

```typescript
// TypeScript equivalent
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableBranch } from "@langchain/core/runnables";

async function conditionalChains() {
  // Define specialized chains
  const physicsChain = PromptTemplate.fromTemplate(
    "You are a physics professor. Answer this physics question: {question}"
  ).pipe(new ChatOpenAI({ 
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
  })).pipe(new StringOutputParser());

  const mathChain = PromptTemplate.fromTemplate(
    "You are a mathematician. Answer this math question: {question}"
  ).pipe(new ChatOpenAI({ 
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
  })).pipe(new StringOutputParser());

  const generalChain = PromptTemplate.fromTemplate(
    "Answer this general question: {question}"
  ).pipe(new ChatOpenAI({ 
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
  })).pipe(new StringOutputParser());

  // Create conditional routing functions
  const isPhysicsQuestion = (x: { question: string }) => {
    const keywords = ["physics", "force", "energy", "quantum", "particle"];
    return keywords.some(word => x.question.toLowerCase().includes(word));
  };

  const isMathQuestion = (x: { question: string }) => {
    const keywords = ["calculate", "equation", "solve", "math", "algebra"];
    return keywords.some(word => x.question.toLowerCase().includes(word));
  };

  // Create the routing chain
  const routerChain = RunnableBranch.from([
    [isPhysicsQuestion, physicsChain],
    [isMathQuestion, mathChain],
    generalChain, // default
  ]);

  try {
    // Test the router
    const [physicsResult, mathResult, generalResult] = await Promise.all([
      routerChain.invoke({ question: "What is the speed of light?" }),
      routerChain.invoke({ question: "What is 25 * 4?" }),
      routerChain.invoke({ question: "What is the capital of France?" })
    ]);

    console.log(`Physics: ${physicsResult}`);
    console.log(`Math: ${mathResult}`);
    console.log(`General: ${generalResult}`);
    
    return { physicsResult, mathResult, generalResult };
  } catch (error) {
    console.error("Conditional chain execution failed:", error);
    throw error;
  }
}

conditionalChains().catch(console.error);
```

## Transform Chains

Apply transformations to data as it flows through the chain:

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def preprocess_text(inputs):
    """Clean and preprocess input text."""
    text = inputs["text"]
    # Remove extra whitespace and convert to lowercase
    cleaned = " ".join(text.strip().split()).lower()
    return {"text": cleaned}

def postprocess_response(response):
    """Format the response."""
    return {"formatted_response": f"✨ {response.strip()} ✨"}

# Create a chain with preprocessing and postprocessing
chain = (
    RunnableLambda(preprocess_text)
    | PromptTemplate.from_template("Improve this text: {text}")
    | ChatOpenAI(model="gpt-3.5-turbo")
    | RunnableLambda(postprocess_response)
)

result = chain.invoke({"text": "   HELLO    WORLD   how   are YOU?  "})
print(result)
```

```typescript
// TypeScript equivalent
import { RunnableLambda } from "@langchain/core/runnables";
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";

async function transformChains() {
  // Preprocessing function
  const preprocessText = (inputs: { text: string }) => {
    const text = inputs.text;
    // Remove extra whitespace and convert to lowercase
    const cleaned = text.trim().split(/\s+/).join(" ").toLowerCase();
    return { text: cleaned };
  };

  // Postprocessing function
  const postprocessResponse = (response: any) => {
    const content = typeof response === 'string' ? response : response.content;
    return { formatted_response: `✨ ${content.trim()} ✨` };
  };

  // Create a chain with preprocessing and postprocessing
  const chain = RunnableLambda.from(preprocessText)
    .pipe(PromptTemplate.fromTemplate("Improve this text: {text}"))
    .pipe(new ChatOpenAI({
      model: "gpt-3.5-turbo",
      openAIApiKey: process.env.OPENAI_API_KEY,
    }))
    .pipe(RunnableLambda.from(postprocessResponse));

  try {
    const result = await chain.invoke({ text: "   HELLO    WORLD   how   are YOU?  " });
    console.log(result);
    return result;
  } catch (error) {
    console.error("Transform chain execution failed:", error);
    throw error;
  }
}

transformChains().catch(console.error);
```

## Error Handling and Fallbacks

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def validate_input(inputs):
    """Validate inputs before processing."""
    if not inputs.get("question") or len(inputs["question"].strip()) < 3:
        raise ValueError("Question must be at least 3 characters long")
    return inputs

def fallback_response(error):
    """Provide fallback response on error."""
    return "I'm sorry, I couldn't process your request. Please try rephrasing your question."

# Create a robust chain with validation and fallbacks
main_chain = (
    RunnableLambda(validate_input)
    | PromptTemplate.from_template("Answer this question clearly: {question}")
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)

# Add fallback handling
robust_chain = main_chain.with_fallbacks([
    RunnableLambda(fallback_response)
])

# Test with valid input
try:
    result1 = robust_chain.invoke({"question": "What is artificial intelligence?"})
    print(f"Success: {result1}")
except Exception as e:
    print(f"Error: {e}")

# Test with invalid input
try:
    result2 = robust_chain.invoke({"question": "Hi"})
    print(f"Fallback: {result2}")
except Exception as e:
    print(f"Error: {e}")
```

```typescript
// TypeScript equivalent
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableLambda } from "@langchain/core/runnables";

async function errorHandlingChains() {
  // Validation function
  const validateInput = (inputs: { question?: string }) => {
    if (!inputs.question || inputs.question.trim().length < 3) {
      throw new Error("Question must be at least 3 characters long");
    }
    return inputs;
  };

  // Fallback function
  const fallbackResponse = () => {
    return "I'm sorry, I couldn't process your request. Please try rephrasing your question.";
  };

  // Create a robust chain with validation and fallbacks
  const mainChain = RunnableLambda.from(validateInput)
    .pipe(PromptTemplate.fromTemplate("Answer this question clearly: {question}"))
    .pipe(new ChatOpenAI({
      model: "gpt-3.5-turbo",
      openAIApiKey: process.env.OPENAI_API_KEY,
    }))
    .pipe(new StringOutputParser());

  // Add fallback handling
  const robustChain = mainChain.withFallbacks([
    RunnableLambda.from(fallbackResponse)
  ]);

  // Test with valid input
  try {
    const result1 = await robustChain.invoke({ question: "What is artificial intelligence?" });
    console.log(`Success: ${result1}`);
  } catch (error) {
    console.log(`Error: ${error}`);
  }

  // Test with invalid input
  try {
    const result2 = await robustChain.invoke({ question: "Hi" });
    console.log(`Fallback: ${result2}`);
  } catch (error) {
    console.log(`Error: ${error}`);
  }
}

errorHandlingChains().catch(console.error);
```

## Streaming Chains

Enable streaming for real-time output:

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Create a streaming chain
prompt = PromptTemplate.from_template("Write a story about {topic}")
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

chain = prompt | llm

# Stream the response
for chunk in chain.stream({"topic": "a robot learning to paint"}):
    print(chunk.content, end="", flush=True)
```

```typescript
// TypeScript equivalent
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";

async function streamingChains() {
  // Create a streaming chain
  const prompt = PromptTemplate.fromTemplate("Write a story about {topic}");
  const llm = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    streaming: true,
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const chain = prompt.pipe(llm);

  try {
    // Stream the response
    const stream = await chain.stream({ topic: "a robot learning to paint" });
    
    for await (const chunk of stream) {
      process.stdout.write(chunk.content || "");
    }
    console.log("\n"); // New line after streaming
  } catch (error) {
    console.error("Streaming chain execution failed:", error);
    throw error;
  }
}

streamingChains().catch(console.error);
```

## Async Chains

Handle multiple requests concurrently:

```python
import asyncio
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

async def async_chain_example():
    # Create async chain
    chain = (
        PromptTemplate.from_template("Explain {topic} in one sentence")
        | ChatOpenAI(model="gpt-3.5-turbo")
        | StrOutputParser()
    )
    
    # Process multiple topics concurrently
    topics = ["quantum computing", "machine learning", "blockchain", "robotics"]
    
    # Run all requests concurrently
    tasks = [chain.ainvoke({"topic": topic}) for topic in topics]
    results = await asyncio.gather(*tasks)
    
    # Display results
    for topic, result in zip(topics, results):
        print(f"{topic.title()}: {result}")

# Run the async example
# asyncio.run(async_chain_example())
```

```typescript
// TypeScript equivalent - naturally async in Node.js
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";

async function asyncChainExample() {
  // Create async chain
  const chain = PromptTemplate.fromTemplate("Explain {topic} in one sentence")
    .pipe(new ChatOpenAI({
      model: "gpt-3.5-turbo",
      openAIApiKey: process.env.OPENAI_API_KEY,
    }))
    .pipe(new StringOutputParser());
  
  // Process multiple topics concurrently
  const topics = ["quantum computing", "machine learning", "blockchain", "robotics"];
  
  try {
    // Run all requests concurrently
    const results = await Promise.all(
      topics.map(topic => chain.invoke({ topic }))
    );
    
    // Display results
    topics.forEach((topic, index) => {
      console.log(`${topic.charAt(0).toUpperCase() + topic.slice(1)}: ${results[index]}`);
    });
    
    return results;
  } catch (error) {
    console.error("Async chain execution failed:", error);
    throw error;
  }
}

// Run the async example
asyncChainExample().catch(console.error);
```

## Custom Output Parsers

Create custom parsers for structured output:

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json
import re

class JsonOutputParser(BaseOutputParser):
    """Parse JSON output from LLM."""
    
    def parse(self, text: str) -> dict:
        # Extract JSON from the response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format"}
        return {"error": "No JSON found in response"}

class ListOutputParser(BaseOutputParser):
    """Parse numbered list output from LLM."""
    
    def parse(self, text: str) -> list:
        lines = text.strip().split('\n')
        items = []
        for line in lines:
            # Match numbered list items (1. Item, 2. Item, etc.)
            match = re.match(r'^\d+\.\s*(.+)', line.strip())
            if match:
                items.append(match.group(1))
        return items

# Use custom parsers
json_chain = (
    PromptTemplate.from_template(
        "Create a JSON object with information about {topic}. "
        "Include name, description, and category fields."
    )
    | ChatOpenAI(model="gpt-3.5-turbo")
    | JsonOutputParser()
)

list_chain = (
    PromptTemplate.from_template(
        "List 5 benefits of {topic}. Format as a numbered list."
    )
    | ChatOpenAI(model="gpt-3.5-turbo")
    | ListOutputParser()
)

# Test the parsers
json_result = json_chain.invoke({"topic": "electric vehicles"})
list_result = list_chain.invoke({"topic": "renewable energy"})

print("JSON Result:", json_result)
print("List Result:", list_result)
```

```typescript
// TypeScript equivalent
import { BaseOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import json5 from "json5";

class JsonOutputParser extends BaseOutputParser {
  // Parse JSON output from LLM
  async parse(text: string): Promise<any> {
    // Extract JSON from the response
    const jsonMatch = text.match(/\{.*\}/s);
    if (jsonMatch) {
      try {
        return json5.parse(jsonMatch[0]);
      } catch (error) {
        return { error: "Invalid JSON format" };
      }
    }
    return { error: "No JSON found in response" };
  }
}

class ListOutputParser extends BaseOutputParser {
  // Parse numbered list output from LLM
  async parse(text: string): Promise<string[]> {
    const lines = text.trim().split("\n");
    const items: string[] = [];
    for (const line of lines) {
      // Match numbered list items (1. Item, 2. Item, etc.)
      const match = line.trim().match(/^(\d+)\.\s*(.+)/);
      if (match) {
        items.push(match[2]);
      }
    }
    return items;
  }
}

// Use custom parsers
const jsonChain = PromptTemplate.fromTemplate(
  "Create a JSON object with information about {topic}. " +
  "Include name, description, and category fields."
)
.pipe(new ChatOpenAI({
  model: "gpt-3.5-turbo",
  openAIApiKey: process.env.OPENAI_API_KEY,
}))
.pipe(new JsonOutputParser());

const listChain = PromptTemplate.fromTemplate(
  "List 5 benefits of {topic}. Format as a numbered list."
)
.pipe(new ChatOpenAI({
  model: "gpt-3.5-turbo",
  openAIApiKey: process.env.OPENAI_API_KEY,
}))
.pipe(new ListOutputParser());

// Test the parsers
const jsonResult = await jsonChain.invoke({ topic: "electric vehicles" });
const listResult = await listChain.invoke({ topic: "renewable energy" });

console.log("JSON Result:", jsonResult);
console.log("List Result:", listResult);
```

## Retrieval Chains

Combine retrieval with generation for RAG (Retrieval-Augmented Generation):

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# Sample documents
docs = [
    Document(page_content="Python is a programming language known for its simplicity."),
    Document(page_content="Machine learning is a subset of artificial intelligence."),
    Document(page_content="LangChain is a framework for building AI applications."),
]

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Create RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PromptTemplate.from_template(
        "Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    | ChatOpenAI(model="gpt-3.5-turbo")
)

# Use the RAG chain
result = rag_chain.invoke("What is LangChain?")
print(result.content)
```

```typescript
// TypeScript equivalent
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { FAISS } from "langchain-embeddings-faiss";
import { Document } from "@langchain/core/documents";
import { RunnablePassthrough } from "@langchain/core/runnables";

async function retrievalChains() {
  // Sample documents
  const docs: Document[] = [
    new Document({ pageContent: "Python is a programming language known for its simplicity." }),
    new Document({ pageContent: "Machine learning is a subset of artificial intelligence." }),
    new Document({ pageContent: "LangChain is a framework for building AI applications." }),
  ];

  // Create vector store
  const embeddings = new OpenAIEmbeddings();
  const vectorstore = await FAISS.fromDocuments(docs, embeddings);
  const retriever = vectorstore.asRetriever();

  // Create RAG chain
  const formatDocs = (docs: Document[]) => docs.map(doc => doc.pageContent).join("\n\n");

  const ragChain = PromptTemplate.fromTemplate(
    "Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
  )
  .pipe(new ChatOpenAI({
    model: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
  }));

  // Use the RAG chain
  try {
    const result = await ragChain.invoke({
      context: await retriever.invoke("What is LangChain?"),
      question: "What is LangChain?"
    });
    console.log(result.content);
  } catch (error) {
    console.error("Retrieval chain execution failed:", error);
    throw error;
  }
}

retrievalChains().catch(console.error);
```

## Chain Composition Patterns

### Map-Reduce Pattern

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Documents to process
documents = [
    "Document 1: AI is transforming healthcare...",
    "Document 2: Machine learning improves diagnosis...",
    "Document 3: Robots assist in surgery...",
]

# Map step: Summarize each document
map_prompt = PromptTemplate.from_template(
    "Summarize this document in one sentence: {doc}"
)

map_chain = map_prompt | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()

# Process each document
summaries = [map_chain.invoke({"doc": doc}) for doc in documents]

# Reduce step: Combine summaries
reduce_prompt = PromptTemplate.from_template(
    "Combine these summaries into a final summary:\n{summaries}"
)

reduce_chain = reduce_prompt | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()

final_summary = reduce_chain.invoke({"summaries": "\n".join(summaries)})
print(final_summary)
```

```typescript
// TypeScript equivalent
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";

async function mapReducePattern() {
  // Documents to process
  const documents = [
    "Document 1: AI is transforming healthcare...",
    "Document 2: Machine learning improves diagnosis...",
    "Document 3: Robots assist in surgery...",
  ];

  // Map step: Summarize each document
  const mapPrompt = PromptTemplate.fromTemplate(
    "Summarize this document in one sentence: {doc}"
  );

  const mapChain = mapPrompt.pipe(new ChatOpenAI({
    model: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
  })).pipe(new StringOutputParser());

  // Process each document
  const summaries = await Promise.all(documents.map(doc => mapChain.invoke({ doc })));

  // Reduce step: Combine summaries
  const reducePrompt = PromptTemplate.fromTemplate(
    "Combine these summaries into a final summary:\n{summaries}"
  );

  const reduceChain = reducePrompt.pipe(new ChatOpenAI({
    model: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
  })).pipe(new StringOutputParser());

  const finalSummary = await reduceChain.invoke({ summaries: summaries.join("\n") });
  console.log(finalSummary);
}

mapReducePattern().catch(console.error);
```

### Pipeline Pattern

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def extract_keywords(text):
    """Extract keywords from text."""
    # Simplified keyword extraction
    words = text.lower().split()
    keywords = [word for word in words if len(word) > 5][:5]
    return {"keywords": ", ".join(keywords)}

def format_output(data):
    """Format the final output."""
    return f"Summary: {data['summary']}\nKeywords: {data['keywords']}"

# Create pipeline
pipeline = (
    # Step 1: Generate summary
    PromptTemplate.from_template("Summarize this text: {text}")
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
    | RunnableLambda(lambda summary: {"summary": summary, "text": summary})
    
    # Step 2: Extract keywords
    | RunnableLambda(lambda data: {
        "summary": data["summary"],
        "keywords": extract_keywords(data["text"])["keywords"]
    })
    
    # Step 3: Format output
    | RunnableLambda(format_output)
)

result = pipeline.invoke({
    "text": "Artificial intelligence is revolutionizing healthcare through machine learning algorithms that can analyze medical images, predict patient outcomes, and assist doctors in making more accurate diagnoses."
})
print(result)
```

```typescript
// TypeScript equivalent
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableLambda } from "@langchain/core/runnables";

async function pipelinePattern() {
  // Step 1: Generate summary
  const step1 = PromptTemplate.fromTemplate("Summarize this text: {text}")
    .pipe(new ChatOpenAI({
      model: "gpt-3.5-turbo",
      openAIApiKey: process.env.OPENAI_API_KEY,
    }))
    .pipe(new StringOutputParser())
    .pipe(RunnableLambda.from((summary: string) => ({ summary, text: summary })));

  // Step 2: Extract keywords
  const step2 = RunnableLambda.from((data: { summary: string, text: string }) => {
    // Simplified keyword extraction
    const words = data.text.toLowerCase().split(" ");
    const keywords = words.filter(word => word.length > 5).slice(0, 5);
    return { summary: data.summary, keywords: keywords.join(", ") };
  });

  // Step 3: Format output
  const step3 = RunnableLambda.from((data: { summary: string, keywords: string }) => {
    return `Summary: ${data.summary}\nKeywords: ${data.keywords}`;
  });

  // Create pipeline
  const pipeline = step1.pipe(step2).pipe(step3);

  const result = await pipeline.invoke({
    text: "Artificial intelligence is revolutionizing healthcare through machine learning algorithms that can analyze medical images, predict patient outcomes, and assist doctors in making more accurate diagnoses."
  });
  console.log(result);
}

pipelinePattern().catch(console.error);
```

## Best Practices

### 1. Use LCEL for Modern Chains
```python
# Preferred: LCEL syntax
chain = prompt | llm | output_parser

# Legacy: Traditional chain classes (avoid for new code)
# chain = LLMChain(llm=llm, prompt=prompt)
```

### 2. Handle Errors Gracefully
```python
from langchain_core.runnables import RunnableLambda

def safe_chain_with_fallback():
    main_chain = prompt | llm | output_parser
    fallback_chain = RunnableLambda(lambda x: "Sorry, I couldn't process your request.")
    
    return main_chain.with_fallbacks([fallback_chain])
```

### 3. Use Type Hints
```python
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda

def typed_processor(inputs: Dict[str, Any]) -> Dict[str, str]:
    return {"processed": inputs["text"].upper()}

chain = RunnableLambda(typed_processor) | prompt | llm
```

### 4. Implement Proper Logging
```python
import logging
from langchain_core.runnables import RunnableLambda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_step(data):
    logger.info(f"Processing: {data}")
    return data

chain = RunnableLambda(log_step) | prompt | llm | RunnableLambda(log_step)
```

### 5. Test Your Chains
```python
def test_chain():
    test_cases = [
        {"input": "test input 1", "expected": "expected output 1"},
        {"input": "test input 2", "expected": "expected output 2"},
    ]
    
    for case in test_cases:
        result = chain.invoke(case["input"])
        assert result is not None, f"Chain failed for input: {case['input']}"
        print(f"✅ Test passed for: {case['input']}")

# test_chain()
```

### 6. Monitor Performance
```python
import time
from langchain_core.runnables import RunnableLambda

def time_operation(data):
    start_time = time.time()
    # Process data here
    end_time = time.time()
    print(f"Operation took {end_time - start_time:.2f} seconds")
    return data

timed_chain = RunnableLambda(time_operation) | prompt | llm
```