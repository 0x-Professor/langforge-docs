import { FeatureCard, QuickStart } from '@/components/DocSection';
import { CodeBlock } from '@/components/CodeBlock';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Code, Database, Zap, Settings, MessageSquare, FileText, Bot, Cpu, Layers, CheckCircle2 } from 'lucide-react';
import { Callout } from '@/components/docs/DocHeader';
import { VersionBadge, BetaBadge, ApiReferenceLink } from '@/templates/LangChainDocPage';
import { DocLayout } from '@/components/docs/DocLayout';

interface LangChainSectionProps {
  // Add any props if needed
}

export const LangChainSection: React.FC<LangChainSectionProps> = () => {
  // Define the features for the models section
  const modelFeatures = [
    {
      title: 'LLMs & Chat Models',
      description: 'Support for various LLM providers with consistent interfaces',
      icon: <MessageSquare className="h-6 w-6 text-blue-500" />,
    },
    {
      title: 'Embeddings',
      description: 'Generate embeddings for text with different embedding models',
      icon: <Layers className="h-6 w-6 text-green-500" />,
    },
    {
      title: 'Token Usage',
      description: 'Track token usage and costs across different models',
      icon: <CheckCircle2 className="h-6 w-6 text-purple-500" />,
    },
  ];

  // Define the code examples for the models section
  const modelCodeExamples = [
    {
      title: 'Basic LLM Usage',
      description: 'Initialize and use a language model',
      code: `from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Initialize LLM
llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_tokens=1000,
    streaming=True
)

# Generate text with token usage tracking
with get_openai_callback() as cb:
    response = llm.invoke("Explain quantum computing in simple terms.")
    print(f"Response: {response}")
    print(f"Tokens used: {cb.total_tokens}")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")
    print(f"Total cost (USD): ${cb.total_cost}")`,
      language: 'python',
    },
    {
      title: 'Chat Models',
      description: 'Use chat models with message history',
      code: `from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize chat model
chat = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# Create messages with system prompt
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain quantum computing in simple terms.")
]

# Get response
response = chat.invoke(messages)
print(response.content)`,
      language: 'python',
    },
  ];
  const basicChatCode = `from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize chat model
model = init_chat_model("gpt-4", model_provider="openai")

# Create messages
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain quantum computing in simple terms.")
]

# Get response
response = model.invoke(messages)
print(response.content)`;

  const chainCode = `from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains {topic} clearly."),
    ("human", "{question}")
])

# Create a chain
chain = prompt | model | StrOutputParser()

# Use the chain
result = chain.invoke({
    "topic": "machine learning",
    "question": "What is supervised learning?"
})

print(result)`;

  const ragCode = `from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load and split documents
loader = TextLoader("docs.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)

# Create retriever
retriever = vectorstore.as_retriever()

# RAG chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=retriever
)

# Ask questions
response = qa_chain.invoke({"query": "What are the main topics?"})
print(response["result"])`;

  const integrationsCode = `# OpenAI
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4")

# Anthropic
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-sonnet-20240229")

# Google
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-pro")

# Azure OpenAI
from langchain_openai import AzureChatOpenAI
model = AzureChatOpenAI(
    deployment_name="gpt-4",
    model_name="gpt-4",
)

# Local models with Ollama
from langchain_community.llms import Ollama
model = Ollama(model="llama2")`;

  const agentCode = `from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
import requests

# Define custom tools
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Mock weather API call
    return f"Weather in {city}: 72°F, sunny"

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Mock search results
    return f"Search results for '{query}': Found relevant information"

# Create agent
llm = ChatOpenAI(model="gpt-4")
tools = [get_weather, search_web]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Use the agent
result = agent_executor.invoke({"input": "What's the weather like in Paris?"})
print(result["output"])`;

  const memoryCode = `from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain

# Conversation Buffer Memory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=model,
    memory=memory,
    verbose=True
)

# Chat with memory
response1 = conversation.predict(input="Hi, my name is Alice")
response2 = conversation.predict(input="What's my name?")

print(f"Response 1: {response1}")
print(f"Response 2: {response2}")

# Summary Memory for long conversations
summary_memory = ConversationSummaryMemory(
    llm=model,
    max_token_limit=100
)

conversation_with_summary = ConversationChain(
    llm=model,
    memory=summary_memory,
    verbose=True
)`;

  const streamingCode = `from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI

# Streaming chat
streaming_llm = ChatOpenAI(
    model="gpt-4",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Stream response
response = streaming_llm.invoke([
    HumanMessage(content="Tell me a story about AI")
])

# Async streaming
import asyncio
from langchain.callbacks.base import AsyncCallbackHandler

class CustomAsyncHandler(AsyncCallbackHandler):
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Token: {token}", end="", flush=True)

async def stream_chat():
    async_llm = ChatOpenAI(
        model="gpt-4",
        streaming=True,
        callbacks=[CustomAsyncHandler()]
    )
    
    response = await async_llm.ainvoke([
        HumanMessage(content="Explain machine learning")
    ])
    
    return response

# Run async
# asyncio.run(stream_chat())`;

  const customChainCode = `from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Custom processing function
def extract_keywords(text: str) -> dict:
    """Extract keywords from text"""
    keywords = text.split()[:5]  # Simple keyword extraction
    return {"keywords": keywords, "word_count": len(text.split())}

# Create custom chain
keyword_extractor = RunnableLambda(extract_keywords)

# Parallel processing
parallel_chain = RunnableParallel({
    "summary": PromptTemplate.from_template("Summarize: {text}") | model | StrOutputParser(),
    "keywords": RunnableLambda(lambda x: x["text"]) | keyword_extractor,
    "sentiment": PromptTemplate.from_template("Analyze sentiment of: {text}") | model | StrOutputParser()
})

# Use parallel chain
result = parallel_chain.invoke({
    "text": "LangChain is an amazing framework for building LLM applications!"
})

print(result)`;

  const productionCode = `import os
from langchain_openai import ChatOpenAI
from langchain.callbacks import LangChainTracer
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up caching
set_llm_cache(InMemoryCache())

# Production configuration
class ProductionLLMConfig:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            max_retries=3,
            request_timeout=30,
            callbacks=[LangChainTracer()]
        )
    
    def create_chain(self, prompt_template):
        """Create a production-ready chain"""
        try:
            chain = prompt_template | self.model | StrOutputParser()
            return chain
        except Exception as e:
            logger.error(f"Failed to create chain: {e}")
            raise
    
    def safe_invoke(self, chain, input_data):
        """Safely invoke chain with error handling"""
        try:
            result = chain.invoke(input_data)
            logger.info(f"Successfully processed: {input_data}")
            return result
        except Exception as e:
            logger.error(f"Chain invocation failed: {e}")
            return {"error": str(e)}

# Usage
config = ProductionLLMConfig()
prompt = ChatPromptTemplate.from_template("Answer: {question}")
chain = config.create_chain(prompt)
result = config.safe_invoke(chain, {"question": "What is AI?"})`;

  const evaluationCode = `from langchain.evaluation import load_evaluator
from langchain.evaluation.criteria import CriteriaEvalChain
from langchain_openai import ChatOpenAI

# Set up evaluator
evaluator_llm = ChatOpenAI(model="gpt-4", temperature=0)

# Criteria evaluation
criteria_evaluator = load_evaluator(
    "criteria",
    criteria="helpfulness",
    llm=evaluator_llm
)

# Evaluate response
evaluation_result = criteria_evaluator.evaluate_strings(
    input="What is machine learning?",
    prediction="Machine learning is a subset of AI that enables computers to learn without explicit programming.",
)

print(f"Evaluation Score: {evaluation_result['score']}")
print(f"Reasoning: {evaluation_result['reasoning']}")

# Custom evaluation
from langchain.evaluation.criteria import LabeledCriteriaEvalChain

custom_criteria = {
    "accuracy": "Is the response factually accurate?",
    "clarity": "Is the response clear and easy to understand?",
    "completeness": "Does the response fully address the question?"
}

custom_evaluator = LabeledCriteriaEvalChain.from_llm(
    llm=evaluator_llm,
    criteria=custom_criteria
)

# Evaluate with custom criteria
custom_result = custom_evaluator.evaluate_strings(
    input="Explain neural networks",
    prediction="Neural networks are computational models inspired by biological neural networks.",
    reference="Neural networks are machine learning models consisting of interconnected nodes that process information similar to neurons in the brain."
)`;

  const chatModelExample = `from langchain.chat_models import init_chat_model

# Initialize chat model
model = init_chat_model("gpt-4", model_provider="openai")

# Create messages
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain quantum computing in simple terms.")
]

# Get response
response = model.invoke(messages)
print(response.content)`;

  const llmExample = `from langchain_openai import ChatOpenAI

# Initialize LLM
model = ChatOpenAI(model="gpt-4")

# Get response
response = model.invoke("Explain machine learning in simple terms.")
print(response)`;

  const embeddingsExample = `from langchain_openai import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Get embeddings
vector = embeddings.encode("This is a test sentence.")
print(vector)`;

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="space-y-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">LangChain Framework</h1>
          <p className="text-xl text-muted-foreground">
            Build powerful LLM applications with modular components, chains, and integrations.
          </p>
          <div className="flex justify-center gap-4 mt-4">
            <a 
              href="https://python.langchain.com/docs/introduction/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Documentation
            </a>
            <a 
              href="https://python.langchain.com/api_reference/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              API Reference
            </a>
          </div>
        </div>
      {/* Core Concepts */}
      <div className="space-y-4">
        <h2 className="text-2xl font-semibold">Core Concepts</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <FeatureCard
            icon={<MessageSquare className="w-6 h-6" />}
            title="Chat Models"
            description="Unified interface for different LLM providers with consistent APIs."
            features={[
              "Multi-provider support",
              "Streaming responses",
              "Token usage tracking",
              "Custom configurations"
            ]}
          />
          <FeatureCard
            icon={<FileText className="w-6 h-6" />}
            title="Prompts & Templates"
            description="Dynamic prompt creation with variables and conditional logic."
            features={[
              "Template variables",
              "Few-shot examples",
              "Conditional prompts",
              "Prompt optimization"
            ]}
          />
          <FeatureCard
            icon={<Zap className="w-6 h-6" />}
            title="Chains & Runnables"
            description="Compose multiple components into powerful processing pipelines."
            features={[
              "Sequential processing",
              "Parallel execution",
              "Error handling",
              "Async support"
            ]}
          />
          <FeatureCard
            icon={<Database className="w-6 h-6" />}
            title="Vector Stores"
            description="Store and retrieve embeddings for semantic search and RAG."
            features={[
              "Multiple backends",
              "Similarity search",
              "Metadata filtering",
              "Hybrid search"
            ]}
          />
          <FeatureCard
            icon={<Code className="w-6 h-6" />}
            title="Output Parsers"
            description="Parse and validate LLM outputs into structured formats."
            features={[
              "JSON parsing",
              "Schema validation",
              "Custom formats",
              "Error recovery"
            ]}
          />
          <FeatureCard
            icon={<Settings className="w-6 h-6" />}
            title="Memory"
            description="Persist conversation context and maintain state across interactions."
            features={[
              "Conversation memory",
              "Entity extraction",
              "Summary memory",
              "Custom storage"
            ]}
          />
        </div>
      </div>
      
      <div className="space-y-8">
        {/* Core Concepts */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Core Concepts</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <FeatureCard
              icon={<MessageSquare className="w-6 h-6" />}
              title="Chat Models"
              description="Unified interface for different LLM providers with consistent APIs."
              features={[
                "Multi-provider support",
                "Streaming responses",
                "Token usage tracking",
                "Custom configurations"
              ]}
            />
            <FeatureCard
              icon={<FileText className="w-6 h-6" />}
              title="Prompts & Templates"
              description="Dynamic prompt creation with variables and conditional logic."
              features={[
                "Template variables",
                "Few-shot examples",
                "Conditional prompts",
                "Prompt optimization"
              ]}
            />
            <FeatureCard
              icon={<Zap className="w-6 h-6" />}
              title="Chains & Runnables"
              description="Compose multiple components into powerful processing pipelines."
              features={[
                "Sequential processing",
                "Parallel execution",
                "Error handling",
                "Async support"
              ]}
            />
            <FeatureCard
              icon={<Database className="w-6 h-6" />}
              title="Vector Stores"
              description="Store and retrieve embeddings for semantic search and RAG."
              features={[
                "Multiple backends",
                "Similarity search",
                "Metadata filtering",
                "Hybrid search"
              ]}
            />
            <FeatureCard
              icon={<Code className="w-6 h-6" />}
              title="Output Parsers"
              description="Parse and validate LLM outputs into structured formats."
              features={[
                "JSON parsing",
                "Schema validation",
                "Custom formats",
                "Error recovery"
              ]}
            />
            <FeatureCard
              icon={<Settings className="w-6 h-6" />}
              title="Memory"
              description="Persist conversation context and maintain state across interactions."
              features={[
                "Conversation memory",
                "Entity extraction",
                "Summary memory",
                "Custom storage"
              ]}
            />
          </div>
        </div>

        {/* Interactive Examples */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Comprehensive Code Examples</h2>
          <Tabs defaultValue="quickstart" className="w-full">
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="quickstart">Quick Start</TabsTrigger>
              <TabsTrigger value="models">Models</TabsTrigger>
              <TabsTrigger value="chat">Chat</TabsTrigger>
              <TabsTrigger value="chains">Chains</TabsTrigger>
              <TabsTrigger value="rag">RAG</TabsTrigger>
            </TabsList>

            <TabsContent value="quickstart" className="space-y-6">
              <QuickStart
                title="Get Started with LangChain"
                description="Start building with LangChain in minutes"
                codeExample={basicChatCode}
                steps={[
                  'Chat with AI models',
                  'Build complex chains',
                  'Integrate with tools',
                  'Add memory to applications'
                ]}
              />
            </TabsContent>

            <TabsContent value="models" className="space-y-6">
              <div className="space-y-4">
                <h2 className="text-2xl font-bold">Language Models</h2>
                <p className="text-muted-foreground">
                  LangChain provides a standard interface for working with various language models, including LLMs and Chat Models.
                </p>

                <Callout type="tip">
                  <p>LangChain supports multiple model providers including OpenAI, Anthropic, Google, and more.</p>
                </Callout>

                <h3 className="text-xl font-semibold mt-6">Chat Models</h3>
                <p>Chat models are a variation of language models that use a message-based interface.</p>
                <CodeBlock
                  code={chatModelExample}
                  language="python"
                  title="Using Chat Models"
                />

                <h3 className="text-xl font-semibold mt-6">LLMs</h3>
                <p>Traditional language models that take a string as input and return a string.</p>
                <CodeBlock
                  code={llmExample}
                  language="python"
                  title="Using LLMs"
                />

                <h3 className="text-xl font-semibold mt-6">Embeddings</h3>
                <p>Embeddings are used to convert text into vector representations for semantic search and other applications.</p>
                <CodeBlock
                  code={embeddingsExample}
                  language="python"
                  title="Working with Embeddings"
                />

                <div className="grid gap-4 md:grid-cols-2 mt-6">
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Supported Providers</CardTitle>
                      <Cpu className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">25+</div>
                      <p className="text-xs text-muted-foreground">model providers supported</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Model Types</CardTitle>
                      <Zap className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">3+</div>
                      <p className="text-xs text-muted-foreground">types of models (LLM, Chat, Embeddings)</p>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="chat" className="space-y-4">
              <Tabs defaultValue="basic" className="w-full">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="basic">Basic Chat</TabsTrigger>
                  <TabsTrigger value="chains">Chains</TabsTrigger>
                  <TabsTrigger value="rag">RAG</TabsTrigger>
                  <TabsTrigger value="integrations">Integrations</TabsTrigger>
                </TabsList>

                <TabsContent value="basic" className="space-y-4">
                  <CodeBlock
                    title="Basic Chat Model Usage"
                    language="python"
                    code={basicChatCode}
                  />
                </TabsContent>
                
                <TabsContent value="chains" className="space-y-4">
                  <CodeBlock
                    title="Building Chains with LCEL"
                    language="python"
                    code={chainCode}
                  />
                </TabsContent>
                
                <TabsContent value="rag" className="space-y-4">
                  <CodeBlock
                    title="Retrieval Augmented Generation"
                    language="python"
                    code={ragCode}
                  />
                </TabsContent>
                
                <TabsContent value="integrations" className="space-y-4">
                  <CodeBlock
                    title="Provider Integrations"
                    language="python"
                    code={integrationsCode}
                  />
                </TabsContent>
              </Tabs>

              <Tabs defaultValue="agents" className="w-full mt-6">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="agents">Agents</TabsTrigger>
                  <TabsTrigger value="memory">Memory</TabsTrigger>
                  <TabsTrigger value="streaming">Streaming</TabsTrigger>
                </TabsList>

                <TabsContent value="agents" className="space-y-4">
                  <CodeBlock
                    title="Building Agents with Tools"
                    language="python"
                    code={agentCode}
                  />
                </TabsContent>

                <TabsContent value="memory" className="space-y-4">
                  <CodeBlock
                    title="Conversation Memory Management"
                    language="python"
                    code={memoryCode}
                  />
                </TabsContent>

                <TabsContent value="streaming" className="space-y-4">
                  <CodeBlock
                    title="Streaming Responses & Async"
                    language="python"
                    code={streamingCode}
                  />
                </TabsContent>
              </Tabs>

              <div className="mt-6">
                <CodeBlock
                  title="Custom Chains & Parallel Processing"
                  language="python"
                  code={customChainCode}
                />
              </div>
            </TabsContent>
          </Tabs>
        </div>

        {/* Production & Evaluation */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Production & Evaluation</h2>
          <Tabs defaultValue="production" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="production">Production Setup</TabsTrigger>
              <TabsTrigger value="evaluation">Evaluation & Testing</TabsTrigger>
            </TabsList>
            
            <TabsContent value="production" className="space-y-4">
              <CodeBlock
                title="Production-Ready Configuration"
                language="python"
                code={productionCode}
              />
            </TabsContent>
            
            <TabsContent value="evaluation" className="space-y-4">
              <CodeBlock
                title="Model Evaluation & Testing"
                language="python"
                code={evaluationCode}
              />
            </TabsContent>
          </Tabs>
        </div>

        {/* Architecture Packages */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Package Architecture</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Core Packages</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <Code className="w-5 h-5 mt-0.5 text-primary" />
                    <div>
                      <h4 className="font-medium">langchain-core</h4>
                      <p className="text-sm text-muted-foreground">Base abstractions and interfaces</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Bot className="w-5 h-5 mt-0.5 text-primary" />
                    <div>
                      <h4 className="font-medium">langchain</h4>
                      <p className="text-sm text-muted-foreground">Chains, agents, and cognitive architecture</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Database className="w-5 h-5 mt-0.5 text-primary" />
                    <div>
                      <h4 className="font-medium">langchain-community</h4>
                      <p className="text-sm text-muted-foreground">Community-maintained integrations</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Integration Packages</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <Settings className="w-5 h-5 mt-0.5 text-primary" />
                    <div>
                      <h4 className="font-medium">langchain-openai</h4>
                      <p className="text-sm text-muted-foreground">OpenAI models and embeddings</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Settings className="w-5 h-5 mt-0.5 text-primary" />
                    <div>
                      <h4 className="font-medium">langchain-anthropic</h4>
                      <p className="text-sm text-muted-foreground">Anthropic Claude integration</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Settings className="w-5 h-5 mt-0.5 text-primary" />
                    <div>
                      <h4 className="font-medium">langchain-google-genai</h4>
                      <p className="text-sm text-muted-foreground">Google Gemini models</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Advanced Features */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Advanced Features</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Agents</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Create autonomous agents that can reason, plan, and use tools to accomplish complex tasks.
                </p>
                <div className="space-y-2">
                  <h5 className="font-medium">Key Components:</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• ReActAgent for reasoning and acting</li>
                    <li>• OpenAI Functions for tool calling</li>
                    <li>• Custom agent executors</li>
                    <li>• Memory and state management</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Document Processing</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Comprehensive document loading, splitting, and processing capabilities.
                </p>
                <div className="space-y-2">
                  <h5 className="font-medium">Supported Formats:</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• PDF, Word, PowerPoint</li>
                    <li>• CSV, JSON, XML</li>
                    <li>• Web pages and APIs</li>
                    <li>• Code repositories</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Evaluation & Testing</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Built-in evaluation frameworks for testing LLM applications.
                </p>
                <div className="space-y-2">
                  <h5 className="font-medium">Evaluation Types:</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Response quality metrics</li>
                    <li>• Retrieval accuracy (RAGAS)</li>
                    <li>• Custom evaluators</li>
                    <li>• A/B testing frameworks</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Streaming & Callbacks</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Real-time streaming responses and comprehensive callback system.
                </p>
                <div className="space-y-2">
                  <h5 className="font-medium">Features:</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Token-by-token streaming</li>
                    <li>• Intermediate step callbacks</li>
                    <li>• Progress tracking</li>
                    <li>• Error handling hooks</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Learning Paths */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Learning Paths</h2>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="shadow-card">
              <CardHeader>
                <CardTitle className="text-primary">Beginner Path</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <h5 className="font-medium">Week 1-2: Foundations</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Install LangChain and setup environment</li>
                    <li>• Basic chat models and prompting</li>
                    <li>• Understanding chains and LCEL</li>
                    <li>• Simple prompt templates</li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <h5 className="font-medium">Week 3-4: Building Blocks</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Output parsers and validation</li>
                    <li>• Document loaders and text splitters</li>
                    <li>• Basic RAG implementation</li>
                    <li>• Memory and conversation history</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle className="text-primary">Intermediate Path</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <h5 className="font-medium">Month 2: Advanced Concepts</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Complex chain compositions</li>
                    <li>• Vector stores and embeddings</li>
                    <li>• Agent frameworks and tools</li>
                    <li>• Async processing and streaming</li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <h5 className="font-medium">Month 3: Real Applications</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Multi-document RAG systems</li>
                    <li>• Custom tool development</li>
                    <li>• Performance optimization</li>
                    <li>• Error handling and debugging</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle className="text-primary">Advanced Path</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <h5 className="font-medium">Month 4-5: Production</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Production deployment patterns</li>
                    <li>• Monitoring and observability</li>
                    <li>• Custom chain architectures</li>
                    <li>• Multi-agent systems</li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <h5 className="font-medium">Month 6+: Mastery</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Custom integrations and extensions</li>
                    <li>• Contributing to LangChain ecosystem</li>
                    <li>• Research and experimentation</li>
                    <li>• Teaching and community building</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Common Use Cases */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Common Use Cases & Solutions</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Document Q&A Systems</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Build intelligent document analysis and question-answering systems.
                </p>
                <div className="space-y-2">
                  <h5 className="font-medium">Implementation Steps:</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>1. Load documents with appropriate loaders</li>
                    <li>2. Split text into manageable chunks</li>
                    <li>3. Generate embeddings and store in vector DB</li>
                    <li>4. Create retrieval chain with semantic search</li>
                    <li>5. Add conversation memory for context</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Chatbots & Virtual Assistants</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Create conversational AI with personality and domain expertise.
                </p>
                <div className="space-y-2">
                  <h5 className="font-medium">Key Components:</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Conversation memory for context retention</li>
                    <li>• Custom prompt templates for personality</li>
                    <li>• Tool integration for external actions</li>
                    <li>• Streaming for real-time responses</li>
                    <li>• Fallback handling for edge cases</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Code Analysis & Generation</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Analyze codebases and generate code with AI assistance.
                </p>
                <div className="space-y-2">
                  <h5 className="font-medium">Applications:</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Code review and bug detection</li>
                    <li>• Documentation generation</li>
                    <li>• Code refactoring suggestions</li>
                    <li>• Test case generation</li>
                    <li>• API documentation from code</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Content Creation & SEO</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Automate content creation with AI-powered writing assistants.
                </p>
                <div className="space-y-2">
                  <h5 className="font-medium">Features:</h5>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Blog post and article generation</li>
                    <li>• SEO optimization suggestions</li>
                    <li>• Social media content creation</li>
                    <li>• Content personalization</li>
                    <li>• Multi-language support</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Troubleshooting Guide */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Troubleshooting Guide</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Common Issues</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-3">
                  <div>
                    <h5 className="font-medium text-destructive">Rate Limit Errors</h5>
                    <p className="text-sm text-muted-foreground">
                      Implement exponential backoff and request queuing.
                    </p>
                  </div>
                  <div>
                    <h5 className="font-medium text-destructive">Memory Issues</h5>
                    <p className="text-sm text-muted-foreground">
                      Use conversation summarization for long chats.
                    </p>
                  </div>
                  <div>
                    <h5 className="font-medium text-destructive">Slow Responses</h5>
                    <p className="text-sm text-muted-foreground">
                      Enable caching and optimize chunk sizes.
                    </p>
                  </div>
                  <div>
                    <h5 className="font-medium text-destructive">Token Limits</h5>
                    <p className="text-sm text-muted-foreground">
                      Implement token counting and context management.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Performance Tips</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-3">
                  <div>
                    <h5 className="font-medium text-success">Async Processing</h5>
                    <p className="text-sm text-muted-foreground">
                      Use async methods for better concurrency.
                    </p>
                  </div>
                  <div>
                    <h5 className="font-medium text-success">Batch Operations</h5>
                    <p className="text-sm text-muted-foreground">
                      Process multiple items together when possible.
                    </p>
                  </div>
                  <div>
                    <h5 className="font-medium text-success">Smart Caching</h5>
                    <p className="text-sm text-muted-foreground">
                      Cache embeddings and expensive computations.
                    </p>
                  </div>
                  <div>
                    <h5 className="font-medium text-success">Model Selection</h5>
                    <p className="text-sm text-muted-foreground">
                      Choose appropriate models for your use case.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Best Practices */}
        <div className="space-y-6">
          <Card className="shadow-card border-l-4 border-l-primary">
            <CardHeader>
              <CardTitle>Best Practices</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <h4 className="font-medium">Development</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Use LCEL (LangChain Expression Language) for chains</li>
                    <li>• Implement proper error handling and retries</li>
                    <li>• Cache expensive operations like embeddings</li>
                    <li>• Use async methods for better performance</li>
                    <li>• Structure prompts with clear instructions</li>
                    <li>• Use templates for consistent formatting</li>
                  </ul>
                </div>
                <div className="space-y-3">
                  <h4 className="font-medium">Production</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Monitor token usage and costs</li>
                    <li>• Implement rate limiting and quotas</li>
                    <li>• Use LangSmith for tracing and debugging</li>
                    <li>• Validate inputs and sanitize outputs</li>
                    <li>• Set up proper logging and monitoring</li>
                    <li>• Test with diverse data sets</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>Code Analysis & Generation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Analyze codebases and generate code with AI assistance.
              </p>
              <div className="space-y-2">
                <h5 className="font-medium">Applications:</h5>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Code review and bug detection</li>
                  <li>• Documentation generation</li>
                  <li>• Code refactoring suggestions</li>
                  <li>• Test case generation</li>
                  <li>• API documentation from code</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>Content Creation & SEO</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Automate content creation with AI-powered writing assistants.
              </p>
              <div className="space-y-2">
                <h5 className="font-medium">Features:</h5>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Blog post and article generation</li>
                  <li>• SEO optimization suggestions</li>
                  <li>• Social media content creation</li>
                  <li>• Content personalization</li>
                  <li>• Multi-language support</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Troubleshooting Guide */}
      <div className="space-y-4">
        <h2 className="text-2xl font-semibold">Troubleshooting Guide</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>Common Issues</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-3">
                <div>
                  <h5 className="font-medium text-destructive">Rate Limit Errors</h5>
                  <p className="text-sm text-muted-foreground">
                    Implement exponential backoff and request queuing.
                  </p>
                </div>
                <div>
                  <h5 className="font-medium text-destructive">Memory Issues</h5>
                  <p className="text-sm text-muted-foreground">
                    Use conversation summarization for long chats.
                  </p>
                </div>
                <div>
                  <h5 className="font-medium text-destructive">Slow Responses</h5>
                  <p className="text-sm text-muted-foreground">
                    Enable caching and optimize chunk sizes.
                  </p>
                </div>
                <div>
                  <h5 className="font-medium text-destructive">Token Limits</h5>
                  <p className="text-sm text-muted-foreground">
                    Implement token counting and context management.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>Performance Tips</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-3">
                <div>
                  <h5 className="font-medium text-success">Async Processing</h5>
                  <p className="text-sm text-muted-foreground">
                    Use async methods for better concurrency.
                  </p>
                </div>
                <div>
                  <h5 className="font-medium text-success">Batch Operations</h5>
                  <p className="text-sm text-muted-foreground">
                    Process multiple items together when possible.
                  </p>
                </div>
                <div>
                  <h5 className="font-medium text-success">Smart Caching</h5>
                  <p className="text-sm text-muted-foreground">
                    Cache embeddings and expensive computations.
                  </p>
                </div>
                <div>
                  <h5 className="font-medium text-success">Model Selection</h5>
                  <p className="text-sm text-muted-foreground">
                    Choose appropriate models for your use case.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Best Practices */}
      <Card className="shadow-card border-l-4 border-l-primary">
        <CardHeader>
          <CardTitle>Best Practices</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h4 className="font-medium">Development</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Use LCEL (LangChain Expression Language) for chains</li>
                <li>• Implement proper error handling and retries</li>
                <li>• Cache expensive operations like embeddings</li>
                <li>• Use async methods for better performance</li>
                <li>• Structure prompts with clear instructions</li>
                <li>• Use templates for consistent formatting</li>
              </ul>
            </div>
            <div className="space-y-3">
              <h4 className="font-medium">Production</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Monitor token usage and costs</li>
                <li>• Implement rate limiting and quotas</li>
                <li>• Use LangSmith for tracing and debugging</li>
                <li>• Validate inputs and sanitize outputs</li>
                <li>• Set up proper logging and monitoring</li>
                <li>• Test with diverse data sets</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default LangChainSection;