import { DocSection, FeatureCard, QuickStart } from '@/components/DocSection';
import { CodeBlock } from '@/components/CodeBlock';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Code, Database, Zap, Settings, MessageSquare, FileText, Bot } from 'lucide-react';

export const LangChainSection = () => {
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

  return (
    <DocSection
      id="langchain"
      title="LangChain Framework"
      description="Build powerful LLM applications with modular components, chains, and integrations."
      badges={["Core Framework", "Production Ready"]}
      externalLinks={[
        { title: "LangChain Docs", url: "https://python.langchain.com/docs/introduction/" },
        { title: "API Reference", url: "https://python.langchain.com/api_reference/" },
        { title: "Tutorials", url: "https://python.langchain.com/docs/tutorials/" }
      ]}
    >
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
          <h2 className="text-2xl font-semibold">Code Examples</h2>
          <Tabs defaultValue="basic" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="basic">Basic Chat</TabsTrigger>
              <TabsTrigger value="chains">Chains</TabsTrigger>
              <TabsTrigger value="rag">RAG System</TabsTrigger>
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
                </ul>
              </div>
              <div className="space-y-3">
                <h4 className="font-medium">Production</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Monitor token usage and costs</li>
                  <li>• Implement rate limiting and quotas</li>
                  <li>• Use LangSmith for tracing and debugging</li>
                  <li>• Validate inputs and sanitize outputs</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DocSection>
  );
};