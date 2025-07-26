import { LangChainDocPage } from '@/templates/LangChainDocPage';
import { Code2, MessageSquare, Layers, CheckCircle2 } from 'lucide-react';
import { VersionBadge, BetaBadge, ApiReferenceLink } from '@/templates/LangChainDocPage';
import { CodeBlock } from '@/components/ui/code-block';

type CalloutProps = {
  type: 'info' | 'warning' | 'danger' | 'success';
  children: React.ReactNode;
};

const Callout = ({ type, children }: CalloutProps) => {
  const typeStyles = {
    info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-500',
    warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500',
    danger: 'bg-red-50 dark:bg-red-900/20 border-red-500',
    success: 'bg-green-50 dark:bg-green-900/20 border-green-500',
  };

  const icon = {
    info: (
      <svg className="h-5 w-5 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
      </svg>
    ),
    warning: (
      <svg className="h-5 w-5 text-yellow-500" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
      </svg>
    ),
    danger: (
      <svg className="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
      </svg>
    ),
    success: (
      <svg className="h-5 w-5 text-green-500" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
      </svg>
    ),
  };

  return (
    <div className={`border-l-4 p-4 my-4 rounded-r ${typeStyles[type]}`}>
      <div className="flex">
        <div className="flex-shrink-0">
          {icon[type]}
        </div>
        <div className="ml-3">
          <div className="text-sm text-gray-700 dark:text-gray-300">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

const llmExample = `from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Initialize LLM with streaming
llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_tokens=1000,
    streaming=True
)

# Generate text with token usage tracking
# The callback will automatically track token usage and cost
cb = get_openai_callback()
try:
    response = llm.invoke("Explain quantum computing in simple terms.")
    print(f"Response: {response}")
    print(f"Tokens used: {cb.total_tokens}")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")
    print(f"Total cost (USD): ${cb.total_cost}")
finally:
    cb.__exit__(None, None, None)
`;

const chatModelExample = `from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Initialize chat model with streaming
chat = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Create messages with system prompt
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain quantum computing in simple terms.")
]

# Stream response
response = chat.invoke(messages)
`;

export default function ModelsDocumentation() {
  return (
    <LangChainDocPage
      title="LangChain Models"
      description="Learn how to work with different types of models in LangChain, including LLMs, chat models, and embeddings."
      installation={{
        npm: 'langchain',
        pip: 'langchain',
      }}
      resources={[
        {
          title: 'LLM Cookbook',
          href: 'https://github.com/langchain-ai/langchain-cookbook',
          type: 'github',
        },
        {
          title: 'Model I/O Documentation',
          href: 'https://python.langchain.com/docs/modules/model_io/',
          type: 'documentation',
        },
      ]}
      apiReference="https://api.python.langchain.com/en/latest/langchain_api_reference.html#module-langchain.llms"
    >

  return (
      <section id="overview" className="space-y-6">
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Overview</h2>
          <p>
            LangChain provides a unified interface for working with various language models, 
            making it easy to switch between different providers and model types. The library 
            supports LLMs, chat models, and embeddings.
          </p>
          
          <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-4 rounded-r">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  <strong>Tip:</strong> When choosing between LLMs and chat models, prefer chat models for conversation-based 
                  applications as they are specifically designed for multi-turn conversations.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
          <div className="border rounded-lg p-6 bg-card">
            <div className="flex items-center mb-3">
              <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400">
                <Code2 className="w-5 h-5" />
              </div>
              <h3 className="ml-3 font-semibold">LLMs</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Text-in, text-out models for single-turn tasks like completion and summarization.
            </p>
          </div>
          
          <div className="border rounded-lg p-6 bg-card">
            <div className="flex items-center mb-3">
              <div className="p-2 rounded-lg bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400">
                <MessageSquare className="w-5 h-5" />
              </div>
              <h3 className="ml-3 font-semibold">Chat Models</h3>
              <BetaBadge />
            </div>
            <p className="text-sm text-muted-foreground">
              Message-in, message-out models designed for multi-turn conversations.
            </p>
          </div>
          
          <div className="border rounded-lg p-6 bg-card">
            <div className="flex items-center mb-3">
              <div className="p-2 rounded-lg bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400">
                <Layers className="w-5 h-5" />
              </div>
              <h3 className="ml-3 font-semibold">Embeddings</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Convert text to vector representations for semantic search and retrieval.
            </p>
          </div>
        </div>
      </section>

      <section id="llms" className="space-y-6">
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">LLMs <VersionBadge version="0.1.0" /></h2>
          <p>
            LLMs (Large Language Models) are text-in, text-out models that take a text string as 
            input and return a text string as output. They are great for single-turn tasks like 
            text completion, summarization, and question answering.
          </p>
          
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-400 p-4 rounded-r">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-yellow-500" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-yellow-700 dark:text-yellow-300">
                  <strong>Note:</strong> For most use cases, we recommend using chat models instead of raw LLMs as they provide
                  better support for conversation history and system prompts.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Basic Usage</h3>
          <CodeBlock 
            code={llmExample} 
            language="python"
            showLineNumbers
          />
          
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium mb-2">Key Features</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start">
                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                  <span>Streaming support for real-time output</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                  <span>Token usage tracking</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                  <span>Automatic retry on rate limits</span>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Common Use Cases</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>Text generation</li>
                <li>Summarization</li>
                <li>Question answering</li>
                <li>Text classification</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Available Providers</h3>
          <p>
            LangChain supports multiple LLM providers through a unified interface. Here are some of the most popular ones:
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
            {[
              { name: 'OpenAI', status: 'stable' },
              { name: 'Anthropic', status: 'beta' },
              { name: 'Cohere', status: 'stable' },
              { name: 'HuggingFace', status: 'stable' },
              { name: 'Replicate', status: 'beta' },
              { name: 'Custom', status: 'stable' },
            ].map((provider) => (
              <div key={provider.name} className="border rounded-lg p-4 flex items-center justify-between">
                <span className="font-medium">{provider.name}</span>
                {provider.status === 'beta' && <BetaBadge />}
                {provider.status === 'stable' && (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">
                    Stable
                  </span>
                )}
              </div>
            ))}
          </div>
          
          <p className="text-sm text-muted-foreground">
            See the <ApiReferenceLink href="https://api.python.langchain.com/en/latest/langchain_api_reference.html#module-langchain.llms">API Reference</ApiReferenceLink> for a complete list of supported providers.
          </p>
        </div>
      </section>

      <section id="chat-models" className="space-y-6">
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Chat Models <BetaBadge /></h2>
          <p>
            Chat models are conversation-based models that take a list of messages as input and return a message as output. They are designed for multi-turn conversations and support system prompts, user messages, and assistant messages.
          </p>
          
          <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-4 rounded-r">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  <strong>Tip:</strong> Chat models are the recommended way to build conversational AI applications as they handle conversation state and message formatting for you.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Basic Usage</h3>
          <CodeBlock 
            code={chatModelExample}
            language="python"
            showLineNumbers
          />
          
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium mb-2">Key Features</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start">
                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                  <span>Streaming support for real-time responses</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                  <span>Built-in message history management</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                  <span>Support for system prompts and message roles</span>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Common Use Cases</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>Chat applications</li>
                <li>AI assistants</li>
                <li>Multi-turn conversations</li>
                <li>Interactive applications</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Available Providers</h3>
          <p>
            Chat models are available from various providers with different capabilities and pricing:
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
            {[
              { name: 'OpenAI Chat', status: 'stable', models: ['gpt-4', 'gpt-3.5-turbo'] },
              { name: 'Anthropic', status: 'beta', models: ['claude-2', 'claude-instant'] },
              { name: 'Google', status: 'beta', models: ['chat-bison', 'codechat-bison'] },
            ].map((provider) => (
              <div key={provider.name} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{provider.name}</span>
                  {provider.status === 'beta' && <BetaBadge />}
                </div>
                <div className="text-sm text-muted-foreground">
                  <div className="font-medium mb-1">Models:</div>
                  <div className="flex flex-wrap gap-1">
                    {provider.models.map(model => (
                      <span key={model} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono bg-muted">
                        {model}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          <p className="text-sm text-muted-foreground">
            See the <ApiReferenceLink href="https://api.python.langchain.com/en/latest/langchain_api_reference.html#module-langchain.chat_models">Chat Models API Reference</ApiReferenceLink> for more details.
          </p>
        </div>
      </section>

      <section id="embeddings" className="space-y-6">
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Embeddings</h2>
          <p>
            Embeddings are vector representations of text that capture semantic meaning.
            They are useful for tasks like semantic search, clustering, and classification.
          </p>
        </div>

        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Basic Usage</h3>
          <CodeBlock
            language="python"
            showLineNumbers
            code={`from langchain.embeddings import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create document embeddings
text = "This is a sample document about quantum computing."
vector = embeddings.embed_query(text)
print(f"Vector length: {len(vector)}")
print(f"First 5 dimensions: {vector[:5]}")`}
          />
        </div>

        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Available Embedding Models</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { name: 'OpenAI', models: ['text-embedding-ada-002', 'text-embedding-3-small'] },
              { name: 'HuggingFace', models: ['all-mpnet-base-v2', 'all-MiniLM-L6-v2'] },
              { name: 'Cohere', models: ['embed-english-v3.0', 'embed-multilingual-v3.0'] },
              { name: 'Google', models: ['text-embedding-004', 'text-multilingual-embedding-002'] },
            ].map((provider) => (
              <div key={provider.name} className="border rounded-lg p-4">
                <div className="font-medium mb-2">{provider.name}</div>
                <div className="flex flex-wrap gap-1">
                  {provider.models.map(model => (
                    <span key={model} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono bg-muted">
                      {model}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="custom-models" className="space-y-6">
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Custom Models</h2>
          <p>
            You can create custom model wrappers to integrate with any API or local model
            that follows the same interface as LangChain's built-in models.
          </p>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-4 rounded-r">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-blue-700 dark:text-blue-300">
                <strong>Tip:</strong> When creating custom models, make sure to implement the required interfaces to ensure
                compatibility with the rest of the LangChain ecosystem.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section id="integrations" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Model Integrations</h2>
        <p className="mb-4">
          LangChain supports a wide range of model providers out of the box. Here are some 
          of the most popular ones:
        </p>
        
        <div className="grid gap-4 md:grid-cols-2">
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">OpenAI</h3>
            <p className="text-sm text-muted-foreground mb-2">GPT-4, GPT-3.5, and embeddings</p>
            <pre className="text-sm bg-muted p-2 rounded">pip install langchain-openai</pre>
          </div>
          
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Anthropic</h3>
            <p className="text-sm text-muted-foreground mb-2">Claude models</p>
            <pre className="text-sm bg-muted p-2 rounded">pip install langchain-anthropic</pre>
          </div>
          
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Google</h3>
            <p className="text-sm text-muted-foreground mb-2">Gemini models</p>
            <pre className="text-sm bg-muted p-2 rounded">pip install langchain-google-genai</pre>
          </div>
          
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Hugging Face</h3>
            <p className="text-sm text-muted-foreground mb-2">Open-source models</p>
            <pre className="text-sm bg-muted p-2 rounded">pip install langchain-huggingface</pre>
          </div>
        </div>
      </section>

      <section id="custom-models" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Custom Models</h2>
        <p className="mb-4">
          You can also use LangChain with custom models by implementing the appropriate 
          interfaces. This allows you to integrate any model that can be called via an API 
          or run locally.
        </p>
        
        <Callout type="info">
          <p>
            When implementing custom models, make sure to properly handle errors and timeouts, 
            and consider adding retry logic for production use.
          </p>
        </Callout>
      </section>
    </LangChainDocPage>
  );
}
