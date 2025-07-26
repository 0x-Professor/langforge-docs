import { DocLayout } from '@/components/docs/DocLayout';
import { CodeBlock } from '@/components/CodeBlock';
import { Callout } from '@/components/docs/DocHeader';

export default function ModelsDocumentation() {
  const toc = [
    { id: 'overview', title: 'Overview', level: 2 },
    { id: 'llms', title: 'LLMs', level: 2 },
    { id: 'chat-models', title: 'Chat Models', level: 2 },
    { id: 'embeddings', title: 'Embeddings', level: 2 },
    { id: 'integrations', title: 'Model Integrations', level: 2 },
    { id: 'custom-models', title: 'Custom Models', level: 2 },
  ];

  const chatModelExample = `from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize chat model
model = init_chat_model(
    model_name="gpt-4",
    model_provider="openai",
    temperature=0.7,
    max_tokens=1000
)

# Create messages
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain quantum computing in simple terms.")
]

# Get response
response = model.invoke(messages)
print(response.content)`;

  const llmExample = `from langchain.llms import OpenAI

# Initialize LLM
llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0.7,
    max_tokens=1000
)

# Generate text
response = llm.invoke("Explain quantum computing in simple terms.")
print(response)`;

  const embeddingsExample = `from langchain.embeddings import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create document embeddings
text = "This is a sample document about quantum computing."
vector = embeddings.embed_query(text)
print(f"Vector length: {len(vector)}")
print(f"First 5 dimensions: {vector[:5]}")`;

  return (
    <DocLayout 
      title="LangChain Models" 
      description="Learn how to work with different types of models in LangChain, including LLMs, chat models, and embeddings."
      toc={toc}
    >
      <section id="overview" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Overview</h2>
        <p className="mb-4">
          LangChain provides a unified interface for working with various language models, 
          making it easy to switch between different providers and model types. The library 
          supports LLMs, chat models, and embeddings.
        </p>
        
        <Callout type="tip">
          <p>
            When choosing between LLMs and chat models, prefer chat models for conversation-based 
            applications as they are specifically designed for multi-turn conversations.
          </p>
        </Callout>
      </section>

      <section id="llms" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">LLMs</h2>
        <p className="mb-4">
          LLMs (Large Language Models) are text-in, text-out models that take a text string as 
          input and return a text string as output. They are great for single-turn tasks like 
          text completion, summarization, and question answering.
        </p>
        
        <CodeBlock 
          code={llmExample} 
          language="python" 
          title="Using LLMs in LangChain"
        />
      </section>

      <section id="chat-models" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Chat Models</h2>
        <p className="mb-4">
          Chat models are conversation-based models that take a list of messages as input and 
          return a message as output. They are designed for multi-turn conversations and 
          maintain conversation history.
        </p>
        
        <CodeBlock 
          code={chatModelExample} 
          language="python" 
          title="Using Chat Models in LangChain"
        />
      </section>

      <section id="embeddings" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Embeddings</h2>
        <p className="mb-4">
          Embeddings are numerical representations of text that capture semantic meaning. 
          They are used for tasks like semantic search, document retrieval, and clustering.
        </p>
        
        <CodeBlock 
          code={embeddingsExample} 
          language="python" 
          title="Using Embeddings in LangChain"
        />
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
        
        <Callout type="important">
          <p>
            When implementing custom models, make sure to properly handle errors and timeouts, 
            and consider adding retry logic for production use.
          </p>
        </Callout>
      </section>
    </DocLayout>
  );
}
