import { NavItem } from '@/components/docs/types';

export const DOCS_BASE_PATH = '/docs';

export interface SearchableItem {
  id: string;
  title: string;
  description?: string;
  content?: string;
  path: string;
  type: 'page' | 'section' | 'heading';
  level?: number;
}

export const docsConfig: {
  mainNav: NavItem[];
  sidebarNav: NavItem[];
  searchIndex: SearchableItem[];
} = {
  mainNav: [
    {
      title: 'Documentation',
      href: '/docs',
    },
    {
      title: 'API Reference',
      href: '/docs/api',
    },
    {
      title: 'Guides',
      href: '/guides',
    },
  ],
  searchIndex: [
    {
      id: 'introduction',
      title: 'Introduction',
      description: 'Learn the basics of LangChain and get started quickly.',
      path: '/docs',
      type: 'page',
    },
  ],
  sidebarNav: [
    {
      title: 'Getting Started',
      href: '/docs',
      items: [
        {
          title: 'Introduction',
          href: '/docs',
          description: 'Learn the basics of LangChain and get started quickly.',
        },
        {
          title: 'Installation',
          href: '/docs/installation',
        },
        {
          title: 'Quickstart',
          href: '/docs/quickstart',
        },
      ],
    },
    {
      title: 'LangChain',
      href: '/docs/langchain',
      items: [
        {
          title: 'Overview',
          href: '/docs/langchain/overview',
        },
        {
          title: 'Models',
          href: '/docs/langchain/models',
        },
        {
          title: 'Prompts',
          href: '/docs/langchain/prompts',
        },
        {
          title: 'Chains',
          href: '/docs/langchain/chains',
        },
        {
          title: 'Memory',
          href: '/docs/langchain/memory',
        },
        {
          title: 'Indexes',
          href: '/docs/langchain/indexes',
        },
        {
          title: 'Agents',
          href: '/docs/langchain/agents',
        },
        {
          title: 'Tools',
          href: '/docs/langchain/tools',
        },
      ],
    },
    {
      title: 'LangGraph',
      href: '/docs/langgraph',
      items: [
        {
          title: 'Overview',
          href: '/docs/langgraph/overview',
        },
        {
          title: 'State Management',
          href: '/docs/langgraph/state',
        },
        {
          title: 'Agents',
          href: '/docs/langgraph/agents',
        },
        {
          title: 'Workflows',
          href: '/docs/langgraph/workflows',
        },
      ],
    },
    {
      title: 'LangSmith',
      href: '/docs/langsmith',
      items: [
        {
          title: 'Overview',
          href: '/docs/langsmith/overview',
        },
        {
          title: 'Tracing',
          href: '/docs/langsmith/tracing',
        },
        {
          title: 'Evaluation',
          href: '/docs/langsmith/evaluation',
        },
        {
          title: 'Monitoring',
          href: '/docs/langsmith/monitoring',
        },
      ],
    },
  ],
};

export const DOCS_LINKS = {
  models: '/docs/langchain/models',
  prompts: '/docs/langchain/prompts',
  chains: '/docs/langchain/chains',
  memory: '/docs/langchain/memory',
  indexes: '/docs/langchain/indexes',
  agents: '/docs/langchain/agents',
  tools: '/docs/langchain/tools',
};

export const MODEL_PROVIDERS = [
  {
    name: 'OpenAI',
    models: ['gpt-4', 'gpt-3.5-turbo', 'text-embedding-ada-002'],
    install: 'pip install langchain-openai',
  },
  {
    name: 'Anthropic',
    models: ['claude-3-opus', 'claude-3-sonnet', 'claude-2.1'],
    install: 'pip install langchain-anthropic',
  },
  {
    name: 'Google',
    models: ['gemini-pro', 'gemini-ultra', 'text-embedding-004'],
    install: 'pip install langchain-google-genai',
  },
  {
    name: 'Hugging Face',
    models: ['BLOOM', 'T5', 'GPT-2', 'BART'],
    install: 'pip install langchain-huggingface',
  },
  {
    name: 'Cohere',
    models: ['command', 'embed-english-v3.0'],
    install: 'pip install langchain-cohere',
  },
];

export const EXAMPLE_CODE = {
  basicChat: `from langchain.chat_models import init_chat_model
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
print(response.content)`,
  
  chainExample: `from langchain_core.prompts import ChatPromptTemplate
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

print(result)`,
};

export const FEATURES = [
  {
    title: 'Unified Interface',
    description: 'Consistent API across different model providers and types.',
    icon: 'code',
  },
  {
    title: 'Extensible',
    description: 'Easily add support for new models and providers.',
    icon: 'layers',
  },
  {
    title: 'Production Ready',
    description: 'Built with performance and scalability in mind.',
    icon: 'zap',
  },
  {
    title: 'Type Safe',
    description: 'Full TypeScript support for better developer experience.',
    icon: 'type',
  },
];
