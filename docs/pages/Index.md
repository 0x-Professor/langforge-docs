# Index

import { 
  IntroductionSection, 
  LangChainSection, 
  LangGraphSection, 
  LangSmithSection,
  LangServeSection,
  MCPSection, 
  AgentArchitectureSection 
} from '@/components/sections';

const Index = () => {
  const [activeSection, setActiveSection] = useState('introduction');

  const renderSection = () => {
    switch (activeSection) {
      case 'introduction':
        return ;
      case 'langchain':
      case 'langchain-core':
      case 'langchain-modules':
      case 'langchain-integrations':
        return ;
      case 'langgraph':
      case 'langgraph-basics':
      case 'langgraph-agents':
      case 'langgraph-streaming':
        return ;
      case 'langsmith':
      case 'langsmith-tracing':
      case 'langsmith-evaluation':
        return ;
      case 'langserve':
      case 'langserve-setup':
      case 'langserve-apis':
        return ;
      case 'mcp':
      case 'mcp-introduction':
      case 'mcp-server':
      case 'mcp-client':
      case 'mcp-sdks':
        return ;
      case 'agent-architecture':
      case 'multi-agent':
      case 'agent-communication':
        return ;
      default:
        return ;
    }
  };

  return (
    
      
      
      {/* Main Content */}
      
        
{renderSection()}

      


  );
};

Index;