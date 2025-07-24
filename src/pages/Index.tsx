import { useState } from 'react';
import { Navigation } from '@/components/Navigation';
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
        return <IntroductionSection />;
      case 'langchain':
      case 'langchain-core':
      case 'langchain-modules':
      case 'langchain-integrations':
        return <LangChainSection />;
      case 'langgraph':
      case 'langgraph-basics':
      case 'langgraph-agents':
      case 'langgraph-streaming':
        return <LangGraphSection />;
      case 'langsmith':
      case 'langsmith-tracing':
      case 'langsmith-evaluation':
        return <LangSmithSection />;
      case 'langserve':
      case 'langserve-setup':
      case 'langserve-apis':
        return <LangServeSection />;
      case 'mcp':
      case 'mcp-introduction':
      case 'mcp-server':
      case 'mcp-client':
      case 'mcp-sdks':
        return <MCPSection />;
      case 'agent-architecture':
      case 'multi-agent':
      case 'agent-communication':
        return <AgentArchitectureSection />;
      default:
        return <IntroductionSection />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation activeSection={activeSection} onSectionChange={setActiveSection} />
      
      {/* Main Content */}
      <div className="lg:ml-72">
        <main className="container mx-auto px-4 py-8 max-w-4xl">
          {renderSection()}
        </main>
      </div>
    </div>
  );
};

export default Index;
