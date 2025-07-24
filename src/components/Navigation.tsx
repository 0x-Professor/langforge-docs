import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { Badge } from '@/components/ui/badge';
import { Menu, Code, Bot, Database, Zap, GitBranch, Settings, ArrowRight } from 'lucide-react';

interface NavItem {
  id: string;
  title: string;
  icon: React.ReactNode;
  badge?: string;
  children?: NavItem[];
}

const navItems: NavItem[] = [
  {
    id: 'introduction',
    title: 'Introduction',
    icon: <ArrowRight className="w-4 h-4" />,
  },
  {
    id: 'langchain',
    title: 'LangChain',
    icon: <Code className="w-4 h-4" />,
    children: [
      { id: 'langchain-core', title: 'Core Concepts', icon: <Code className="w-4 h-4" /> },
      { id: 'langchain-modules', title: 'Modules', icon: <Database className="w-4 h-4" /> },
      { id: 'langchain-integrations', title: 'Integrations', icon: <Settings className="w-4 h-4" /> },
    ]
  },
  {
    id: 'langgraph',
    title: 'LangGraph',
    icon: <GitBranch className="w-4 h-4" />,
    badge: 'Agent Framework',
    children: [
      { id: 'langgraph-basics', title: 'Getting Started', icon: <Zap className="w-4 h-4" /> },
      { id: 'langgraph-agents', title: 'Building Agents', icon: <Bot className="w-4 h-4" /> },
      { id: 'langgraph-streaming', title: 'Streaming & State', icon: <ArrowRight className="w-4 h-4" /> },
    ]
  },
  {
    id: 'langsmith',
    title: 'LangSmith',
    icon: <Settings className="w-4 h-4" />,
    badge: 'Monitoring',
    children: [
      { id: 'langsmith-tracing', title: 'Tracing', icon: <Database className="w-4 h-4" /> },
      { id: 'langsmith-evaluation', title: 'Evaluation', icon: <Zap className="w-4 h-4" /> },
    ]
  },
  {
    id: 'langserve',
    title: 'LangServe',
    icon: <Database className="w-4 h-4" />,
    badge: 'Deployment',
    children: [
      { id: 'langserve-setup', title: 'Setup', icon: <Settings className="w-4 h-4" /> },
      { id: 'langserve-apis', title: 'REST APIs', icon: <Code className="w-4 h-4" /> },
    ]
  },
  {
    id: 'mcp',
    title: 'Model Context Protocol',
    icon: <Bot className="w-4 h-4" />,
    badge: 'MCP',
    children: [
      { id: 'mcp-introduction', title: 'Introduction', icon: <ArrowRight className="w-4 h-4" /> },
      { id: 'mcp-server', title: 'MCP Server', icon: <Database className="w-4 h-4" /> },
      { id: 'mcp-client', title: 'MCP Client', icon: <Code className="w-4 h-4" /> },
      { id: 'mcp-sdks', title: 'SDKs', icon: <Settings className="w-4 h-4" /> },
    ]
  },
  {
    id: 'agent-architecture',
    title: 'Agent to Agent',
    icon: <Bot className="w-4 h-4" />,
    badge: 'Advanced',
    children: [
      { id: 'multi-agent', title: 'Multi-Agent Systems', icon: <Bot className="w-4 h-4" /> },
      { id: 'agent-communication', title: 'Communication', icon: <ArrowRight className="w-4 h-4" /> },
    ]
  }
];

interface NavigationProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

const NavItemComponent = ({ item, activeSection, onSectionChange, level = 0 }: {
  item: NavItem;
  activeSection: string;
  onSectionChange: (section: string) => void;
  level?: number;
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const isActive = activeSection === item.id;
  const hasActiveChild = item.children?.some(child => activeSection === child.id);

  return (
    <div className="space-y-1">
      <Button
        variant={isActive ? "secondary" : "ghost"}
        className={`w-full justify-start font-normal ${level > 0 ? 'pl-8' : 'pl-4'} ${
          isActive ? 'bg-primary text-primary-foreground font-medium' : ''
        } ${hasActiveChild ? 'border-l-2 border-primary/20' : ''}`}
        onClick={() => {
          onSectionChange(item.id);
          if (item.children) setIsExpanded(!isExpanded);
        }}
      >
        {item.icon}
        <span className="ml-2 flex-1 text-left">{item.title}</span>
        {item.badge && (
          <Badge variant="outline" className="ml-auto text-xs">
            {item.badge}
          </Badge>
        )}
      </Button>
      {item.children && isExpanded && (
        <div className="ml-2 space-y-1">
          {item.children.map((child) => (
            <NavItemComponent
              key={child.id}
              item={child}
              activeSection={activeSection}
              onSectionChange={onSectionChange}
              level={level + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export const Navigation = ({ activeSection, onSectionChange }: NavigationProps) => {
  const NavigationContent = () => (
    <div className="space-y-4 py-4">
      <div className="px-3 py-2">
        <h2 className="mb-2 px-4 text-lg font-semibold tracking-tight">
          LangChain Ecosystem
        </h2>
        <div className="space-y-1">
          {navItems.map((item) => (
            <NavItemComponent
              key={item.id}
              item={item}
              activeSection={activeSection}
              onSectionChange={onSectionChange}
            />
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Mobile Navigation */}
      <div className="lg:hidden">
        <Sheet>
          <SheetTrigger asChild>
            <Button variant="outline" size="icon" className="fixed top-4 left-4 z-40">
              <Menu className="h-4 w-4" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-72 p-0">
            <NavigationContent />
          </SheetContent>
        </Sheet>
      </div>

      {/* Desktop Navigation */}
      <div className="hidden lg:block fixed left-0 top-0 z-30 h-full w-72 border-r bg-background">
        <NavigationContent />
      </div>
    </>
  );
};