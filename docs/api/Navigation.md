# Navigation

### NavItem

```typescript
interface NavItem {
  id: string;
  title: string;
  icon: React.ReactNode;
  badge?: string;
  children?: NavItem[];
}
```

**Properties:**


const navItems: NavItem[] = [
  {
    id: 'introduction',
    title: 'Introduction',
    icon: ,
  },
  {
    id: 'langchain',
    title: 'LangChain',
    icon: ,
    children: [
      { id: 'langchain-core', title: 'Core Concepts', icon:  },
      { id: 'langchain-modules', title: 'Modules', icon:  },
      { id: 'langchain-integrations', title: 'Integrations', icon:  },
    ]
  },
  {
    id: 'langgraph',
    title: 'LangGraph',
    icon: ,
    badge: 'Agent Framework',
    children: [
      { id: 'langgraph-basics', title: 'Getting Started', icon:  },
      { id: 'langgraph-agents', title: 'Building Agents', icon:  },
      { id: 'langgraph-streaming', title: 'Streaming & State', icon:  },
    ]
  },
  {
    id: 'langsmith',
    title: 'LangSmith',
    icon: ,
    badge: 'Monitoring',
    children: [
      { id: 'langsmith-tracing', title: 'Tracing', icon:  },
      { id: 'langsmith-evaluation', title: 'Evaluation', icon:  },
    ]
  },
  {
    id: 'langserve',
    title: 'LangServe',
    icon: ,
    badge: 'Deployment',
    children: [
      { id: 'langserve-setup', title: 'Setup', icon:  },
      { id: 'langserve-apis', title: 'REST APIs', icon:  },
    ]
  },
  {
    id: 'mcp',
    title: 'Model Context Protocol',
    icon: ,
    badge: 'MCP',
    children: [
      { id: 'mcp-introduction', title: 'Introduction', icon:  },
      { id: 'mcp-server', title: 'MCP Server', icon:  },
      { id: 'mcp-client', title: 'MCP Client', icon:  },
      { id: 'mcp-sdks', title: 'SDKs', icon:  },
    ]
  },
  {
    id: 'agent-architecture',
    title: 'Agent to Agent',
    icon: ,
    badge: 'Advanced',
    children: [
      { id: 'multi-agent', title: 'Multi-Agent Systems', icon:  },
      { id: 'agent-communication', title: 'Communication', icon:  },
    ]
  }
];

### NavigationProps

```typescript
interface NavigationProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}
```

**Properties:**


const NavItemComponent = ({ item, activeSection, onSectionChange, level = 0 }: {
  item: NavItem;
  activeSection: string;
  onSectionChange: (section: string) => void;
  level?: number;
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const isParent = item.children && item.children.length > 0;
  const isActive = activeSection === item.id;
  const hasActiveChild = item.children?.some(child => activeSection === child.id);

  return (
    
       0 ? 'pl-8' : 'pl-4'} ${
          isActive ? 'bg-primary text-primary-foreground font-medium' : ''
        } ${hasActiveChild ? 'border-l-2 border-primary/20' : ''}`}
        onClick={() => {
          if (isParent) {
            setIsExpanded(!isExpanded);
          }
          onSectionChange(item.id);
        }}
      >
        {item.icon}
        
{item.title}

        {item.badge && (
          
{item.badge}

        )}
      
      {item.children && isExpanded && (
        
          {item.children.map((child) => (
            
))}

      )}
    
  );
};

# const Navigation = ({ activeSection, onSectionChange }: NavigationProps) => {
  const NavigationContent = () => (
    
      
        
LangChain Ecosystem

        
          {navItems.map((item) => (
            
))}

      


  );

  return (
    
      {/* Mobile Navigation */}
      
        
          
            
              


          
          
            


        



      {/* Desktop Navigation */}
      
        


    
  );
};