# SidebarNav

const navItems: NavItem[] = [
  {
    title: 'Getting Started',
    href: '/docs',
    items: [
      { title: 'Introduction', href: '/docs/introduction' },
      { title: 'Installation', href: '/docs/installation' },
      { title: 'Quickstart', href: '/docs/quickstart' },
    ],
  },
  {
    title: 'LangChain',
    href: '/docs/langchain',
    items: [
      { title: 'Overview', href: '/docs/langchain/overview' },
      { title: 'Models', href: '/docs/langchain/models' },
      { title: 'Prompts', href: '/docs/langchain/prompts' },
      { title: 'Chains', href: '/docs/langchain/chains' },
      { title: 'Memory', href: '/docs/langchain/memory' },
      { title: 'Indexes', href: '/docs/langchain/indexes' },
      { title: 'Agents', href: '/docs/langchain/agents' },
      { title: 'Tools', href: '/docs/langchain/tools' },
    ],
  },
  {
    title: 'LangGraph',
    href: '/docs/langgraph',
    items: [
      { title: 'Overview', href: '/docs/langgraph/overview' },
      { title: 'State Management', href: '/docs/langgraph/state' },
      { title: 'Agents', href: '/docs/langgraph/agents' },
      { title: 'Workflows', href: '/docs/langgraph/workflows' },
    ],
  },
  {
    title: 'LangSmith',
    href: '/docs/langsmith',
    items: [
      { title: 'Overview', href: '/docs/langsmith/overview' },
      { title: 'Tracing', href: '/docs/langsmith/tracing' },
      { title: 'Evaluation', href: '/docs/langsmith/evaluation' },
      { title: 'Monitoring', href: '/docs/langsmith/monitoring' },
    ],
  },
  {
    title: 'Guides',
    href: '/docs/guides',
    items: [
      { title: 'Building a Chatbot', href: '/docs/guides/chatbot' },
      { title: 'RAG Implementation', href: '/docs/guides/rag' },
      { title: 'Deployment', href: '/docs/guides/deployment' },
    ],
  },
];

# function SidebarNav() {
  const location = useLocation();
  const [expandedItems, setExpandedItems] = useState>({});

  const toggleItem = (title: string) => {
    setExpandedItems(prev => ({
      ...prev,
      [title]: !prev[title]
    }));
  };

  const isActive = (href: string) => {
    return location.pathname === href || location.pathname.startsWith(`${href}/`);
  };

  const renderItems = (items: NavItem[], level = 0) => {
    return items.map((item) => (
       0 ? 'ml-4' : ''}>
        
          {item.items ? (
             toggleItem(item.title)}
              className="flex items-center w-full text-left hover:text-foreground"
            >
              {expandedItems[item.title] ? (
                
              ) : (
                
              )}
              
{item.title}

            
          ) : (
            
{item.title}

          )}
        
        {item.items && expandedItems[item.title] && (
          
{renderItems(item.items, level + 1)}

        )}
      
    ));
  };

  return 
{renderItems(navItems)}
;
}