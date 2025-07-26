import { NavItem } from './types';
import { cn } from '@/lib/utils';
import { useLocation } from 'react-router-dom';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';

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

export function SidebarNav() {
  const location = useLocation();
  const [expandedItems, setExpandedItems] = useState<Record<string, boolean>>({});

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
      <div key={item.href} className={level > 0 ? 'ml-4' : ''}>
        <div className="flex items-center">
          {item.items ? (
            <button
              onClick={() => toggleItem(item.title)}
              className="flex items-center w-full text-left hover:text-foreground"
            >
              {expandedItems[item.title] ? (
                <ChevronDown className="h-4 w-4 mr-1" />
              ) : (
                <ChevronRight className="h-4 w-4 mr-1" />
              )}
              <span className={cn(
                'py-1 text-sm font-medium',
                isActive(item.href) ? 'text-foreground' : 'text-muted-foreground'
              )}>
                {item.title}
              </span>
            </button>
          ) : (
            <a
              href={item.href}
              className={cn(
                'block py-1 text-sm',
                isActive(item.href)
                  ? 'text-foreground font-medium'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              {item.title}
            </a>
          )}
        </div>
        {item.items && expandedItems[item.title] && (
          <div className="mt-1">
            {renderItems(item.items, level + 1)}
          </div>
        )}
      </div>
    ));
  };

  return <nav className="space-y-2">{renderItems(navItems)}</nav>;
}
