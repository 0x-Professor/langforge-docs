import { NavItem } from '@/components/docs/types';

// Base URL for documentation
const DOCS_BASE_PATH = '/docs';

// Main navigation items
export const mainNavItems: NavItem[] = [
  {
    title: 'Documentation',
    href: `${DOCS_BASE_PATH}/introduction`,
    type: 'page',
  },
  {
    title: 'API Reference',
    href: `${DOCS_BASE_PATH}/api-reference`,
    type: 'page',
  },
  {
    title: 'Guides',
    href: `${DOCS_BASE_PATH}/guides`,
    type: 'page',
  },
  {
    title: 'Examples',
    href: `${DOCS_BASE_PATH}/examples`,
    type: 'page',
  },
];

// Documentation sidebar navigation
export const docsNavItems: NavItem[] = [
  {
    title: 'Getting Started',
    href: `${DOCS_BASE_PATH}/introduction`,
    type: 'section',
    items: [
      {
        title: 'Introduction',
        href: `${DOCS_BASE_PATH}/introduction`,
        type: 'page',
      },
      {
        title: 'Installation',
        href: `${DOCS_BASE_PATH}/installation`,
        type: 'page',
      },
      {
        title: 'Quickstart',
        href: `${DOCS_BASE_PATH}/quickstart`,
        type: 'page',
      },
    ],
  },
  {
    title: 'LangChain',
    href: `${DOCS_BASE_PATH}/langchain`,
    type: 'section',
    items: [
      {
        title: 'Overview',
        href: `${DOCS_BASE_PATH}/langchain/overview`,
        type: 'page',
      },
      {
        title: 'Models',
        href: `${DOCS_BASE_PATH}/langchain/models`,
        type: 'page',
      },
      {
        title: 'Prompts',
        href: `${DOCS_BASE_PATH}/langchain/prompts`,
        type: 'page',
      },
      {
        title: 'Chains',
        href: `${DOCS_BASE_PATH}/langchain/chains`,
        type: 'page',
      },
      {
        title: 'Memory',
        href: `${DOCS_BASE_PATH}/langchain/memory`,
        type: 'page',
      },
      {
        title: 'Indexes',
        href: `${DOCS_BASE_PATH}/langchain/indexes`,
        type: 'page',
      },
      {
        title: 'Agents',
        href: `${DOCS_BASE_PATH}/langchain/agents`,
        type: 'page',
      },
      {
        title: 'Tools',
        href: `${DOCS_BASE_PATH}/langchain/tools`,
        type: 'page',
      },
    ],
  },
  {
    title: 'LangGraph',
    href: `${DOCS_BASE_PATH}/langgraph`,
    type: 'section',
    items: [
      {
        title: 'Overview',
        href: `${DOCS_BASE_PATH}/langgraph/overview`,
        type: 'page',
      },
      {
        title: 'State Management',
        href: `${DOCS_BASE_PATH}/langgraph/state`,
        type: 'page',
      },
      {
        title: 'Agents',
        href: `${DOCS_BASE_PATH}/langgraph/agents`,
        type: 'page',
      },
      {
        title: 'Workflows',
        href: `${DOCS_BASE_PATH}/langgraph/workflows`,
        type: 'page',
      },
    ],
  },
];

/**
 * Flattens the nested navigation structure for search and breadcrumbs
 */
export function flattenNavItems(
  items: NavItem[] = [],
  parentPath = ''
): NavItem[] {
  return items.reduce<NavItem[]>((acc, item) => {
    const fullPath = item.href.startsWith('/')
      ? item.href
      : `${parentPath}/${item.href}`.replace(/\/+/g, '/');

    const flatItem = {
      ...item,
      href: fullPath,
    };

    return [
      ...acc,
      flatItem,
      ...(item.items ? flattenNavItems(item.items, fullPath) : []),
    ];
  }, []);
}

/**
 * Gets the active navigation item based on the current path
 */
export function getActiveNavItem(
  pathname: string,
  items: NavItem[] = []
): NavItem | undefined {
  const flatItems = flattenNavItems(items);
  
  // Try to find an exact match first
  let activeItem = flatItems.find(
    (item) => item.href === pathname
  );

  // If no exact match, try to find a partial match (for nested routes)
  if (!activeItem) {
    activeItem = flatItems
      .sort((a, b) => b.href.length - a.href.length) // Sort by path length (longest first)
      .find((item) => pathname.startsWith(item.href));
  }

  return activeItem;
}

/**
 * Gets the breadcrumb trail for the current path
 */
export function getBreadcrumbs(
  pathname: string,
  items: NavItem[] = []
): NavItem[] {
  const flatItems = flattenNavItems(items);
  const breadcrumbs: NavItem[] = [];
  let currentPath = '';
  
  // Split the path into segments and build up the path
  const segments = pathname
    .split('/')
    .filter(Boolean)
    .map((segment) => `/${segment}`);

  segments.forEach((segment) => {
    currentPath = `${currentPath}${segment}`.replace(/\/+/g, '/');
    
    // Find the item for this path segment
    const item = flatItems.find((item) => item.href === currentPath);
    
    if (item) {
      breadcrumbs.push(item);
    }
  });

  return breadcrumbs;
}

/**
 * Gets the sidebar navigation items for the current path
 */
export function getSidebarNavItems(pathname: string): NavItem[] {
  // Find the active section
  const activeSection = docsNavItems.find(
    (section) =>
      section.href === pathname || pathname.startsWith(`${section.href}/`)
  );

  // If we're in a section, return its items, otherwise return all top-level items
  return activeSection?.items || [];
}

/**
 * Gets the previous and next navigation items for pagination
 */
export function getPagination(
  pathname: string
): { prev?: NavItem; next?: NavItem } {
  const flatItems = flattenNavItems(docsNavItems);
  const currentIndex = flatItems.findIndex((item) => item.href === pathname);

  if (currentIndex === -1) {
    return {};
  }

  return {
    prev: flatItems[currentIndex - 1],
    next: flatItems[currentIndex + 1],
  };
}
