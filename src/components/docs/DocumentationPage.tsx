import React, { useMemo } from 'react';
import { useLocation } from 'react-router-dom';
import { DocPage } from './DocPage';
import { docsNavItems, getActiveNavItem, getBreadcrumbs } from '@/lib/docs/navigation';
import { useFetch } from '@/lib/docs/fetchDocs';
import { lazyLoad } from '@/lib/docs/lazyLoad';

// Lazy load markdown content
const MarkdownContent = lazyLoad(
  () => import('@/components/ui/MarkdownContent'),
  {
    loadingFallback: <div className="min-h-[50vh]" />,
  }
);

interface DocumentationPageProps {
  /**
   * The title of the documentation page
   */
  title: string;
  
  /**
   * Optional description for the page
   */
  description?: string;
  
  /**
   * Optional path to the markdown file (relative to public/docs/)
   * If provided, this will be loaded and rendered as the page content
   */
  contentPath?: string;
  
  /**
   * Custom content to render instead of loading from a markdown file
   */
  children?: React.ReactNode;
  
  /**
   * Whether to show the table of contents
   * @default true
   */
  showToc?: boolean;
  
  /**
   * Whether to show the breadcrumb navigation
   * @default true
   */
  showBreadcrumbs?: boolean;
  
  /**
   * Custom table of contents items
   * If not provided, will be generated from the markdown content
   */
  toc?: Array<{ id: string; title: string; level: number }>;
  
  /**
   * Additional class names to apply to the page
   */
  className?: string;
}

/**
 * A reusable documentation page component that handles:
 * - Loading states
 * - Error boundaries
 * - Navigation
 * - Table of contents
 * - Markdown rendering
 * - Responsive layout
 */
export function DocumentationPage({
  title,
  description,
  contentPath,
  children,
  showToc = true,
  showBreadcrumbs = true,
  toc: customToc,
  className,
}: DocumentationPageProps) {
  const location = useLocation();
  const { pathname } = location;
  
  // Fetch markdown content if contentPath is provided
  const { data: markdown, isLoading, error } = useFetch<string>(
    async () => {
      if (!contentPath) return '';
      
      const response = await fetch(`/docs/${contentPath}`);
      if (!response.ok) {
        throw new Error(`Failed to load content: ${response.statusText}`);
      }
      return response.text();
    },
    '',
    [contentPath]
  );
  
  // Get navigation context
  const activeNavItem = useMemo(
    () => getActiveNavItem(pathname, docsNavItems),
    [pathname]
  );
  
  const breadcrumbs = useMemo(
    () => (showBreadcrumbs ? getBreadcrumbs(pathname, docsNavItems) : []),
    [pathname, showBreadcrumbs]
  );
  
  // Generate table of contents from markdown if not provided
  const toc = useMemo(() => {
    if (customToc) return customToc;
    if (!markdown) return [];
    
    // Simple regex to extract headings from markdown
    const headingRegex = /^(#{1,6})\s+(.+)$/gm;
    const headings: Array<{ id: string; title: string; level: number }> = [];
    let match;
    
    while ((match = headingRegex.exec(markdown)) !== null) {
      const level = match[1].length;
      const title = match[2].trim();
      const id = title
        .toLowerCase()
        .replace(/[^\w\s-]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-');
      
      headings.push({ id, title, level });
    }
    
    return headings;
  }, [markdown, customToc]);
  
  // Determine the page title and description
  const pageTitle = title || activeNavItem?.title || 'Documentation';
  const pageDescription = description || activeNavItem?.description || '';
  
  return (
    <DocPage
      title={pageTitle}
      description={pageDescription}
      loading={isLoading}
      error={error}
      showToc={showToc && toc.length > 0}
      toc={toc}
      className={className}
    >
      {showBreadcrumbs && breadcrumbs.length > 0 && (
        <nav className="mb-6 text-sm text-muted-foreground">
          <ol className="flex flex-wrap items-center gap-2">
            {breadcrumbs.map((item, index) => (
              <li key={item.href} className="flex items-center">
                {index > 0 && <span className="mx-2">/</span>}
                <a
                  href={item.href}
                  className="hover:text-foreground transition-colors"
                  aria-current={index === breadcrumbs.length - 1 ? 'page' : undefined}
                >
                  {item.title}
                </a>
              </li>
            ))}
          </ol>
        </nav>
      )}
      
      {contentPath ? (
        <div className={className}>
          <MarkdownContent>{markdown || ''}</MarkdownContent>
        </div>
      ) : (
        children
      )}
      
      {/* Add edit this page link */}
      {contentPath && (
        <div className="mt-12 pt-6 border-t border-border">
          <a
            href={`https://github.com/your-org/your-repo/edit/main/public/docs/${contentPath}`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <svg
              className="w-4 h-4 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
              />
            </svg>
            Edit this page on GitHub
          </a>
        </div>
      )}
    </DocPage>
  );
}

/**
 * Creates a documentation page with the given configuration
 * @param config The configuration for the documentation page
 * @returns A pre-configured DocumentationPage component
 */
export function createDocumentationPage(config: Omit<DocumentationPageProps, 'children'>) {
  return function ConfiguredDocumentationPage() {
    return <DocumentationPage {...config} />;
  };
}
