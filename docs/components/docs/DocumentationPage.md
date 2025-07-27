# DocumentationPage

// Lazy load markdown content
const MarkdownContent = lazyLoad(
  () => import('./components/ui/MarkdownContent'),
  {
    loadingFallback: ,
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
  toc?: Array;
  
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
  const { data: markdown, isLoading, error } = useFetch(
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
    const headings: Array = [];
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
     0}
      toc={toc}
      className={className}
    >
      {showBreadcrumbs && breadcrumbs.length > 0 && (
        
          
            {breadcrumbs.map((item, index) => (
              
                {index > 0 && 
/
}
                
{item.title}

              
))}

        
      )}
      
      {contentPath ? (
        
          
{markdown || ''}

        
      ) : (
        children
      )}
      
      {/* Add edit this page link */}
      {contentPath && (
        
          
            
              

            Edit this page on GitHub
          

      )}
    
  );
}

/**
 * Creates a documentation page with the given configuration
 * @param config The configuration for the documentation page
 * @returns A pre-configured DocumentationPage component
 */
export function createDocumentationPage(config: Omit) {
  return function ConfiguredDocumentationPage() {
    return ;
  };
}