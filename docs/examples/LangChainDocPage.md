# LangChainDocPage

interface LangChainDocPageProps {
  /**
   * The title of the documentation page
   */
  title: string;
  
  /**
   * Optional description for the page
   */
  description?: string;
  
  /**
   * Optional path to the markdown file (relative to public/docs/langchain/)
   * If provided, this will be loaded and rendered as the page content
   */
  contentPath?: string;
  
  /**
   * Custom content to render instead of loading from a markdown file
   */
  children?: React.ReactNode;
  
  /**
   * Features to display in a grid at the top of the page
   */
  features?: Array;
  
  /**
   * Code examples to display in tabs
   */
  codeExamples?: Array;
  
  /**
   * Installation instructions
   */
  installation?: {
    npm?: string;
    yarn?: string;
    pip?: string;
    conda?: string;
  };
  
  /**
   * Links to related resources
   */
  resources?: Array;
  
  /**
   * API reference links
   */
  apiReference?: string;
  
  /**
   * Whether to show the table of contents
   * @default true
   */
  showToc?: boolean;
}

/**
 * A template for LangChain documentation pages
 * Provides consistent layout and styling for all documentation pages
 */
export const LangChainDocPage = ({
  title,
  description,
  contentPath,
  children,
  features = [],
  codeExamples = [],
  installation,
  resources = [],
  apiReference,
  showToc = true,
}: LangChainDocPageProps) => {
  const location = useLocation();
  
  // Generate table of contents from the content
  const toc = useMemo(() => {
    // This would be populated from markdown headings or passed as a prop
    return [];
  }, []);
  
  // Get active navigation item for highlighting
  const activeNavItem = useMemo(() => {
    return getActiveNavItem(location.pathname);
  }, [location.pathname]);
  
  // Get breadcrumbs for the current page
  const breadcrumbs = useMemo(() => {
    return getBreadcrumbs(location.pathname);
  }, [location.pathname]);
  const fullContentPath = contentPath ? `langchain/${contentPath}` : undefined;
  
  // Default features if none provided
  const defaultFeatures = features.length > 0 ? features : [
    {
      title: 'Comprehensive Documentation',
      description: 'Detailed guides and API references for all LangChain components',
      icon: ,
    },
    {
      title: 'Code Examples',
      description: 'Ready-to-use code snippets for common use cases',
      icon: ,
    },
    {
      title: 'Best Practices',
      description: 'Learn the recommended patterns and approaches',
      icon: ,
    },
  ];

  return (
    
      {/* Features Grid */}
      {defaultFeatures.length > 0 && (
        
          {defaultFeatures.map((feature, index) => (
            
))}

      )}

      {/* Main Content */}
      
        {contentPath ? (
          
) : (
          children
        )}

      
      {/* Installation Instructions */}
      {installation && (
        
          
Installation

          
            
              {installation.npm && 
npm
}
              {installation.yarn && 
yarn
}
              {installation.pip && 
pip
}
              {installation.conda && 
conda
}
            
            
              {installation.npm && (
                
                  

              )}
              {installation.yarn && (
                
                  

              )}
              {installation.pip && (
                
                  

              )}
              {installation.conda && (
                
                  

              )}
            

        
      )}
      
      {/* Code Examples */}
      {codeExamples.length > 0 && (
        
          
Examples

          
            
              {codeExamples.map((example) => (
                
{example.title}

              ))}
            
            
              {codeExamples.map((example) => (
                
                  {example.description && 
{example.description}
}
                  

              ))}
            

        
      )}
      
      {/* Resources */}
      {(resources.length > 0 || apiReference) && (
        
          
Resources

          
            {apiReference && (
              
                
                  
                  
Code

                
                
API Reference

                

            )}
            
            {resources.map((resource, index) => (
              
                
                  {resource.type === 'github' ? (
                    
                  ) : resource.type === 'example' ? (
                    
                  ) : resource.type === 'blog' ? (
                    
                  ) : (
                    
                  )}
                  
{resource.title}

                  

              
))}

        
)}

  );
};

// Helper component for version badges
export function VersionBadge({ version }: { version: string }) {
  return (
    
v{version}

  );
}

// Helper component for experimental features
export function ExperimentalBadge() {
  return (
    
Experimental

  );
}

// Helper component for beta features
export function BetaBadge() {
  return (
    
Beta

  );
}

// Helper component for deprecation notices
export function DeprecatedNotice({ since, message }: { since?: string; message?: string }) {
  return (
    
{message || 'This feature is no longer recommended for use and may be removed in a future version.'}

  );
}

// Helper component for API reference links
export function ApiReferenceLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    
      {children}
      

  );
}