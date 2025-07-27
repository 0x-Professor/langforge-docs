# DocPage

export interface DocPageProps {
  title: string;
  description?: string;
  children: React.ReactNode;
  loading?: boolean;
  error?: Error | null;
  fallback?: React.ReactNode;
  showToc?: boolean;
  toc?: Array;
  className?: string;
}

export function DocPage({
  title,
  description,
  children,
  loading = false,
  error = null,
  fallback = null,
  showToc = true,
  toc = [],
  ...props
}: DocPageProps) {
  const renderContent = () => {
    if (error) {
      return (
        
          
Error Loading Content

          
{error.message || 'An error occurred while loading this page.'}

          
window.location.reload()}
            className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
          >
            Retry

        
      );
    }

    if (loading) {
      return fallback || (
        
          

      );
    }

    return children;
  };

  return (
    
          
Something went wrong

          
We're sorry, but we encountered an error while rendering this page.

          
window.location.reload()}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
          >
            Reload Page

        
      }
    >
      
        
              

          }
        >
          {renderContent()}
        

    
  );
}

// Create a higher-order component for documentation pages
export function withDocPage(
  WrappedComponent: React.ComponentType,
  options: Omit = {}
) {
  return function WithDocPage(props: T) {
    return (
      
        

    );
  };
}