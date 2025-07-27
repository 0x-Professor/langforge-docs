# lazyLoad

// Default loading component
export function DefaultLoadingFallback() {
  return (
    
      

  );
}

// Error boundary for lazy-loaded components
class LazyLoadErrorBoundary extends React.Component {
  constructor(props: { fallback: ReactNode; children: ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Lazy loading error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        
Failed to load component. Please try again.

      );
    }

    return this.props.children;
  }
}

// Type for lazy loading options
export interface LazyLoadOptions {
  loadingFallback?: ReactNode;
  errorFallback?: ReactNode;
  /**
   * Time in milliseconds to wait before showing the loading state.
   * This prevents flashing of loading state for very fast loads.
   */
  delay?: number;
}

/**
 * Higher-order component for lazy loading a component with loading and error states
 * @param importFn Function that returns a dynamic import() of the component
 * @param options Options for lazy loading behavior
 * @returns A React component with loading and error states
 */
export function lazyLoad(
  importFn: () => Promise }>,
  options: LazyLoadOptions = {}
) {
  const {
    loadingFallback = ,
    errorFallback,
    delay = 200,
  } = options;

  const LazyComponent = lazy(importFn);

  const LazyWrapper: React.FC = (props) => (
    
      
        
          

      

  );

  return LazyWrapper;
}

// Component to delay showing loading state for fast loads
const DelayedLoading: React.FC = ({ delay, fallback, children }) => {
  const [showLoading, setShowLoading] = React.useState(false);

  React.useEffect(() => {
    const timer = setTimeout(() => {
      setShowLoading(true);
    }, delay);

    return () => clearTimeout(timer);
  }, [delay]);

  // Show children immediately if they load before the delay
  if (!showLoading) {
    return 
{children}
;
  }

  return 
{fallback}
;
};

/**
 * Creates a preload function for a lazy-loaded component
 * @param importFn The same import function passed to lazyLoad
 * @returns A function to preload the component
 */
export function createPreloader(
  importFn: () => Promise }>
) {
  let promise: Promise }> | null = null;
  
  return () => {
    if (!promise) {
      promise = importFn();
    }
    return promise;
  };
}