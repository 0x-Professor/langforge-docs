# ErrorBoundary

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  className?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { 
      hasError: false,
      error: null 
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { 
      hasError: true, 
      error 
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Error caught by ErrorBoundary:', error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        
          
            
            
Something went wrong

          
          
            
{this.state.error?.message || 'An unexpected error occurred'}

            
              
Try again

              
                
Go to home

              

          

      );
    }

    return this.props.children;
  }
}

export function withErrorBoundary(
  WrappedComponent: React.ComponentType,
  errorBoundaryProps?: Omit
) {
  return function WithErrorBoundary(props: T) {
    return (
      
        

    );
  };
}