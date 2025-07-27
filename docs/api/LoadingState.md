# LoadingState

interface LoadingStateProps {
  isLoading: boolean;
  children?: React.ReactNode;
  fallback?: React.ReactNode;
  className?: string;
  text?: string;
  fullScreen?: boolean;
}

export function LoadingState({
  isLoading,
  children,
  fallback,
  className,
  text = 'Loading...',
  fullScreen = false,
}: LoadingStateProps) {
  if (!isLoading) {
    return 
{children}
;
  }

  if (fallback) {
    return 
{fallback}
;
  }

  return (
    
      
      {text && 
{text}
}
    
  );
}

interface WithLoadingProps {
  data: T | null | undefined;
  isLoading: boolean;
  error: Error | null;
  children: (data: T) => React.ReactNode;
  loadingFallback?: React.ReactNode;
  errorFallback?: (error: Error) => React.ReactNode;
  emptyFallback?: React.ReactNode;
  validateData?: (data: T) => boolean;
}

export function WithLoading({
  data,
  isLoading,
  error,
  children,
  loadingFallback,
  errorFallback,
  emptyFallback,
  validateData,
}: WithLoadingProps) {
  if (isLoading) {
    return loadingFallback ? (
      
{loadingFallback}

    ) : (
      
    );
  }

  if (error) {
    return errorFallback ? (
      
{errorFallback(error)}

    ) : (
      
        
Error: {error.message}

      
    );
  }

  if (!data || (validateData && !validateData(data))) {
    return emptyFallback ? (
      
{emptyFallback}

    ) : (
      
        
No data available

      
    );
  }

  return 
{children(data)}
;
}