import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

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
    return <>{children}</>;
  }

  if (fallback) {
    return <>{fallback}</>;
  }

  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-4 p-8',
        fullScreen ? 'min-h-screen' : 'min-h-[200px]',
        className
      )}
    >
      <Loader2 className="w-8 h-8 animate-spin text-primary" />
      {text && <p className="text-sm text-muted-foreground">{text}</p>}
    </div>
  );
}

interface WithLoadingProps<T> {
  data: T | null | undefined;
  isLoading: boolean;
  error: Error | null;
  children: (data: T) => React.ReactNode;
  loadingFallback?: React.ReactNode;
  errorFallback?: (error: Error) => React.ReactNode;
  emptyFallback?: React.ReactNode;
  validateData?: (data: T) => boolean;
}

export function WithLoading<T>({
  data,
  isLoading,
  error,
  children,
  loadingFallback,
  errorFallback,
  emptyFallback,
  validateData,
}: WithLoadingProps<T>) {
  if (isLoading) {
    return loadingFallback ? (
      <>{loadingFallback}</>
    ) : (
      <LoadingState isLoading={true} text="Loading data..." />
    );
  }

  if (error) {
    return errorFallback ? (
      <>{errorFallback(error)}</>
    ) : (
      <div className="p-4 text-center text-destructive">
        <p>Error: {error.message}</p>
      </div>
    );
  }

  if (!data || (validateData && !validateData(data))) {
    return emptyFallback ? (
      <>{emptyFallback}</>
    ) : (
      <div className="p-4 text-center text-muted-foreground">
        <p>No data available</p>
      </div>
    );
  }

  return <>{children(data)}</>;
}
