import React, { Suspense } from 'react';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { LoadingState } from '@/components/ui/LoadingState';
import { DocLayout } from './DocLayout';

export interface DocPageProps {
  title: string;
  description?: string;
  children: React.ReactNode;
  loading?: boolean;
  error?: Error | null;
  fallback?: React.ReactNode;
  showToc?: boolean;
  toc?: Array<{ id: string; title: string; level: number }>;
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
        <div className="p-8 text-center">
          <h2 className="text-2xl font-bold text-destructive mb-2">Error Loading Content</h2>
          <p className="text-muted-foreground">{error.message || 'An error occurred while loading this page.'}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
          >
            Retry
          </button>
        </div>
      );
    }

    if (loading) {
      return fallback || (
        <div className="flex items-center justify-center min-h-[50vh]">
          <LoadingState isLoading={true} text="Loading documentation..." />
        </div>
      );
    }

    return children;
  };

  return (
    <ErrorBoundary
      fallback={
        <div className="p-8">
          <h2 className="text-2xl font-bold text-destructive mb-2">Something went wrong</h2>
          <p className="text-muted-foreground mb-4">
            We're sorry, but we encountered an error while rendering this page.
          </p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
          >
            Reload Page
          </button>
        </div>
      }
    >
      <DocLayout title={title} description={description} toc={showToc ? toc : []} {...props}>
        <Suspense
          fallback={
            <div className="flex items-center justify-center min-h-[50vh]">
              <LoadingState isLoading={true} text="Loading content..." />
            </div>
          }
        >
          {renderContent()}
        </Suspense>
      </DocLayout>
    </ErrorBoundary>
  );
}

// Create a higher-order component for documentation pages
export function withDocPage<T extends object>(
  WrappedComponent: React.ComponentType<T>,
  options: Omit<DocPageProps, 'children'> = {}
) {
  return function WithDocPage(props: T) {
    return (
      <DocPage {...options}>
        <WrappedComponent {...(props as any)} />
      </DocPage>
    );
  };
}
