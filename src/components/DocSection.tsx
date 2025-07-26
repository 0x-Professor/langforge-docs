import React from 'react';
import { BookOpen, ExternalLink } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { LoadingState } from '@/components/ui/LoadingState';
import { DocSectionProps, FeatureCardProps, QuickStartProps } from '@/types/docs';

export function DocSection({
  id,
  title,
  description,
  children,
  badges = [],
  externalLinks = [],
  className,
}: DocSectionProps) {
  return (
    <section id={id} className={cn('space-y-6', className)}>
      <ErrorBoundary>
        <div className="space-y-2">
          <div className="flex items-center gap-4">
            <h2 className="text-3xl font-bold tracking-tight">{title}</h2>
            {badges.length > 0 && (
              <div className="flex items-center gap-2">
                {badges.map((badge) => (
                  <Badge key={badge} variant="secondary">
                    {badge}
                  </Badge>
                ))}
              </div>
            )}
          </div>
          <p className="text-lg text-muted-foreground">{description}</p>
          
          {externalLinks.length > 0 && (
            <div className="flex flex-wrap gap-2 pt-2">
              {externalLinks.map((link) => (
                <Button
                  key={link.url}
                  variant="outline"
                  size="sm"
                  className="gap-2"
                  asChild
                >
                  <a 
                    href={link.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    aria-label={`${link.title} (opens in new tab)`}
                  >
                    {link.icon || <ExternalLink className="w-4 h-4" />}
                    {link.title}
                  </a>
                </Button>
              ))}
            </div>
          )}
        </div>
        
        <div className="space-y-8">
          <ErrorBoundary>
            {children}
          </ErrorBoundary>
        </div>
      </ErrorBoundary>
    </section>
  );
}

export function FeatureCard({
  title,
  description,
  icon,
  link,
  className,
  features = [],
  ...props
}: FeatureCardProps) {
  const content = (
    <Card 
      className={cn(
        'h-full transition-shadow hover:shadow-md',
        'group relative overflow-hidden',
        className
      )}
      {...props}
    >
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10 text-primary">
            {icon}
          </div>
          <CardTitle className="text-lg">{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">{description}</p>
        {features.length > 0 && (
          <ul className="space-y-1.5">
            {features.map((feature, index) => (
              <li key={index} className="flex items-start text-sm text-muted-foreground">
                <svg
                  className="w-3.5 h-3.5 mr-2 mt-0.5 text-primary flex-shrink-0"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
                <span>{feature}</span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
      {link && (
        <div className="absolute inset-0 bg-gradient-to-t from-background/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-4">
          <Button 
            variant="outline" 
            size="sm" 
            className="w-full justify-center"
            asChild
          >
            <a href={link}>
              Learn more <span className="sr-only">about {title}</span>
            </a>
          </Button>
        </div>
      )}
    </Card>
  );

  if (link) {
    return (
      <a 
        href={link} 
        className="block focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-lg"
        aria-label={`Learn more about ${title}`}
      >
        {content}
      </a>
    );
  }

  return content;
}

export function QuickStart({
  title,
  description,
  steps,
  codeExample,
  className,
  ...props
}: QuickStartProps) {
  return (
    <Card className={cn('shadow-card overflow-hidden', className)} {...props}>
      <CardHeader className="border-b">
        <div className="flex items-center space-x-2">
          <BookOpen className="w-5 h-5 text-primary" />
          <CardTitle>{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="pt-6 space-y-4">
        <p className="text-muted-foreground">{description}</p>
        <div className="space-y-3">
          {steps.map((step, index) => (
            <div key={index} className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-sm font-medium">
                {index + 1}
              </div>
              <p className="text-sm text-muted-foreground pt-0.5">{step}</p>
            </div>
          ))}
        </div>
        {codeExample && (
          <div className="mt-4 rounded-lg overflow-hidden border">
            <div className="bg-muted/50 px-4 py-2 border-b text-sm font-mono">
              Example
            </div>
            <pre className="bg-gradient-code text-code-foreground p-4 text-sm overflow-x-auto">
              <code>{codeExample}</code>
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Memoize components for better performance
export const MemoizedDocSection = React.memo(DocSection);
export const MemoizedFeatureCard = React.memo(FeatureCard);
export const MemoizedQuickStart = React.memo(QuickStart);