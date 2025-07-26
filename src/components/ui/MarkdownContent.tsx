import React from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import { cn } from '@/lib/utils';

// Import syntax highlighting styles
import 'highlight.js/styles/github-dark.css';

// Custom components for markdown elements
const components = {
  h1: ({ node, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h1 className="text-3xl font-bold mt-8 mb-4" {...props} />
  ),
  h2: ({ node, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h2 
      className="text-2xl font-bold mt-8 mb-4 pt-8 border-t border-border first:mt-0 first:border-t-0 first:pt-0" 
      {...props} 
    />
  ),
  h3: ({ node, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h3 className="text-xl font-semibold mt-6 mb-3" {...props} />
  ),
  h4: ({ node, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h4 className="text-lg font-medium mt-5 mb-2" {...props} />
  ),
  p: ({ node, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
    <p className="my-4 leading-relaxed" {...props} />
  ),
  a: ({ node, href, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a 
      href={href} 
      className="text-primary underline underline-offset-4 hover:text-primary/80 transition-colors"
      target="_blank"
      rel="noopener noreferrer"
      {...props} 
    />
  ),
  ul: ({ node, ordered, ...props }: React.HTMLAttributes<HTMLUListElement> & { ordered?: boolean }) => (
    <ul className={cn("my-4 pl-6 list-disc", ordered && "list-decimal")} {...props} />
  ),
  ol: ({ node, ...props }: React.HTMLAttributes<HTMLOListElement>) => (
    <ol className="my-4 pl-6 list-decimal" {...props} />
  ),
  li: ({ node, ordered, ...props }: React.LiHTMLAttributes<HTMLLIElement> & { ordered?: boolean }) => (
    <li className="my-2 pl-1" {...props} />
  ),
  blockquote: ({ node, ...props }: React.BlockquoteHTMLAttributes<HTMLQuoteElement>) => (
    <blockquote 
      className="border-l-4 border-primary/20 pl-4 py-1 my-4 text-muted-foreground italic" 
      {...props} 
    />
  ),
  code: ({ 
    node, 
    inline, 
    className, 
    children, 
    ...props 
  }: React.HTMLAttributes<HTMLElement> & { 
    inline?: boolean; 
    className?: string;
  }) => {
    const match = /language-(\w+)/.exec(className || '');
    
    return !inline ? (
      <div className="my-4 rounded-lg overflow-hidden border border-border">
        {match && (
          <div className="px-4 py-2 text-xs font-mono bg-muted/50 border-b border-border">
            {match[1]}
          </div>
        )}
        <pre className="p-4 overflow-x-auto bg-muted/10">
          <code 
            className={cn(
              'font-mono text-sm',
              match ? `language-${match[1]}` : '',
              className
            )}
            {...props}
          >
            {children}
          </code>
        </pre>
      </div>
    ) : (
      <code 
        className={cn(
          'px-1.5 py-0.5 rounded bg-muted/50 text-sm font-mono',
          className
        )}
        {...props}
      >
        {children}
      </code>
    );
  },
  table: ({ node, ...props }: React.HTMLAttributes<HTMLTableElement>) => (
    <div className="my-6 overflow-x-auto">
      <table className="w-full border-collapse" {...props} />
    </div>
  ),
  thead: ({ node, ...props }: React.HTMLAttributes<HTMLTableSectionElement>) => (
    <thead className="bg-muted/50" {...props} />
  ),
  th: ({ node, ...props }: React.ThHTMLAttributes<HTMLTableCellElement>) => (
    <th 
      className="px-4 py-2 text-left border border-border font-semibold" 
      {...props} 
    />
  ),
  td: ({ node, ...props }: React.TdHTMLAttributes<HTMLTableCellElement>) => (
    <td 
      className="px-4 py-2 border border-border" 
      {...props} 
    />
  ),
  hr: ({ node, ...props }: React.HTMLAttributes<HTMLHRElement>) => (
    <hr className="my-8 border-border" {...props} />
  ),
  // Add more custom components as needed
};

interface MarkdownContentProps {
  /**
   * The markdown content to render
   */
  content: string;
  
  /**
   * Additional class name for the container
   */
  className?: string;
  
  /**
   * Whether to allow HTML in the markdown
   * @default false
   */
  allowHtml?: boolean;
  
  /**
   * Whether to enable GitHub Flavored Markdown
   * @default true
   */
  gfm?: boolean;
  
  /**
   * Custom components to override the default ones
   */
  components?: Record<string, React.ComponentType<any>>;
}

/**
 * A component that renders markdown content with syntax highlighting and custom styling
 */
export function MarkdownContent({
  content,
  className,
  allowHtml = false,
  gfm = true,
  components: customComponents,
  ...props
}: MarkdownContentProps) {
  return (
    <div className={cn('prose dark:prose-invert max-w-none', className)}>
      <ReactMarkdown
        components={{
          ...components,
          ...customComponents,
        }}
        remarkPlugins={gfm ? [remarkGfm] : []}
        rehypePlugins={[
          ...(allowHtml ? [rehypeRaw] : []),
          rehypeSanitize,
          [rehypeHighlight, { ignoreMissing: true }],
        ]}
        {...props}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

export default MarkdownContent;
