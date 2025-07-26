import React from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import { cn } from '@/lib/utils';
import 'highlight.js/styles/github.css';

interface MarkdownContentProps {
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
// Default components for ReactMarkdown
const defaultComponents = {
  // Add any default component overrides here
  // Example:
  // h1: ({ node, ...props }) => <h1 className="text-3xl font-bold mb-4" {...props} />,
};

export function MarkdownContent({
  content,
  className,
  allowHtml = false,
  gfm = true,
  components: customComponents = {},
  ...props
}: MarkdownContentProps) {
  return (
    <div className={cn('prose dark:prose-invert max-w-none', className)}>
      <ReactMarkdown
        components={{
          ...defaultComponents,
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
