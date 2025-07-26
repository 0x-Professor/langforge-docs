import * as React from 'react';
import { cn } from '@/lib/utils';
import { CopyButton } from '@/components/ui/copy-button';

interface CodeBlockProps extends React.HTMLAttributes<HTMLDivElement> {
  /** The code to display */
  code: string;
  /** The language of the code */
  language?: string;
  /** Whether to show line numbers */
  showLineNumbers?: boolean;
  /** Whether to show the copy button */
  showCopyButton?: boolean;
  /** Class name for the container */
  className?: string;
  /** Class name for the pre element */
  preClassName?: string;
  /** Class name for the code element */
  codeClassName?: string;
}

export function CodeBlock({
  code,
  language = 'typescript',
  showLineNumbers = false,
  showCopyButton = true,
  className,
  preClassName,
  codeClassName,
  ...props
}: CodeBlockProps) {
  return (
    <div className={cn('relative group', className)} {...props}>
      {showCopyButton && (
        <CopyButton 
          value={code} 
          className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity" 
        />
      )}
      <pre
        className={cn(
          'relative rounded-md bg-muted p-4 overflow-x-auto text-sm font-mono',
          showLineNumbers && 'line-numbers',
          preClassName
        )}
        data-language={language}
      >
        <code 
          className={cn(
            'block whitespace-pre',
            codeClassName
          )}
        >
          {code}
        </code>
      </pre>
    </div>
  );
}

interface CodeBlockWrapperProps extends React.HTMLAttributes<HTMLDivElement> {
  /** The code to display */
  code: string;
  /** The language of the code */
  language?: string;
  /** Whether to show line numbers */
  showLineNumbers?: boolean;
  /** Whether to show the copy button */
  showCopyButton?: boolean;
  /** Class name for the container */
  className?: string;
  /** Class name for the pre element */
  preClassName?: string;
  /** Class name for the code element */
  codeClassName?: string;
}

export function CodeBlockWrapper({
  children,
  className,
  ...props
}: React.PropsWithChildren<CodeBlockWrapperProps>) {
  return (
    <div className={cn('not-prose my-6', className)}>
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child) && child.type === 'code') {
          const code = child.props.children?.toString() || '';
          const language = child.props.className?.replace('language-', '') || 'typescript';
          
          return (
            <CodeBlock
              code={code.trim()}
              language={language}
              showLineNumbers={true}
              showCopyButton={true}
              {...props}
            />
          );
        }
        return child;
      })}
    </div>
  );
}
