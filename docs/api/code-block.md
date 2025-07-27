# code-block

interface CodeBlockProps extends React.HTMLAttributes {
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

# function CodeBlock({
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
    
      {showCopyButton && (
        
      )}
      
        
{code}

      

  );
}

interface CodeBlockWrapperProps extends React.HTMLAttributes {
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

# function CodeBlockWrapper({
  children,
  className,
  ...props
}: React.PropsWithChildren) {
  return (
    
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child) && child.type === 'code') {
          const code = child.props.children?.toString() || '';
          const language = child.props.className?.replace('language-', '') || 'typescript';
          
          return (
            
);
        }
        return child;
      })}

  );
}