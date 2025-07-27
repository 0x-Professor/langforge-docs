# MarkdownContent

import 'highlight.js/styles/github.css';

### MarkdownContentProps

```typescript
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
  components?: Record>;
}
```

**Properties:**

/**
 * A component that renders markdown content with syntax highlighting and custom styling
 */
// Default components for ReactMarkdown
const defaultComponents = {
  // Add any default component overrides here
  // Example:
  // h1: ({ node, ...props }) => ,
};

# function MarkdownContent({
  content,
  className,
  allowHtml = false,
  gfm = true,
  components: customComponents = {},
  ...props
}: MarkdownContentProps) {
  return (
    
      
{content}

    
  );
}

MarkdownContent;