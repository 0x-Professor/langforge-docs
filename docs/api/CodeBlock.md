# CodeBlock

### CodeBlockProps

```typescript
interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
  showLineNumbers?: boolean;
  highlightLines?: number[];
}
```

**Properties:**


# const CodeBlock = ({ 
  code, 
  language = 'python', 
  title, 
  showLineNumbers = true,
  highlightLines = []
}: CodeBlockProps) => {
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      toast({
        title: "Copied!",
        description: "Code copied to clipboard",
        duration: 2000,
      });
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      toast({
        title: "Failed to copy",
        description: "Please copy manually",
        variant: "destructive",
      });
    }
  };

  const lines = code.split('\n');

  return (
    
      {title && (
        
          
            
{title}

            
{language}

          
          
            {copied ?  : 
}

        
      )}
      
        
          
            {showLineNumbers ? (
              
                {lines.map((line, index) => (
                  
                    
{index + 1}

                    
{line || ' '}

                  
))}

            ) : (
              code
            )}
          


        {!title && (
          
            {copied ?  : 
}

        )}
      


  );
};

# const InlineCode = ({ children }: { children: React.ReactNode }) => (
  
{children}

);