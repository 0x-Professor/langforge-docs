import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Copy, Check } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
  showLineNumbers?: boolean;
  highlightLines?: number[];
}

export const CodeBlock = ({ 
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
    <div className="relative group">
      {title && (
        <div className="flex items-center justify-between px-4 py-2 bg-muted border border-border rounded-t-lg">
          <div className="flex items-center space-x-2">
            <span className="font-medium text-sm">{title}</span>
            <Badge variant="outline" className="text-xs">
              {language}
            </Badge>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={copyToClipboard}
            className="opacity-0 group-hover:opacity-100 transition-opacity"
          >
            {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
          </Button>
        </div>
      )}
      <div className={`relative ${title ? 'rounded-t-none' : ''} rounded-lg overflow-hidden shadow-code`}>
        <pre className={`bg-gradient-code text-code-foreground p-4 overflow-x-auto ${title ? 'rounded-t-none' : ''}`}>
          <code className="text-sm font-mono">
            {showLineNumbers ? (
              <div className="table w-full">
                {lines.map((line, index) => (
                  <div 
                    key={index} 
                    className={`table-row ${highlightLines.includes(index + 1) ? 'bg-primary-muted' : ''}`}
                  >
                    <span className="table-cell text-right pr-4 text-code-comment select-none w-8">
                      {index + 1}
                    </span>
                    <span className="table-cell">{line || ' '}</span>
                  </div>
                ))}
              </div>
            ) : (
              code
            )}
          </code>
        </pre>
        {!title && (
          <Button
            variant="ghost"
            size="sm"
            onClick={copyToClipboard}
            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-code-background/80 hover:bg-code-background text-code-foreground"
          >
            {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
          </Button>
        )}
      </div>
    </div>
  );
};

export const InlineCode = ({ children }: { children: React.ReactNode }) => (
  <code className="relative rounded px-[0.3rem] py-[0.2rem] bg-code-background text-code-foreground text-sm font-mono">
    {children}
  </code>
);