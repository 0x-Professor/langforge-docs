# copy-button

interface CopyButtonProps extends ButtonProps {
  /** The value to copy to the clipboard */
  value: string;
  /** Callback when the copy action is completed */
  onCopy?: () => void;
  /** The text to show in the tooltip when not copied */
  copyText?: string;
  /** The text to show in the tooltip when copied */
  copiedText?: string;
  /** The duration in milliseconds to show the copied state */
  timeout?: number;
}

# function CopyButton({
  value,
  onCopy,
  copyText = 'Copy to clipboard',
  copiedText = 'Copied!',
  timeout = 2000,
  className,
  variant = 'ghost',
  size = 'icon',
  ...props
}: CopyButtonProps) {
  const [hasCopied, setHasCopied] = React.useState(false);
  const [isHovered, setIsHovered] = React.useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setHasCopied(true);
      onCopy?.();
      
      const timer = setTimeout(() => {
        setHasCopied(false);
      }, timeout);

      return () => clearTimeout(timer);
    } catch (error) {
      console.error('Failed to copy text:', error);
    }
  };

  return (
     setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={cn(
        'h-8 w-8 p-0 rounded-md',
        'transition-opacity',
        'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
        className
      )}
      aria-label={hasCopied ? copiedText : copyText}
      title={hasCopied ? copiedText : copyText}
      {...props}
    >
      {hasCopied ? (
        
      ) : (
        
      )}
      
{hasCopied ? copiedText : copyText}

    
  );
}