# DocHeader

import { 
  Info, 
  AlertTriangle, 
  AlertCircle, 
  Lightbulb, 
  AlertOctagon 
} from 'lucide-react';

# function DocHeader({ title, description, tags = [] }: DocHeaderProps) {
  return (
    
      
{title}

      
      {description && (
        
{description}

      )}
      
      {tags.length > 0 && (
        
          {tags.map((tag) => (
            
{tag}

          ))}
        
)}

  );
}

// Component for displaying version information
# function VersionInfo({ version, date }: { version: string; date: string }) {
  return (
    
      
        
Version:

        
{version}

        
•

        
{date}

      

  );
}

// Component for displaying related articles
# function RelatedArticles({ items }: { items: { title: string; href: string }[] }) {
  if (items.length === 0) return null;
  
  return (
    
      
RELATED ARTICLES

      
        {items.map((item) => (
          
            
{item.title} →

          
))}

    
  );
}

// Component for displaying a callout/note
# function Callout({ 
  type = 'note', 
  title, 
  children 
}: { 
  type?: 'note' | 'warning' | 'important' | 'tip' | 'caution';
  title?: string;
  children: React.ReactNode;
}) {
  const getTypeStyles = () => {
    switch (type) {
      case 'warning':
        return 'border-amber-500/20 bg-amber-50 dark:bg-amber-900/20';
      case 'important':
        return 'border-blue-500/20 bg-blue-50 dark:bg-blue-900/20';
      case 'tip':
        return 'border-emerald-500/20 bg-emerald-50 dark:bg-emerald-900/20';
      case 'caution':
        return 'border-red-500/20 bg-red-50 dark:bg-red-900/20';
      case 'note':
      default:
        return 'border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/50';
    }
  };

  const getTitle = () => {
    if (title) return title;
    switch (type) {
      case 'warning':
        return 'Warning';
      case 'important':
        return 'Important';
      case 'tip':
        return 'Tip';
      case 'caution':
        return 'Caution';
      case 'note':
      default:
        return 'Note';
    }
  };

  return (
    
      
        
          {type === 'note' && }
          {type === 'warning' && }
          {type === 'important' && }
          {type === 'tip' && }
          {type === 'caution' && 
}

        
          
{getTitle()}

          
{children}

        

    
  );
}