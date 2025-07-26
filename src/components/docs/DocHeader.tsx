import { Badge } from '@/components/ui/badge';
import { DocHeaderProps } from './types';
import { 
  Info, 
  AlertTriangle, 
  AlertCircle, 
  Lightbulb, 
  AlertOctagon 
} from 'lucide-react';

export function DocHeader({ title, description, tags = [] }: DocHeaderProps) {
  return (
    <div className="mb-8 space-y-2">
      <h1 className="text-3xl font-bold tracking-tight">{title}</h1>
      
      {description && (
        <p className="text-lg text-muted-foreground">{description}</p>
      )}
      
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-2 pt-2">
          {tags.map((tag) => (
            <Badge key={tag} variant="secondary">
              {tag}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

// Component for displaying version information
export function VersionInfo({ version, date }: { version: string; date: string }) {
  return (
    <div className="mt-4 p-4 bg-muted/50 rounded-lg text-sm">
      <div className="flex items-center gap-2">
        <span className="font-medium">Version:</span>
        <Badge variant="outline">{version}</Badge>
        <span className="text-muted-foreground">•</span>
        <span className="text-muted-foreground">{date}</span>
      </div>
    </div>
  );
}

// Component for displaying related articles
export function RelatedArticles({ items }: { items: { title: string; href: string }[] }) {
  if (items.length === 0) return null;
  
  return (
    <div className="mt-8 border-t pt-6">
      <h3 className="text-sm font-medium text-muted-foreground mb-4">RELATED ARTICLES</h3>
      <ul className="space-y-3">
        {items.map((item) => (
          <li key={item.href}>
            <a 
              href={item.href}
              className="text-sm font-medium text-foreground hover:underline underline-offset-4"
            >
              {item.title} →
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

// Component for displaying a callout/note
export function Callout({ 
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
    <div className={`my-6 rounded-lg border p-4 ${getTypeStyles()}`}>
      <div className="flex items-start gap-3">
        <div className="mt-0.5">
          {type === 'note' && <Info className="h-4 w-4 text-gray-600 dark:text-gray-400" />}
          {type === 'warning' && <AlertTriangle className="h-4 w-4 text-amber-600 dark:text-amber-400" />}
          {type === 'important' && <AlertCircle className="h-4 w-4 text-blue-600 dark:text-blue-400" />}
          {type === 'tip' && <Lightbulb className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />}
          {type === 'caution' && <AlertOctagon className="h-4 w-4 text-red-600 dark:text-red-400" />}
        </div>
        <div className="flex-1">
          <h4 className="mb-2 font-medium">{getTitle()}</h4>
          <div className="prose-sm prose-a:underline-offset-4 prose-a:font-medium dark:prose-invert">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}
