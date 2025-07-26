import { TableOfContentsItem } from './types';
import { cn } from '@/lib/utils';

export function TableOfContents({ items }: { items: TableOfContentsItem[] }) {
  if (items.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">
        No headings in this document
      </div>
    );
  }

  return (
    <nav className="space-y-2">
      <ul className="space-y-2">
        {items.map((item) => (
          <li key={item.id}>
            <a
              href={`#${item.id}`}
              className={cn(
                'block text-sm transition-colors hover:text-foreground',
                item.level === 2 ? 'font-medium' : 'text-muted-foreground',
                item.level === 3 && 'ml-3',
                item.level === 4 && 'ml-6',
                item.level >= 5 && 'ml-9'
              )}
              onClick={(e) => {
                e.preventDefault();
                const element = document.getElementById(item.id);
                if (element) {
                  const headerOffset = 100;
                  const elementPosition = element.getBoundingClientRect().top;
                  const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
                  
                  window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                  });

                  // Update URL without adding to history
                  window.history.pushState(null, '', `#${item.id}`);
                }
              }}
            >
              {item.title}
            </a>
            {item.children && item.children.length > 0 && (
              <ul className="mt-1 space-y-1">
                {item.children.map((child) => (
                  <li key={child.id} className="ml-3">
                    <a
                      href={`#${child.id}`}
                      className="block text-sm text-muted-foreground hover:text-foreground transition-colors"
                      onClick={(e) => {
                        e.preventDefault();
                        const element = document.getElementById(child.id);
                        if (element) {
                          element.scrollIntoView({ behavior: 'smooth' });
                          window.history.pushState(null, '', `#${child.id}`);
                        }
                      }}
                    >
                      {child.title}
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </nav>
  );
}

// Helper function to generate table of contents from markdown content
export function generateToc(content: string): TableOfContentsItem[] {
  const lines = content.split('\n');
  const headings: TableOfContentsItem[] = [];
  
  for (const line of lines) {
    const match = line.match(/^(#{1,6})\s+(.+)$/);
    if (match) {
      const level = match[1].length;
      const title = match[2].replace(/\{.*\}/, '').trim();
      const id = title
        .toLowerCase()
        .replace(/[^\w\s-]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-');
      
      const heading: TableOfContentsItem = { id, title, level };
      
      if (level === 2) {
        headings.push(heading);
      } else if (level > 2 && headings.length > 0) {
        let parent = headings[headings.length - 1];
        
        // Find the appropriate parent heading
        while (parent.level < level - 1) {
          if (!parent.children || parent.children.length === 0) {
            parent.children = [];
            break;
          }
          parent = parent.children[parent.children.length - 1];
        }
        
        if (!parent.children) {
          parent.children = [];
        }
        
        parent.children.push(heading);
      }
    }
  }
  
  return headings;
}
