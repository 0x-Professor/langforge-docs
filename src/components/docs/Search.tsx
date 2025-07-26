'use client';

import { useState, useRef, useEffect } from 'react';
import { Search as SearchIcon, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useRouter } from 'next/navigation';
import { useDebounce } from '@/hooks/use-debounce';
import { docsConfig } from '@/config/docs';

interface SearchResult {
  id: string;
  title: string;
  description?: string;
  path: string;
  type: 'page' | 'section' | 'heading';
  level?: number;
}

export function Search() {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const searchContainerRef = useRef<HTMLDivElement>(null);
  
  // Debounce search to avoid too many re-renders
  const debouncedQuery = useDebounce(query, 200);

  // Handle click outside to close search
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Handle keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      // Cmd+K or Ctrl+K to focus search
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault();
        searchInputRef.current?.focus();
        setIsOpen(true);
      }
      
      // Escape to close search
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Search function
  const performSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setResults([]);
      return;
    }

    setIsLoading(true);
    
    try {
      // In a real implementation, you would call a search API here
      // For now, we'll do a simple client-side search
      const normalizedQuery = searchQuery.toLowerCase();
      const allResults: SearchResult[] = [];

      // Search through documentation pages
      docsConfig.sidebarNav.forEach((section) => {
        section.items.forEach((item) => {
          // Match page title or description
          if (
            item.title.toLowerCase().includes(normalizedQuery) ||
            (item.description && item.description.toLowerCase().includes(normalizedQuery))
          ) {
            allResults.push({
              id: item.href,
              title: item.title,
              description: item.description,
              path: item.href,
              type: 'page',
            });
          }

          // Match headings in the page
          if (item.headings) {
            item.headings.forEach((heading) => {
              if (heading.title.toLowerCase().includes(normalizedQuery)) {
                allResults.push({
                  id: `${item.href}#${heading.id}`,
                  title: heading.title,
                  path: `${item.href}#${heading.id}`,
                  type: 'heading',
                  level: heading.level,
                });
              }
            });
          }
        });
      });

      setResults(allResults);
    } catch (error) {
      console.error('Search failed:', error);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Trigger search when query changes (debounced)
  useEffect(() => {
    if (debouncedQuery) {
      performSearch(debouncedQuery);
    } else {
      setResults([]);
    }
  }, [debouncedQuery]);

  const handleResultClick = (path: string) => {
    router.push(path);
    setIsOpen(false);
    setQuery('');
  };

  return (
    <div className="relative w-full max-w-xl" ref={searchContainerRef}>
      <div className="relative">
        <SearchIcon className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          ref={searchInputRef}
          type="text"
          placeholder="Search documentation..."
          className="w-full pl-10 pr-10"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setIsOpen(true)}
        />
        {query && (
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-1 top-1/2 h-8 w-8 -translate-y-1/2"
            onClick={() => setQuery('')}
          >
            <X className="h-4 w-4" />
            <span className="sr-only">Clear search</span>
          </Button>
        )}
        <div className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md border px-1.5 py-0.5 text-xs text-muted-foreground">
          ⌘K
        </div>
      </div>

      {isOpen && (
        <div
          className={cn(
            'absolute left-0 top-full z-50 mt-2 w-full overflow-hidden rounded-md border bg-popover shadow-lg',
            'animate-in fade-in-20 slide-in-from-top-2',
          )}
        >
          {isLoading ? (
            <div className="flex h-32 items-center justify-center">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
            </div>
          ) : results.length > 0 ? (
            <ScrollArea className="max-h-[60vh] overflow-y-auto">
              <div className="divide-y">
                {results.map((result) => (
                  <div
                    key={result.id}
                    className="cursor-pointer p-4 hover:bg-accent hover:text-accent-foreground"
                    onClick={() => handleResultClick(result.path)}
                  >
                    <div className="flex items-start">
                      <div className="mr-3 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
                        {result.type === 'page' ? (
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                            <polyline points="14 2 14 8 20 8" />
                          </svg>
                        ) : (
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="M4 9h16" />
                            <path d="M4 15h16" />
                            <path d="M10 3L8 21" />
                            <path d="M16 3l-2 18" />
                          </svg>
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center">
                          <span className="font-medium">{result.title}</span>
                          {result.level && (
                            <span className="ml-2 text-xs text-muted-foreground">
                              {Array(result.level).fill('#').join('')}
                            </span>
                          )}
                        </div>
                        {result.description && (
                          <p className="mt-1 text-sm text-muted-foreground line-clamp-1">
                            {result.description}
                          </p>
                        )}
                        <div className="mt-1 text-xs text-muted-foreground">
                          {result.path}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          ) : query ? (
            <div className="p-6 text-center text-sm text-muted-foreground">
              No results found for "{query}"
            </div>
          ) : (
            <div className="p-6 text-center text-sm text-muted-foreground">
              Type to search documentation
            </div>
          )}
        </div>
      )}
    </div>
  );
}
