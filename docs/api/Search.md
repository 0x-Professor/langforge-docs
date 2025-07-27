# Search

'use client';

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
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const searchInputRef = useRef(null);
  const searchContainerRef = useRef(null);
  
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
    
      
        
         setQuery(e.target.value)}
          onFocus={() => setIsOpen(true)}
        />
        {query && (
           setQuery('')}
          >
            
            
Clear search

          
        )}
        
⌘K

      

      {isOpen && (
        
          {isLoading ? (
            
              

          ) : results.length > 0 ? (
            
              
                {results.map((result) => (
                   handleResultClick(result.path)}
                  >
                    
                      
                        {result.type === 'page' ? (
                          
                            
                            

                        ) : (
                          
                            
                            
                            
                            

                        )}
                      
                      
                        
                          
{result.title}

                          {result.level && (
                            
{Array(result.level).fill('#').join('')}

                          )}
                        
                        {result.description && (
                          
{result.description}

                        )}
                        
{result.path}

                      

                  
))}

            
          ) : query ? (
            
No results found for "{query}"

          ) : (
            
Type to search documentation

          )}
        
)}

  );
}