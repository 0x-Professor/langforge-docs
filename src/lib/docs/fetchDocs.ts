import { useState, useEffect, useCallback } from 'react';

type FetchFunction<T> = () => Promise<T>;

interface UseFetchResult<T> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

export function useFetch<T>(
  fetchFn: FetchFunction<T>,
  initialData: T | null = null,
  deps: any[] = []
): UseFetchResult<T> {
  const [data, setData] = useState<T | null>(initialData);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await fetchFn();
      setData(result);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError(err instanceof Error ? err : new Error('An unknown error occurred'));
    } finally {
      setIsLoading(false);
    }
  }, [fetchFn]);

  useEffect(() => {
    fetchData();
  }, [fetchData, ...deps]);

  return {
    data,
    isLoading,
    error,
    refetch: fetchData,
  };
}

// Cache for storing fetched data
const cache = new Map<string, any>();

export async function fetchWithCache<T>(
  key: string,
  fetchFn: () => Promise<T>,
  ttl: number = 1000 * 60 * 5 // 5 minutes default TTL
): Promise<T> {
  const now = Date.now();
  const cached = cache.get(key);

  if (cached && (now - cached.timestamp < ttl)) {
    return cached.data;
  }

  try {
    const data = await fetchFn();
    cache.set(key, { data, timestamp: now });
    return data;
  } catch (error) {
    // If there's a cached version, return it even if it's stale
    if (cached) {
      console.warn('Using stale data after fetch error:', error);
      return cached.data;
    }
    throw error;
  }
}

// Preload function for critical data
export function preload<T>(key: string, promise: Promise<T>): void {
  promise.then(data => {
    cache.set(key, { data, timestamp: Date.now() });
  }).catch(console.error);
}

// Clear specific cache entry
export function clearCache(key: string): void {
  cache.delete(key);
}

// Clear all cache
export function clearAllCache(): void {
  cache.clear();
}
