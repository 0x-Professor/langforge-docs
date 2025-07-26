import { ReactNode } from 'react';

export interface DocSectionProps {
  id: string;
  title: string;
  description: string;
  children: ReactNode;
  badges?: string[];
  externalLinks?: Array<{
    title: string;
    url: string;
    icon?: ReactNode;
  }>;
  className?: string;
}

export interface FeatureCardProps {
  title: string;
  description: string;
  icon: ReactNode;
  link?: string;
  className?: string;
}

export interface QuickStartProps {
  title: string;
  description: string;
  steps: string[];
  codeExample?: string;
  className?: string;
}

export interface CodeBlockProps {
  title: string;
  language: string;
  code: string;
  className?: string;
  showLineNumbers?: boolean;
  highlightLines?: number[];
}

export interface TabItem {
  value: string;
  label: string;
  icon?: ReactNode;
  content: ReactNode;
}

export interface TabsProps {
  defaultValue: string;
  items: TabItem[];
  className?: string;
  orientation?: 'horizontal' | 'vertical';
}

export interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

export interface LoadingStateProps {
  isLoading: boolean;
  children: ReactNode;
  fallback?: ReactNode;
  className?: string;
}

export interface SearchResult {
  id: string;
  title: string;
  description: string;
  path: string;
  type: 'page' | 'section' | 'api';
  category?: string;
  tags?: string[];
}

export interface NavItem {
  title: string;
  href: string;
  description?: string;
  disabled?: boolean;
  external?: boolean;
  icon?: ReactNode;
  label?: string;
  items?: NavItem[];
}

export interface DocsConfig {
  mainNav: NavItem[];
  sidebarNav: NavItem[];
  footerNav: NavItem[];
  searchIndex: SearchResult[];
}
