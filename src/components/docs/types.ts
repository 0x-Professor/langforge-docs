export interface NavItem {
  title: string;
  href: string;
  description?: string;
  items?: NavItem[];
  external?: boolean;
  disabled?: boolean;
  label?: string;
  icon?: React.ReactNode;
  // For search functionality
  content?: string;
  // For documentation structure
  type?: 'page' | 'section' | 'heading';
  level?: number;
  // For headings in documentation
  headings?: Array<{
    id: string;
    title: string;
    level: number;
  }>;
}

export interface TableOfContentsItem {
  id: string;
  title: string;
  level: number;
  children?: TableOfContentsItem[];
}

export interface DocHeaderProps {
  title: string;
  description?: string;
  tags?: string[];
}

export interface CodeExample {
  title: string;
  description?: string;
  code: string;
  language?: string;
}

export interface ComponentDoc {
  title: string;
  description: string;
  importStatement: string;
  props?: {
    name: string;
    type: string;
    required: boolean;
    default?: string;
    description: string;
  }[];
  examples?: CodeExample[];
  usage: string;
  notes?: string[];
  seeAlso?: { title: string; href: string }[];
}
