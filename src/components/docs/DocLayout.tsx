import { ReactNode } from 'react';
import { SidebarNav } from './SidebarNav';
import { DocHeader } from './DocHeader';
import { TableOfContents } from './TableOfContents';

interface DocLayoutProps {
  children: ReactNode;
  title: string;
  description?: string;
  toc?: Array<{ id: string; title: string; level: number }>;
}

export function DocLayout({ children, title, description, toc = [] }: DocLayoutProps) {
  return (
    <div className="flex min-h-screen bg-background">
      {/* Sidebar */}
      <aside className="fixed top-0 left-0 h-full w-64 border-r border-border overflow-y-auto bg-background z-10">
        <div className="p-4">
          <h2 className="text-lg font-semibold mb-4">LangChain Docs</h2>
          <SidebarNav />
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 pl-64 pr-80 pt-4">
        <article className="prose dark:prose-invert max-w-4xl mx-auto py-8 px-4">
          <DocHeader title={title} description={description} />
          <div className="prose dark:prose-invert max-w-none">
            {children}
          </div>
        </article>
      </main>

      {/* Table of Contents */}
      <aside className="fixed top-0 right-0 h-full w-80 border-l border-border overflow-y-auto bg-background z-10">
        <div className="p-4">
          <h3 className="text-sm font-medium mb-4 text-muted-foreground">ON THIS PAGE</h3>
          <TableOfContents items={toc} />
        </div>
      </aside>
    </div>
  );
}
