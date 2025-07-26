import React, { useMemo } from 'react';
import { DocumentationPage } from '@/components/docs/DocumentationPage';
import { FeatureCard } from '@/components/DocSection';
import { CodeBlock } from '@/components/ui/CodeBlock';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Callout } from '@/components/docs/DocHeader';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ExternalLink, Github, Code, BookOpen, Zap, Cpu, MessageSquare, Database, Settings } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import { getActiveNavItem, getBreadcrumbs } from '@/lib/docs/navigation';
import { DocLayout } from '@/components/docs/DocLayout';

interface LangChainDocPageProps {
  /**
   * The title of the documentation page
   */
  title: string;
  
  /**
   * Optional description for the page
   */
  description?: string;
  
  /**
   * Optional path to the markdown file (relative to public/docs/langchain/)
   * If provided, this will be loaded and rendered as the page content
   */
  contentPath?: string;
  
  /**
   * Custom content to render instead of loading from a markdown file
   */
  children?: React.ReactNode;
  
  /**
   * Features to display in a grid at the top of the page
   */
  features?: Array<{
    title: string;
    description: string;
    icon: React.ReactNode;
    link?: string;
  }>;
  
  /**
   * Code examples to display in tabs
   */
  codeExamples?: Array<{
    title: string;
    description?: string;
    code: string;
    language?: string;
  }>;
  
  /**
   * Installation instructions
   */
  installation?: {
    npm?: string;
    yarn?: string;
    pip?: string;
    conda?: string;
  };
  
  /**
   * Links to related resources
   */
  resources?: Array<{
    title: string;
    href: string;
    type?: 'github' | 'documentation' | 'example' | 'blog';
  }>;
  
  /**
   * API reference links
   */
  apiReference?: string;
  
  /**
   * Whether to show the table of contents
   * @default true
   */
  showToc?: boolean;
}

/**
 * A template for LangChain documentation pages
 * Provides consistent layout and styling for all documentation pages
 */
export const LangChainDocPage = ({
  title,
  description,
  contentPath,
  children,
  features = [],
  codeExamples = [],
  installation,
  resources = [],
  apiReference,
  showToc = true,
}: LangChainDocPageProps) => {
  const location = useLocation();
  
  // Generate table of contents from the content
  const toc = useMemo(() => {
    // This would be populated from markdown headings or passed as a prop
    return [];
  }, []);
  
  // Get active navigation item for highlighting
  const activeNavItem = useMemo(() => {
    return getActiveNavItem(location.pathname);
  }, [location.pathname]);
  
  // Get breadcrumbs for the current page
  const breadcrumbs = useMemo(() => {
    return getBreadcrumbs(location.pathname);
  }, [location.pathname]);
  const fullContentPath = contentPath ? `langchain/${contentPath}` : undefined;
  
  // Default features if none provided
  const defaultFeatures = features.length > 0 ? features : [
    {
      title: 'Comprehensive Documentation',
      description: 'Detailed guides and API references for all LangChain components',
      icon: <BookOpen className="h-6 w-6 text-blue-500" />,
    },
    {
      title: 'Code Examples',
      description: 'Ready-to-use code snippets for common use cases',
      icon: <Code className="h-6 w-6 text-green-500" />,
    },
    {
      title: 'Best Practices',
      description: 'Learn the recommended patterns and approaches',
      icon: <Zap className="h-6 w-6 text-yellow-500" />,
    },
  ];

  return (
    <DocLayout title={title} description={description} toc={toc}>
      {/* Features Grid */}
      {defaultFeatures.length > 0 && (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 mb-12">
          {defaultFeatures.map((feature, index) => (
            <FeatureCard
              key={index}
              title={feature.title}
              description={feature.description}
              icon={feature.icon}
              link={feature.link}
              className="h-full"
            />
          ))}
        </div>
      )}

      {/* Main Content */}
      <div className="prose dark:prose-invert max-w-none">
        {contentPath ? (
          <DocumentationPage
            title={title}
            description={description}
            contentPath={contentPath}
            showToc={showToc}
          />
        ) : (
          children
        )}
      </div>
      
      {/* Installation Instructions */}
      {installation && (
        <div className="my-8">
          <h2 className="text-2xl font-bold mb-4">Installation</h2>
          <Tabs defaultValue={installation.npm ? 'npm' : installation.pip ? 'pip' : 'yarn'}>
            <TabsList>
              {installation.npm && <TabsTrigger value="npm">npm</TabsTrigger>}
              {installation.yarn && <TabsTrigger value="yarn">yarn</TabsTrigger>}
              {installation.pip && <TabsTrigger value="pip">pip</TabsTrigger>}
              {installation.conda && <TabsTrigger value="conda">conda</TabsTrigger>}
            </TabsList>
            <div className="mt-4">
              {installation.npm && (
                <TabsContent value="npm">
                  <CodeBlock
                    code={`npm install ${installation.npm}`}
                    language="bash"
                    showLineNumbers={false}
                  />
                </TabsContent>
              )}
              {installation.yarn && (
                <TabsContent value="yarn">
                  <CodeBlock
                    code={`yarn add ${installation.yarn}`}
                    language="bash"
                    showLineNumbers={false}
                  />
                </TabsContent>
              )}
              {installation.pip && (
                <TabsContent value="pip">
                  <CodeBlock
                    code={`pip install ${installation.pip}`}
                    language="bash"
                    showLineNumbers={false}
                  />
                </TabsContent>
              )}
              {installation.conda && (
                <TabsContent value="conda">
                  <CodeBlock
                    code={`conda install -c conda-forge ${installation.conda}`}
                    language="bash"
                    showLineNumbers={false}
                  />
                </TabsContent>
              )}
            </div>
          </Tabs>
        </div>
      )}
      
      {/* Code Examples */}
      {codeExamples.length > 0 && (
        <div className="my-8">
          <h2 className="text-2xl font-bold mb-4">Examples</h2>
          <Tabs defaultValue={codeExamples[0].title.toLowerCase().replace(/\s+/g, '-')}>
            <TabsList>
              {codeExamples.map((example) => (
                <TabsTrigger 
                  key={example.title}
                  value={example.title.toLowerCase().replace(/\s+/g, '-')}
                >
                  {example.title}
                </TabsTrigger>
              ))}
            </TabsList>
            <div className="mt-4">
              {codeExamples.map((example) => (
                <TabsContent 
                  key={example.title}
                  value={example.title.toLowerCase().replace(/\s+/g, '-')}
                  className="space-y-4"
                >
                  {example.description && <p>{example.description}</p>}
                  <CodeBlock
                    code={example.code}
                    language={example.language || 'python'}
                  />
                </TabsContent>
              ))}
            </div>
          </Tabs>
        </div>
      )}
      
      {/* Resources */}
      {(resources.length > 0 || apiReference) && (
        <div className="mt-12 pt-6 border-t border-border">
          <h2 className="text-xl font-semibold mb-4">Resources</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {apiReference && (
              <a
                href={apiReference}
                target="_blank"
                rel="noopener noreferrer"
                className="p-4 border rounded-lg hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center space-x-2">
                  <Code className="h-4 w-4" />
                  <span>Code</span>
                </div>
                <span className="font-medium">API Reference</span>
                <ExternalLink className="w-4 h-4 ml-auto text-muted-foreground" />
              </a>
            )}
            
            {resources.map((resource, index) => (
              <a
                key={index}
                href={resource.href}
                target="_blank"
                rel="noopener noreferrer"
                className="p-4 border rounded-lg hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  {resource.type === 'github' ? (
                    <Github className="w-5 h-5 text-foreground" />
                  ) : resource.type === 'example' ? (
                    <Code2 className="w-5 h-5 text-foreground" />
                  ) : resource.type === 'blog' ? (
                    <Newspaper className="w-5 h-5 text-foreground" />
                  ) : (
                    <BookOpen className="w-5 h-5 text-foreground" />
                  )}
                  <span className="font-medium">{resource.title}</span>
                  <ExternalLink className="w-4 h-4 ml-auto text-muted-foreground" />
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
    </DocLayout>
  );
};

// Helper component for version badges
export function VersionBadge({ version }: { version: string }) {
  return (
    <Badge variant="outline" className="ml-2 align-middle">
      v{version}
    </Badge>
  );
}

// Helper component for experimental features
export function ExperimentalBadge() {
  return (
    <Badge variant="secondary" className="ml-2">
      Experimental
    </Badge>
  );
}

// Helper component for beta features
export function BetaBadge() {
  return (
    <Badge variant="outline" className="ml-2 bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400">
      Beta
    </Badge>
  );
}

// Helper component for deprecation notices
export function DeprecatedNotice({ since, message }: { since?: string; message?: string }) {
  return (
    <Callout type="warning" title={message || 'This feature is deprecated' + (since ? ` since v${since}` : '')}>
      {message || 'This feature is no longer recommended for use and may be removed in a future version.'}
    </Callout>
  );
}

// Helper component for API reference links
export function ApiReferenceLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center text-primary hover:underline underline-offset-4"
    >
      {children}
      <ExternalLink className="w-3.5 h-3.5 ml-1" />
    </a>
  );
}
