import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ExternalLink, BookOpen, Code, Zap } from 'lucide-react';

interface DocSectionProps {
  id: string;
  title: string;
  description: string;
  badges?: string[];
  children: React.ReactNode;
  externalLinks?: { title: string; url: string }[];
}

export const DocSection = ({ 
  id, 
  title, 
  description, 
  badges = [], 
  children, 
  externalLinks = [] 
}: DocSectionProps) => {
  return (
    <section id={id} className="space-y-6">
      {/* Section Header */}
      <div className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center space-x-3">
            <h1 className="text-3xl font-bold tracking-tight">{title}</h1>
            {badges.map((badge, index) => (
              <Badge key={index} variant="outline" className="text-sm">
                {badge}
              </Badge>
            ))}
          </div>
          <p className="text-xl text-muted-foreground leading-relaxed">
            {description}
          </p>
        </div>

        {/* External Links */}
        {externalLinks.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {externalLinks.map((link, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                asChild
                className="h-8"
              >
                <a href={link.url} target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="w-3 h-3 mr-1" />
                  {link.title}
                </a>
              </Button>
            ))}
          </div>
        )}
      </div>

      {/* Section Content */}
      <div className="space-y-6">
        {children}
      </div>
    </section>
  );
};

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  features?: string[];
}

export const FeatureCard = ({ icon, title, description, features = [] }: FeatureCardProps) => (
  <Card className="h-full shadow-card hover:shadow-lg transition-shadow">
    <CardHeader>
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-primary/10 rounded-lg text-primary">
          {icon}
        </div>
        <CardTitle className="text-lg">{title}</CardTitle>
      </div>
    </CardHeader>
    <CardContent className="space-y-3">
      <p className="text-muted-foreground">{description}</p>
      {features.length > 0 && (
        <ul className="space-y-1">
          {features.map((feature, index) => (
            <li key={index} className="flex items-center text-sm text-muted-foreground">
              <Zap className="w-3 h-3 mr-2 text-primary" />
              {feature}
            </li>
          ))}
        </ul>
      )}
    </CardContent>
  </Card>
);

interface QuickStartProps {
  title: string;
  description: string;
  steps: string[];
  codeExample?: string;
}

export const QuickStart = ({ title, description, steps, codeExample }: QuickStartProps) => (
  <Card className="shadow-card">
    <CardHeader>
      <div className="flex items-center space-x-2">
        <BookOpen className="w-5 h-5 text-primary" />
        <CardTitle>{title}</CardTitle>
      </div>
    </CardHeader>
    <CardContent className="space-y-4">
      <p className="text-muted-foreground">{description}</p>
      <div className="space-y-3">
        {steps.map((step, index) => (
          <div key={index} className="flex items-start space-x-3">
            <div className="flex-shrink-0 w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-sm font-medium">
              {index + 1}
            </div>
            <p className="text-sm text-muted-foreground pt-0.5">{step}</p>
          </div>
        ))}
      </div>
      {codeExample && (
        <div className="mt-4">
          <pre className="bg-gradient-code text-code-foreground p-4 rounded-lg text-sm overflow-x-auto">
            <code>{codeExample}</code>
          </pre>
        </div>
      )}
    </CardContent>
  </Card>
);