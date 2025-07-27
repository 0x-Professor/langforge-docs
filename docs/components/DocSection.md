# DocSection

# function DocSection({
  id,
  title,
  description,
  children,
  badges = [],
  externalLinks = [],
  className,
}: DocSectionProps) {
  return (
    
      
        
          
            
{title}

            {badges.length > 0 && (
              
                {badges.map((badge) => (
                  
{badge}

                ))}
              
)}

          
{description}

          
          {externalLinks.length > 0 && (
            
              {externalLinks.map((link) => (
                
                  
                    {link.icon || 
}
                    {link.title}

                
))}

          )}
        
        
        
          
{children}

        

    
  );
}

# function FeatureCard({
  title,
  description,
  icon,
  link,
  className,
  features = [],
  ...props
}: FeatureCardProps) {
  const content = (
    
      
        
          
{icon}

          
{title}

        

      
        
{description}

        {features.length > 0 && (
          
            {features.map((feature, index) => (
              
                
                  

                
{feature}

              
))}

        )}
      
      {link && (
        
          
            
              Learn more 
about {title}

            

        
)}

  );

  if (link) {
    return (
      
{content}

    );
  }

  return content;
}

# function QuickStart({
  title,
  description,
  steps,
  codeExample,
  className,
  ...props
}: QuickStartProps) {
  return (
    
      
        
          
          
{title}

        

      
        
{description}

        
          {steps.map((step, index) => (
            
              
{index + 1}

              
{step}

            
))}

        {codeExample && (
          
            
Example

            
              
{codeExample}

            

        )}
      

  );
}

// Memoize components for better performance
export const MemoizedDocSection = React.memo(DocSection);
export const MemoizedFeatureCard = React.memo(FeatureCard);
export const MemoizedQuickStart = React.memo(QuickStart);