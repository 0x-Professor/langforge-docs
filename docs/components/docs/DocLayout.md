# DocLayout

interface DocLayoutProps {
  children: ReactNode;
  title: string;
  description?: string;
  toc?: Array;
}

export function DocLayout({ children, title, description, toc = [] }: DocLayoutProps) {
  return (
    
      {/* Sidebar */}
      
        
          
LangChain Docs

          

      

      {/* Main Content */}
      
        
          
          
{children}

        

      {/* Table of Contents */}
      
        
          
ON THIS PAGE

          

      

  );
}