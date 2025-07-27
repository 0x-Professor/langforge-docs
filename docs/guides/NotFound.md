# NotFound

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    
      
        
404

        
Oops! Page not found

        
Return to Home

      


  );
};

NotFound;