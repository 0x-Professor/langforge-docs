# scroll-area

const ScrollArea = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, children, ...props }, ref) => (
  
    
{children}

    
    


))
ScrollArea.displayName = ScrollAreaPrimitive.Root.displayName

const ScrollBar = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, orientation = "vertical", ...props }, ref) => (
  
    


))
ScrollBar.displayName = ScrollAreaPrimitive.ScrollAreaScrollbar.displayName

export { ScrollArea, ScrollBar }