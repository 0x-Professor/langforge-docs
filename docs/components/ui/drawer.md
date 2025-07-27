# drawer

const Drawer = ({
  shouldScaleBackground = true,
  ...props
}: React.ComponentProps) => (
  
)
Drawer.displayName = "Drawer"

const DrawerTrigger = DrawerPrimitive.Trigger

const DrawerPortal = DrawerPrimitive.Portal

const DrawerClose = DrawerPrimitive.Close

const DrawerOverlay = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
))
DrawerOverlay.displayName = DrawerPrimitive.Overlay.displayName

const DrawerContent = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, children, ...props }, ref) => (
  
    
    
      
{children}

  
))
DrawerContent.displayName = "DrawerContent"

const DrawerHeader = ({
  className,
  ...props
}: React.HTMLAttributes) => (
  
)
DrawerHeader.displayName = "DrawerHeader"

const DrawerFooter = ({
  className,
  ...props
}: React.HTMLAttributes) => (
  
)
DrawerFooter.displayName = "DrawerFooter"

const DrawerTitle = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
))
DrawerTitle.displayName = DrawerPrimitive.Title.displayName

const DrawerDescription = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
))
DrawerDescription.displayName = DrawerPrimitive.Description.displayName

export {
  Drawer,
  DrawerPortal,
  DrawerOverlay,
  DrawerTrigger,
  DrawerClose,
  DrawerContent,
  DrawerHeader,
  DrawerFooter,
  DrawerTitle,
  DrawerDescription,
}