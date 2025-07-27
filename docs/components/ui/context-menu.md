# context-menu

const ContextMenu = ContextMenuPrimitive.Root

const ContextMenuTrigger = ContextMenuPrimitive.Trigger

const ContextMenuGroup = ContextMenuPrimitive.Group

const ContextMenuPortal = ContextMenuPrimitive.Portal

const ContextMenuSub = ContextMenuPrimitive.Sub

const ContextMenuRadioGroup = ContextMenuPrimitive.RadioGroup

const ContextMenuSubTrigger = React.forwardRef,
  React.ComponentPropsWithoutRef & {
    inset?: boolean
  }
>(({ className, inset, children, ...props }, ref) => (
  
    {children}
    


))
ContextMenuSubTrigger.displayName = ContextMenuPrimitive.SubTrigger.displayName

const ContextMenuSubContent = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
))
ContextMenuSubContent.displayName = ContextMenuPrimitive.SubContent.displayName

const ContextMenuContent = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
    


))
ContextMenuContent.displayName = ContextMenuPrimitive.Content.displayName

const ContextMenuItem = React.forwardRef,
  React.ComponentPropsWithoutRef & {
    inset?: boolean
  }
>(({ className, inset, ...props }, ref) => (
  
))
ContextMenuItem.displayName = ContextMenuPrimitive.Item.displayName

const ContextMenuCheckboxItem = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, children, checked, ...props }, ref) => (
  
    
      
        


    
{children}

))
ContextMenuCheckboxItem.displayName =
  ContextMenuPrimitive.CheckboxItem.displayName

const ContextMenuRadioItem = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, children, ...props }, ref) => (
  
    
      
        


    
{children}

))
ContextMenuRadioItem.displayName = ContextMenuPrimitive.RadioItem.displayName

const ContextMenuLabel = React.forwardRef,
  React.ComponentPropsWithoutRef & {
    inset?: boolean
  }
>(({ className, inset, ...props }, ref) => (
  
))
ContextMenuLabel.displayName = ContextMenuPrimitive.Label.displayName

const ContextMenuSeparator = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
))
ContextMenuSeparator.displayName = ContextMenuPrimitive.Separator.displayName

const ContextMenuShortcut = ({
  className,
  ...props
}: React.HTMLAttributes) => {
  return (
    
  )
}
ContextMenuShortcut.displayName = "ContextMenuShortcut"

export {
  ContextMenu,
  ContextMenuTrigger,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuCheckboxItem,
  ContextMenuRadioItem,
  ContextMenuLabel,
  ContextMenuSeparator,
  ContextMenuShortcut,
  ContextMenuGroup,
  ContextMenuPortal,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuRadioGroup,
}