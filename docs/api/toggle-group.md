# toggle-group

const ToggleGroupContext = React.createContext
>({
  size: "default",
  variant: "default",
})

const ToggleGroup = React.forwardRef,
  React.ComponentPropsWithoutRef &
    VariantProps
>(({ className, variant, size, children, ...props }, ref) => (
  
    
{children}

  
))

ToggleGroup.displayName = ToggleGroupPrimitive.Root.displayName

const ToggleGroupItem = React.forwardRef,
  React.ComponentPropsWithoutRef &
    VariantProps
>(({ className, children, variant, size, ...props }, ref) => {
  const context = React.useContext(ToggleGroupContext)

  return (
    
{children}

  )
})

ToggleGroupItem.displayName = ToggleGroupPrimitive.Item.displayName

export { ToggleGroup, ToggleGroupItem }