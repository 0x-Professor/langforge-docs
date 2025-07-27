# label

const labelVariants = cva(
  "text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
)

const Label = React.forwardRef,
  React.ComponentPropsWithoutRef &
    VariantProps
>(({ className, ...props }, ref) => (
  
))
Label.displayName = LabelPrimitive.Root.displayName

export { Label }