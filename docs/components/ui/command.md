# command

const Command = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
))
Command.displayName = CommandPrimitive.displayName

interface CommandDialogProps extends DialogProps {}

const CommandDialog = ({ children, ...props }: CommandDialogProps) => {
  return (
    
      
        
{children}

      


  )
}

const CommandInput = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
    
    


))

CommandInput.displayName = CommandPrimitive.Input.displayName

const CommandList = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
))

CommandList.displayName = CommandPrimitive.List.displayName

const CommandEmpty = React.forwardRef,
  React.ComponentPropsWithoutRef
>((props, ref) => (
  
))

CommandEmpty.displayName = CommandPrimitive.Empty.displayName

const CommandGroup = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
))

CommandGroup.displayName = CommandPrimitive.Group.displayName

const CommandSeparator = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
))
CommandSeparator.displayName = CommandPrimitive.Separator.displayName

const CommandItem = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
))

CommandItem.displayName = CommandPrimitive.Item.displayName

const CommandShortcut = ({
  className,
  ...props
}: React.HTMLAttributes) => {
  return (
    
  )
}
CommandShortcut.displayName = "CommandShortcut"

export {
  Command,
  CommandDialog,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandShortcut,
  CommandSeparator,
}