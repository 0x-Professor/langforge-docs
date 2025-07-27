# toaster

import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from "@/components/ui/toast"

# function Toaster() {
  const { toasts } = useToast()

  return (
    
      {toasts.map(function ({ id, title, description, action, ...props }) {
        return (
          
            
              {title && 
{title}
}
              {description && (
                
{description}

              )}
            
            {action}
            


        )
      })}
      


  )
}