# input-otp

const InputOTP = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, containerClassName, ...props }, ref) => (
  
))
InputOTP.displayName = "InputOTP"

const InputOTPGroup = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => (
  
))
InputOTPGroup.displayName = "InputOTPGroup"

const InputOTPSlot = React.forwardRef,
  React.ComponentPropsWithoutRef & { index: number }
>(({ index, className, ...props }, ref) => {
  const inputOTPContext = React.useContext(OTPInputContext)
  const { char, hasFakeCaret, isActive } = inputOTPContext.slots[index]

  return (
    
      {char}
      {hasFakeCaret && (
        
          


      )}
    
  )
})
InputOTPSlot.displayName = "InputOTPSlot"

const InputOTPSeparator = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ ...props }, ref) => (
  
    


))
InputOTPSeparator.displayName = "InputOTPSeparator"

export { InputOTP, InputOTPGroup, InputOTPSlot, InputOTPSeparator }