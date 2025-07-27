# form

import {
  Controller,
  ControllerProps,
  FieldPath,
  FieldValues,
  FormProvider,
  useFormContext,
} from "react-hook-form"




const Form = FormProvider

type FormFieldContextValue = FieldPath
> = {
  name: TName
}

const FormFieldContext = React.createContext(
  {} as FormFieldContextValue
)

const FormField =  = FieldPath
>({
  ...props
}: ControllerProps) => {
  return (
    
      


  )
}

const useFormField = () => {
  const fieldContext = React.useContext(FormFieldContext)
  const itemContext = React.useContext(FormItemContext)
  const { getFieldState, formState } = useFormContext()

  const fieldState = getFieldState(fieldContext.name, formState)

  if (!fieldContext) {
    throw new Error("useFormField should be used within ")
  }

  const { id } = itemContext

  return {
    id,
    name: fieldContext.name,
    formItemId: `${id}-form-item`,
    formDescriptionId: `${id}-form-item-description`,
    formMessageId: `${id}-form-item-message`,
    ...fieldState,
  }
}

type FormItemContextValue = {
  id: string
}

const FormItemContext = React.createContext(
  {} as FormItemContextValue
)

const FormItem = React.forwardRef
>(({ className, ...props }, ref) => {
  const id = React.useId()

  return (
    
      


  )
})
FormItem.displayName = "FormItem"

const FormLabel = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ className, ...props }, ref) => {
  const { error, formItemId } = useFormField()

  return (
    
  )
})
FormLabel.displayName = "FormLabel"

const FormControl = React.forwardRef,
  React.ComponentPropsWithoutRef
>(({ ...props }, ref) => {
  const { error, formItemId, formDescriptionId, formMessageId } = useFormField()

  return (
    
  )
})
FormControl.displayName = "FormControl"

const FormDescription = React.forwardRef
>(({ className, ...props }, ref) => {
  const { formDescriptionId } = useFormField()

  return (
    
  )
})
FormDescription.displayName = "FormDescription"

const FormMessage = React.forwardRef
>(({ className, children, ...props }, ref) => {
  const { error, formMessageId } = useFormField()
  const body = error ? String(error?.message) : children

  if (!body) {
    return null
  }

  return (
    
{body}

  )
})
FormMessage.displayName = "FormMessage"

export {
  useFormField,
  Form,
  FormItem,
  FormLabel,
  FormControl,
  FormDescription,
  FormMessage,
  FormField,
}