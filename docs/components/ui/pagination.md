# pagination

const Pagination = ({ className, ...props }: React.ComponentProps) => (
  
)
Pagination.displayName = "Pagination"

const PaginationContent = React.forwardRef
>(({ className, ...props }, ref) => (
  
))
PaginationContent.displayName = "PaginationContent"

const PaginationItem = React.forwardRef
>(({ className, ...props }, ref) => (
  
))
PaginationItem.displayName = "PaginationItem"

type PaginationLinkProps = {
  isActive?: boolean
} & Pick &
  React.ComponentProps

const PaginationLink = ({
  className,
  isActive,
  size = "icon",
  ...props
}: PaginationLinkProps) => (
  
)
PaginationLink.displayName = "PaginationLink"

const PaginationPrevious = ({
  className,
  ...props
}: React.ComponentProps) => (
  
    
    
Previous

  
)
PaginationPrevious.displayName = "PaginationPrevious"

const PaginationNext = ({
  className,
  ...props
}: React.ComponentProps) => (
  
    
Next

    


)
PaginationNext.displayName = "PaginationNext"

const PaginationEllipsis = ({
  className,
  ...props
}: React.ComponentProps) => (
  
    
    
More pages

  
)
PaginationEllipsis.displayName = "PaginationEllipsis"

export {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
}