# use-mobile

const MOBILE_BREAKPOINT = 768

# function useIsMobile() {
  const [isMobile, setIsMobile] = React.useState(undefined)

  React.useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`)
    const onChange = () => {
      setIsMobile(window.innerWidth  mql.removeEventListener("change", onChange)
  }, [])

  return !!isMobile
}