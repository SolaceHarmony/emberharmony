export const Mark = (props: { class?: string }) => {
  return (
    <svg
      data-component="logo-mark"
      classList={{ [props.class ?? ""]: !!props.class }}
      viewBox="0 0 16 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path data-slot="logo-logo-mark-shadow" d="M12 16H4V8H12V16Z" fill="var(--icon-weak-base)" />
      <path data-slot="logo-logo-mark-o" d="M12 4H4V16H12V4ZM16 20H0V0H16V20Z" fill="var(--icon-strong-base)" />
    </svg>
  )
}

export const Splash = (props: { class?: string }) => {
  return (
    <svg
      data-component="logo-splash"
      classList={{ [props.class ?? ""]: !!props.class }}
      viewBox="0 0 80 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path d="M60 80H20V40H60V80Z" fill="var(--icon-base)" />
      <path d="M60 20H20V80H60V20ZM80 100H0V0H80V100Z" fill="var(--icon-strong-base)" />
    </svg>
  )
}

export const Logo = (props: { class?: string }) => {
  // Placeholder wordmark until a bespoke pixel-art "emberharmony" design lands.
  // Uses a monospace system font to stay visually consistent with the Mark block.
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 360 42"
      fill="none"
      classList={{ [props.class ?? ""]: !!props.class }}
    >
      <text
        x="0"
        y="32"
        font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
        font-size="36"
        font-weight="700"
        letter-spacing="-1"
        fill="var(--icon-strong-base)"
      >
        emberharmony
      </text>
    </svg>
  )
}
