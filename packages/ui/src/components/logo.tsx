export const Mark = (props: { class?: string }) => {
  return (
    <svg
      data-component="logo-mark"
      classList={{ [props.class ?? ""]: !!props.class }}
      viewBox="0 0 64 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Ember flame mark */}
      <path
        data-slot="logo-mark-glow"
        d="M32 72c-12 0-20-8-20-20 0-8 4-16 10-24l10-16 10 16c6 8 10 16 10 24 0 12-8 20-20 20z"
        fill="var(--icon-weak-base)"
      />
      <path
        data-slot="logo-mark-flame"
        d="M32 8L18 32c-4 6-6 12-6 18 0 14 10 22 20 22s20-8 20-22c0-6-2-12-6-18L32 8zm0 56c-8 0-14-6-14-16 0-5 2-10 5-15l9-14 9 14c3 5 5 10 5 15 0 10-6 16-14 16z"
        fill="var(--icon-strong-base)"
      />
      <path
        data-slot="logo-mark-core"
        d="M32 40c-3 0-6 3-6 8s3 10 6 10 6-5 6-10-3-8-6-8z"
        fill="var(--icon-strong-base)"
      />
    </svg>
  )
}

export const Splash = (props: { class?: string }) => {
  return (
    <svg
      data-component="logo-splash"
      classList={{ [props.class ?? ""]: !!props.class }}
      viewBox="0 0 64 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Large ember flame for splash screens */}
      <path
        d="M32 72c-12 0-20-8-20-20 0-8 4-16 10-24l10-16 10 16c6 8 10 16 10 24 0 12-8 20-20 20z"
        fill="var(--icon-base)"
      />
      <path
        d="M32 8L18 32c-4 6-6 12-6 18 0 14 10 22 20 22s20-8 20-22c0-6-2-12-6-18L32 8zm0 56c-8 0-14-6-14-16 0-5 2-10 5-15l9-14 9 14c3 5 5 10 5 15 0 10-6 16-14 16z"
        fill="var(--icon-strong-base)"
      />
      <path
        d="M32 40c-3 0-6 3-6 8s3 10 6 10 6-5 6-10-3-8-6-8z"
        fill="var(--icon-strong-base)"
      />
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
