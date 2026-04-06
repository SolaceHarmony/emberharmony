import { MetaProvider, Title, Meta } from "@solidjs/meta"
import { Router } from "@solidjs/router"
import { FileRoutes } from "@solidjs/start/router"
import { Suspense } from "solid-js"
import { Favicon } from "@thesolaceproject/emberharmony-ui/favicon"
import { Font } from "@thesolaceproject/emberharmony-ui/font"
import "@ibm/plex/css/ibm-plex.css"
import "./app.css"

export default function App() {
  return (
    <Router
      explicitLinks={true}
      root={(props) => (
        <MetaProvider>
          <Title>EmberHarmony</Title>
          <Meta name="description" content="EmberHarmony - The open source coding agent." />
          <Favicon />
          <Font />
          <Suspense>{props.children}</Suspense>
        </MetaProvider>
      )}
    >
      <FileRoutes />
    </Router>
  )
}
