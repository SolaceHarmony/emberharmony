/**
 * Ported from @livekit/components-react `hooks/internal/useObservableState.ts` (Apache-2.0).
 * React useState/useEffect translated to a Solid signal with automatic (re)subscription.
 */
import { type Accessor, createRenderEffect, createSignal, onCleanup } from "solid-js"
import type { Observable } from "rxjs"

export function observableState<T>(
  observable: Accessor<Observable<T> | undefined>,
  startWith: Accessor<T>,
): Accessor<T> {
  const [state, setState] = createSignal<T>(startWith())
  createRenderEffect(() => {
    const obs = observable()
    setState(() => startWith())
    if (!obs) return
    const subscription = obs.subscribe((value) => setState(() => value))
    onCleanup(() => subscription.unsubscribe())
  })
  return state
}
