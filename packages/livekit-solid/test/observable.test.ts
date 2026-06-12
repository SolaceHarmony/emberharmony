import { describe, expect, test } from "bun:test"
import { type Accessor, createRoot, createSignal } from "solid-js"
import { Subject, type Observable } from "rxjs"
import { observableState } from "../src/observable"

describe("observableState", () => {
  test("emits observable values and resets when the observable changes", () => {
    const subjectA = new Subject<number>()
    const subjectB = new Subject<number>()
    const [source, setSource] = createSignal(subjectA.asObservable())

    let state!: Accessor<number>
    const dispose = createRoot((d) => {
      state = observableState(source, () => -1)
      return d
    })
    expect(state()).toBe(-1)

    subjectA.next(1)
    expect(state()).toBe(1)
    subjectA.next(2)
    expect(state()).toBe(2)

    // swapping the observable resets to startWith, then tracks the new source
    setSource(subjectB.asObservable())
    expect(state()).toBe(-1)
    subjectA.next(99)
    expect(state()).toBe(-1)
    subjectB.next(7)
    expect(state()).toBe(7)

    dispose()
    subjectB.next(8)
    expect(state()).toBe(7)
  })

  test("handles undefined observables", () => {
    const subject = new Subject<string>()
    const [source, setSource] = createSignal<Observable<string> | undefined>(undefined)

    let state!: Accessor<string>
    const dispose = createRoot((d) => {
      state = observableState(source, () => "start")
      return d
    })
    expect(state()).toBe("start")

    setSource(subject.asObservable())
    subject.next("hello")
    expect(state()).toBe("hello")

    setSource(undefined)
    expect(state()).toBe("start")
    dispose()
  })
})
