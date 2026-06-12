/**
 * Ported from @livekit/components-react `context/room-context.ts` (Apache-2.0).
 */
import { createContext, useContext, type ParentProps } from "solid-js"
import type { Room } from "livekit-client"

const RoomCtx = createContext<Room>()

export function RoomContext(props: ParentProps<{ room: Room }>) {
  return <RoomCtx.Provider value={props.room}>{props.children}</RoomCtx.Provider>
}

export function useMaybeRoomContext(): Room | undefined {
  return useContext(RoomCtx)
}

export function useRoomContext(): Room {
  const room = useContext(RoomCtx)
  if (!room) throw new Error("useRoomContext must be used within a RoomContext provider")
  return room
}

export function useEnsureRoom(room?: Room): Room {
  const contextRoom = useMaybeRoomContext()
  const r = room ?? contextRoom
  if (!r) throw new Error("No room provided and no room context available")
  return r
}
