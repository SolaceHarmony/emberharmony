// EmberHarmony CLI logo — figlet "standard" font wordmark paired with a
// 6-row ember flame. Directive chars are replaced at draw time by the UI:
//   ^ → top-half block in foreground colour over background (flame highlight)
//   ~ → top-half block in shadow colour (ember shadow)
// All other glyphs render literally. In particular `_` is NOT a directive —
// it is used extensively by the figlet "standard" font and must pass through
// untouched.
export const logo = {
  left: [
    "         ",
    "    ▄    ",
    "   ▟█▙   ",
    "  ▟█^█▙  ",
    "  ▀███▀  ",
    "   ▀~▀   ",
  ],
  right: [
    "  ______           _               _   _                                         ",
    " | ____|_ __ ___ | |__   ___ _ __| | | | __ _ _ __ _ __ ___   ___  _ __  _   _  ",
    " |  _| | '_ ` _ \\| '_ \\ / _ \\ '__| |_| |/ _` | '__| '_ ` _ \\ / _ \\| '_ \\| | | | ",
    " | |___| | | | | | |_) |  __/ |  |  _  | (_| | |  | | | | | | (_) | | | | |_| | ",
    " |_____|_| |_| |_|_.__/ \\___|_|  |_| |_|\\__,_|_|  |_| |_| |_|\\___/|_| |_|\\__, | ",
    "                                                                         |___/  ",
  ],
}

export const marks = "^~"
