<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="โลโก้ EmberHarmony">
    </picture>
  </a>
</p>
<p align="center">เอเจนต์ AI สำหรับเขียนโค้ดแบบโอเพนซอร์ส</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="สถานะการบิลด์" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![EmberHarmony Terminal UI](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## EmberHarmony คืออะไร?

EmberHarmony เป็นเอเจนต์ AI สำหรับเขียนโค้ดแบบโอเพนซอร์สที่ทำงานในเทอร์มินัลของคุณ มันไม่ผูกติดกับผู้ให้บริการรายใดรายหนึ่ง — ใช้งานได้กับ Claude, OpenAI, Google, โมเดลที่รันในเครื่องผ่าน Ollama หรือ endpoint ใด ๆ ที่รองรับ OpenAI มันมาพร้อม TUI ที่อุดมไปด้วยฟีเจอร์ รองรับ LSP ในตัว และสถาปัตยกรรมแบบ client/server ที่ให้คุณควบคุมมันจากระยะไกลได้

### โมเดลในเครื่องที่ไม่ต้องตั้งค่า

EmberHarmony ค้นพบทุกโมเดลที่ติดตั้งในอินสแตนซ์ [Ollama](https://ollama.com) ในเครื่องของคุณโดยอัตโนมัติ ไม่ต้องใช้ API key ไม่ต้องมีไฟล์ตั้งค่า ไม่ต้องตั้งค่าด้วยตนเอง หาก Ollama กำลังทำงานอยู่ โมเดลของคุณจะปรากฏขึ้น:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

สลับระหว่างโมเดลบนคลาวด์และในเครื่องได้ระหว่างบทสนทนา รันการวิเคราะห์โค้ดที่ละเอียดอ่อนทั้งหมดบนเครื่องของคุณเอง ใช้โมเดลบนคลาวด์เมื่อคุณต้องการความสามารถระดับแนวหน้า และใช้โมเดลในเครื่องเมื่อคุณต้องการความเป็นส่วนตัวหรือการเข้าถึงแบบออฟไลน์

**ความแตกต่างสำคัญจากเครื่องมือ AI สำหรับเขียนโค้ดอื่น ๆ:**

- **ให้ความสำคัญกับเครื่องในเครื่องก่อน** — ค้นพบโมเดล Ollama อัตโนมัติ ไม่ต้องตั้งค่า ไม่ต้องใช้ key
- โอเพนซอร์ส 100% (MIT)
- ไม่ผูกติดกับผู้ให้บริการรายใดรายหนึ่ง — ทำงานได้กับ Claude, OpenAI, Google, Ollama และอื่น ๆ
- รองรับ LSP พร้อมใช้งานทันทีสำหรับการนำทางโค้ดอย่างชาญฉลาด
- UI เทอร์มินัลที่อุดมไปด้วยฟีเจอร์ ผลักดันขีดจำกัดของสิ่งที่เป็นไปได้ในเทอร์มินัล
- สถาปัตยกรรมแบบ client/server — รันบนเครื่องของคุณ ควบคุมได้จากทุกที่

### การติดตั้ง

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### การบิลด์ + ติดตั้งในเครื่อง

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### แอปเดสก์ท็อป (เบต้า)

EmberHarmony ยังมีให้ใช้งานในรูปแบบแอปพลิเคชันเดสก์ท็อปอีกด้วย ดาวน์โหลดได้โดยตรงจาก[หน้า releases](https://github.com/SolaceHarmony/emberharmony/releases)

| แพลตฟอร์ม             | ดาวน์โหลด                                  |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm`, หรือ AppImage             |

#### ไดเรกทอรีสำหรับติดตั้ง

สคริปต์ติดตั้งจะยึดตามลำดับความสำคัญดังต่อไปนี้:

1. `$EMBERHARMONY_INSTALL_DIR` — ไดเรกทอรีติดตั้งแบบกำหนดเอง
2. `$XDG_BIN_DIR` — พาธที่สอดคล้องกับ XDG Base Directory
3. `$HOME/bin` — ไดเรกทอรีไบนารีมาตรฐานของผู้ใช้ (หากมีอยู่)
4. `$HOME/.emberharmony/bin` — ค่าสำรองเริ่มต้น

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### เอเจนต์

EmberHarmony มาพร้อมเอเจนต์ในตัวสองตัวที่คุณสลับระหว่างกันได้ด้วยปุ่ม `Tab`

- **build** — เอเจนต์เริ่มต้นที่เข้าถึงได้เต็มรูปแบบสำหรับงานพัฒนา
- **plan** — เอเจนต์แบบอ่านอย่างเดียวสำหรับการวิเคราะห์และสำรวจโค้ด
  - ปฏิเสธการแก้ไขไฟล์โดยค่าเริ่มต้น
  - ขออนุญาตก่อนรันคำสั่ง bash
  - เหมาะสำหรับการสำรวจโค้ดเบสที่ไม่คุ้นเคยหรือการวางแผนการเปลี่ยนแปลง

ยังมีซับเอเจนต์ **general** ให้ใช้งานสำหรับการค้นหาที่ซับซ้อนและงานหลายขั้นตอน มันถูกใช้งานภายในและสามารถเรียกใช้ได้ด้วย `@general` ในข้อความ

### การรองรับผู้ให้บริการ

EmberHarmony ทำงานได้กับ API ใด ๆ ที่รองรับ OpenAI โดยรองรับในตัวสำหรับ:

| ผู้ให้บริการ | โมเดล | การตั้งค่าที่ต้องใช้ |
|----------|--------|---------------|
| **Ollama (ในเครื่อง)** | ค้นพบอัตโนมัติจาก `localhost:11434` | ไม่ต้องมี — แค่รัน Ollama |
| **Ollama Cloud** | โมเดล Ollama แบบโฮสต์ | API key |
| **Anthropic** | Claude Opus, Sonnet, Haiku | API key |
| **OpenAI** | GPT-4o, o1, o3 | API key |
| **Google** | Gemini Pro, Flash | API key |
| **ที่รองรับ OpenAI ใด ๆ** | LM Studio, vLLM, Together, Groq ฯลฯ | Endpoint + key |

#### การตั้งค่า Ollama ในเครื่อง

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

หากต้องการใช้ที่อยู่ Ollama ที่ไม่ใช่ค่าเริ่มต้น ให้เพิ่มลงใน `~/.config/emberharmony/config.json`:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### การมีส่วนร่วม

หากคุณสนใจที่จะมีส่วนร่วมในการพัฒนา EmberHarmony โปรดอ่าน[คู่มือการมีส่วนร่วม](./CONTRIBUTING.md) ของเราก่อนส่ง pull request

### การต่อยอดบน EmberHarmony

หากคุณกำลังทำโปรเจกต์ที่เกี่ยวข้องกับ EmberHarmony ซึ่งใช้คำว่า "emberharmony" ในชื่อ โปรดเพิ่มหมายเหตุใน README ของคุณเพื่อชี้แจงว่ามันไม่ได้สร้างโดย The Solace Project และไม่มีส่วนเกี่ยวข้องกับเรา

### กิตติกรรมประกาศ

EmberHarmony เป็น fork ของ [opencode](https://github.com/anomalyco/opencode) โดยทีม opencode upstream เรารู้สึกซาบซึ้งอย่างยิ่งต่องานรากฐานของพวกเขาในการสร้างเอเจนต์ AI สำหรับเขียนโค้ดแบบโอเพนซอร์สที่ยอดเยี่ยม โปรเจกต์นี้ต่อยอดจากวิสัยทัศน์และงานวิศวกรรมของพวกเขา

### ผู้ดูแล

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**ชุมชน:** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
