<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="شعار EmberHarmony">
    </picture>
  </a>
</p>
<p align="center">وكيل البرمجة بالذكاء الاصطناعي مفتوح المصدر.</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![واجهة EmberHarmony الطرفية](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## ما هو EmberHarmony؟

EmberHarmony هو وكيل برمجة بالذكاء الاصطناعي مفتوح المصدر يعمل في الطرفية (terminal) لديك. إنه محايد تجاه المزوّدين — استخدمه مع Claude أو OpenAI أو Google أو النماذج المحلية عبر Ollama، أو أي نقطة نهاية متوافقة مع OpenAI. يتميّز بواجهة TUI غنية، ودعم مدمج لـ LSP، وبنية عميل/خادم تتيح لك تشغيله عن بُعد.

### نماذج محلية بدون أي إعداد

يكتشف EmberHarmony تلقائيًا كل نموذج مثبّت في نسخة [Ollama](https://ollama.com) المحلية لديك. لا مفاتيح API، ولا ملفات إعداد، ولا تهيئة يدوية. إذا كان Ollama قيد التشغيل، ستظهر نماذجك:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

بدّل بين النماذج السحابية والمحلية في منتصف المحادثة. شغّل تحليل الشيفرة الحسّاسة بالكامل على جهازك. استخدم النماذج السحابية حين تحتاج إلى قدرات الصدارة، والنماذج المحلية حين تحتاج إلى الخصوصية أو الوصول دون اتصال.

**الفروقات الرئيسية عن أدوات الذكاء الاصطناعي البرمجية الأخرى:**

- **محلي أولًا** — اكتشاف تلقائي لنماذج Ollama، بدون أي إعداد، ودون الحاجة إلى مفاتيح
- مفتوح المصدر بنسبة 100% (MIT)
- غير مرتبط بأي مزوّد واحد — يعمل مع Claude وOpenAI وGoogle وOllama وغيرها
- دعم LSP جاهز للاستخدام للتنقل الذكي في الشيفرة
- واجهة طرفية غنية تتخطى حدود ما هو ممكن في الطرفية
- بنية عميل/خادم — شغّله على جهازك، وتحكّم به من أي مكان

### التثبيت

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### البناء والتثبيت محليًا

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### تطبيق سطح المكتب (إصدار تجريبي)

يتوفّر EmberHarmony أيضًا كتطبيق لسطح المكتب. نزّله مباشرة من [صفحة الإصدارات](https://github.com/SolaceHarmony/emberharmony/releases).

| المنصة                | التنزيل                                    |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`، `.rpm`، أو AppImage               |

#### دليل التثبيت

يحترم سكربت التثبيت ترتيب الأولوية التالي:

1. `$EMBERHARMONY_INSTALL_DIR` — دليل تثبيت مخصّص
2. `$XDG_BIN_DIR` — مسار متوافق مع معيار XDG Base Directory
3. `$HOME/bin` — دليل الملفات التنفيذية القياسي للمستخدم (إن وُجد)
4. `$HOME/.emberharmony/bin` — البديل الافتراضي

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### الوكلاء

يتضمّن EmberHarmony وكيلين مدمجين يمكنك التبديل بينهما بمفتاح `Tab`.

- **build** — الوكيل الافتراضي الكامل الصلاحيات لأعمال التطوير
- **plan** — وكيل للقراءة فقط مخصّص للتحليل واستكشاف الشيفرة
  - يرفض تعديلات الملفات افتراضيًا
  - يطلب الإذن قبل تشغيل أوامر bash
  - مثالي لاستكشاف قواعد شيفرة غير مألوفة أو للتخطيط للتغييرات

يتوفّر أيضًا وكيل فرعي **general** لعمليات البحث المعقّدة والمهام متعددة الخطوات. يُستخدم داخليًا ويمكن استدعاؤه باستخدام `@general` في الرسائل.

### دعم المزوّدين

يعمل EmberHarmony مع أي واجهة API متوافقة مع OpenAI. مع دعم مدمج لـ:

| المزوّد | النماذج | الإعداد المطلوب |
|----------|--------|---------------|
| **Ollama (محلي)** | يُكتشف تلقائيًا من `localhost:11434` | لا شيء — فقط شغّل Ollama |
| **Ollama Cloud** | نماذج Ollama المُستضافة | مفتاح API |
| **Anthropic** | Claude Opus، Sonnet، Haiku | مفتاح API |
| **OpenAI** | GPT-4o، o1، o3 | مفتاح API |
| **Google** | Gemini Pro، Flash | مفتاح API |
| **أي واجهة متوافقة مع OpenAI** | LM Studio، vLLM، Together، Groq، إلخ. | نقطة النهاية + المفتاح |

#### إعداد Ollama المحلي

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

لاستخدام عنوان Ollama غير الافتراضي، أضف إلى `~/.config/emberharmony/config.json`:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### المساهمة

إذا كنت مهتمًا بالمساهمة في EmberHarmony، فيُرجى قراءة [دليل المساهمة](./CONTRIBUTING.md) الخاص بنا قبل إرسال طلب سحب (pull request).

### البناء على EmberHarmony

إذا كنت تعمل على مشروع متعلق بـ EmberHarmony يستخدم "emberharmony" في اسمه، فيُرجى إضافة ملاحظة في ملف README الخاص بك توضّح أنه ليس من بناء The Solace Project وأنه غير مرتبط بنا.

### شكر وتقدير

EmberHarmony هو نسخة معدّلة (fork) من [opencode](https://github.com/anomalyco/opencode) بواسطة فريق [SST](https://sst.dev). نحن ممتنون للغاية لعملهم التأسيسي في بناء وكيل برمجة بالذكاء الاصطناعي مفتوح المصدر استثنائي. يبني هذا المشروع على رؤيتهم وهندستهم.

### المشرف

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**المجتمع:** [Discord](https://discord.gg/EdF8f7JR) | [المشكلات](https://github.com/SolaceHarmony/emberharmony/issues) | [الإصدارات](https://github.com/SolaceHarmony/emberharmony/releases)
