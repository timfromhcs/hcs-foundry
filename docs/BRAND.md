# Brand — HCS Foundry

One mark, one palette, one voice. Everything below is what ships in
`scripts/make_brand_assets.py`, so the site, the app and the docs cannot drift apart.

## Mark

A capital **H** on a dark rounded square, with a small accent dot top-right.

| File | Use |
|---|---|
| `site/assets/logo-mark.svg` | header, favicon source, docs |
| `site/assets/logo.svg` | wordmark lockup (mark + "HCS Foundry" + "LOCAL AI MODEL LAB") |
| `site/assets/favicon-32.png`, `favicon-192.png`, `apple-touch-icon.png` | browser + iOS |
| `site/assets/og.png` | 1200×630 social card |

Regenerate after a palette change:

```bash
python scripts/make_brand_assets.py
```

## Palette

| Token | Value | Use |
|---|---|---|
| background | `#080a0d` | page |
| surface | `#0f1318` / `#151b22` | cards |
| border | `#222a34` / `#303b48` | hairlines |
| text | `#e9eff6` | primary |
| text-2 | `#a5b3c2` | secondary |
| accent | `#5eead4` | primary action, mark, links |
| accent-2 | `#8b8cf9` | secondary accent (charts) |
| ok / warn / danger | `#3ddc97` / `#f5c451` / `#ff6b6b` | status only, always with a label |

The app uses the same values as CSS custom properties in `web/src/styles.css`; the PIL generator
uses `(11, 13, 16)`, `(94, 234, 212)`, `(232, 237, 243)`, `(152, 163, 176)`.

## Type

- UI and docs: **Inter** with `Segoe UI` / system fallbacks; no webfont download.
- Code and metrics: `JetBrains Mono`, `Cascadia Code`, `Consolas`, monospace.

## Voice

- Short sentences, concrete numbers, no exclamation marks.
- Say what was measured and when; if something is not measured, say "not measured".
- Never market a capability the machine does not have. "Refused with numbers" is a feature.

## Don't

- Don't recolour the mark or place it on a light background without the rounded square.
- Don't stretch, rotate or outline it.
- Don't add gradients to the wordmark.
- Don't use the accent colour for body text (contrast fails at small sizes).
