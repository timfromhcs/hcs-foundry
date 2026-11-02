"""Render CHANGELOG.md into a static page for the deployed site.

The changelog is part of the product's honesty story — what shipped, what broke, what was not
done — so it is published next to the landing page instead of buried in the repository.

    python scripts/make_site_changelog.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "CHANGELOG.md"
TARGET = ROOT / "site" / "changelog.html"

HEAD = """<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>HCS Foundry — changelog</title>
<meta name="description" content="Every HCS Foundry release with what shipped, what broke and what was deliberately not done." />
<link rel="canonical" href="https://hcs-foundry-site.vercel.app/changelog.html" />
<link rel="icon" href="assets/favicon-32.png" sizes="32x32" />
<meta property="og:title" content="HCS Foundry — changelog" />
<meta property="og:description" content="Every release, with the numbers behind it and the gaps that remain." />
<meta property="og:image" content="https://hcs-foundry-site.vercel.app/assets/og.png" />
<meta name="theme-color" content="#0a0c0f" />
<style>
:root{--bg:#080a0d;--surface:#0f1318;--border:#222a34;--text:#e9eff6;--text-2:#a5b3c2;--accent:#5eead4;
  --font:"Inter","Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif;
  --mono:"JetBrains Mono","Cascadia Code",Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.65 var(--font)}
.wrap{width:min(880px,calc(100% - 40px));margin:0 auto}
header{border-bottom:1px solid var(--border);position:sticky;top:0;background:rgba(8,10,13,.86);backdrop-filter:blur(12px)}
nav{display:flex;align-items:center;gap:14px;height:64px}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.brand{display:flex;align-items:center;gap:10px;color:var(--text);font-weight:700}
.brand img{width:28px;height:28px;border-radius:8px}
nav .spacer{margin-left:auto}
main{padding:40px 0 72px}
h1{font-size:clamp(26px,4vw,38px);letter-spacing:-.02em;margin:0 0 6px}
h2{font-size:20px;margin:36px 0 8px;padding-top:16px;border-top:1px solid var(--border)}
h3{font-size:15px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-2);margin:22px 0 6px}
ul{margin:6px 0 12px;padding-left:20px}
li{margin:5px 0;color:var(--text-2)}
li strong{color:var(--text)}
code{font-family:var(--mono);font-size:13px;background:#0d1116;border:1px solid var(--border);border-radius:6px;padding:1px 5px}
p{color:var(--text-2)}
.lead{color:var(--text-2);margin:0 0 24px}
footer{border-top:1px solid var(--border);padding:22px 0 40px;color:var(--text-2);font-size:14px}
</style>
</head>
<body>
<header><div class="wrap"><nav>
  <a class="brand" href="/"><img src="assets/logo-mark.svg" alt="" /> HCS Foundry</a>
  <span class="spacer"></span>
  <a href="/">Home</a><a href="/#docs">Docs</a>
  <a href="https://github.com/timfromhcs/hcs-foundry">GitHub</a>
</nav></div></header>
<main><div class="wrap">
<h1>Changelog</h1>
<p class="lead">Every release with what shipped, what broke and what was deliberately not done.</p>
"""

FOOT = """</div></main>
<footer><div class="wrap">Local-first · MIT · timfromhcs &amp; hcsmedia ·
<a href="https://github.com/timfromhcs/hcs-foundry/blob/main/CHANGELOG.md">source</a></div></footer>
</body></html>
"""


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def render(markdown: str) -> str:
    lines = markdown.splitlines()
    parts: list[str] = []
    items: list[str] = []

    def flush_items() -> None:
        if items:
            parts.append("<ul>")
            parts.extend(f"<li>{item}</li>" for item in items)
            parts.append("</ul>")
            items.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            flush_items()
            parts.append(f"<h2>{_inline(stripped[3:])}</h2>")
            continue
        if stripped.startswith("### "):
            flush_items()
            parts.append(f"<h3>{_inline(stripped[4:])}</h3>")
            continue
        if stripped.startswith("# "):
            continue  # document title, rendered by the template
        if stripped.startswith("- "):
            items.append(_inline(stripped[2:]))
            continue
        if items and (line.startswith("  ") or line.startswith("\t")):
            items[-1] = f"{items[-1]} {_inline(stripped)}"
            continue
        if not stripped:
            flush_items()
            continue
        flush_items()
        parts.append(f"<p>{_inline(stripped)}</p>")
    flush_items()
    return "\n".join(parts)


def main() -> int:
    body = render(SOURCE.read_text("utf-8"))
    TARGET.write_text(HEAD + body + FOOT, encoding="utf-8", newline="")
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(body)} chars of body)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
