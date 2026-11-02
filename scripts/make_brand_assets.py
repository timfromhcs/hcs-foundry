"""Render the brand assets from one SVG source.

The site, the app and the docs all reference the same mark, so the geometry lives here once and
is rasterised to the sizes social platforms and browsers need.

    python scripts/make_brand_assets.py
"""

from __future__ import annotations

from pathlib import Path

# Palette (kept in sync with docs/BRAND.md and the CSS custom properties)
BG = (11, 13, 16)
SURFACE = (18, 22, 27)
ACCENT = (94, 234, 212)
INK = (232, 237, 243)
MUTED = (152, 163, 176)

ROOT = Path(__file__).resolve().parents[1]
SITE_ASSETS = ROOT / "site" / "assets"
WEB_PUBLIC = ROOT / "web" / "public"

MARK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" role="img" aria-label="HCS Foundry">
  <rect width="48" height="48" rx="12" fill="#0b0d10"/>
  <path d="M13 34V14h5v7.5h12V14h5v20h-5v-7.5H18V34z" fill="#5eead4"/>
  <circle cx="35.5" cy="12.5" r="2.5" fill="#5eead4" opacity="0.55"/>
</svg>
"""

LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 64" role="img" aria-label="HCS Foundry">
  <rect width="320" height="64" rx="14" fill="#0b0d10"/>
  <rect x="8" y="8" width="48" height="48" rx="12" fill="#0f2a2a"/>
  <path d="M21 42V22h5v7h12v-7h5v20h-5v-7H26v7z" fill="#5eead4"/>
  <text x="70" y="33" font-family="Segoe UI, Inter, sans-serif" font-size="21" font-weight="700" fill="#e8edf3">HCS Foundry</text>
  <text x="70" y="49" font-family="Segoe UI, Inter, sans-serif" font-size="11.5" letter-spacing="0.6" fill="#98a3b0">LOCAL AI MODEL LAB</text>
</svg>
"""


def _draw_mark(size: int) -> "object":
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    radius = max(2, size // 4)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=BG)
    # the "H" from the SVG, scaled with simple boxes so no font is required
    margin = size * 0.27
    bar = max(1, int(size * 0.09))
    top, bottom = margin, size - margin
    left, right = margin, size - margin
    mid_top, mid_bottom = top + (bottom - top) * 0.38, top + (bottom - top) * 0.62
    draw.rectangle((left, top, left + bar, bottom), fill=ACCENT)
    draw.rectangle((right - bar, top, right, bottom), fill=ACCENT)
    draw.rectangle((left, mid_top, right, mid_bottom), fill=ACCENT)
    return image


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _og_image(width: int = 1200, height: int = 630):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    for step in range(30, 0, -1):
        alpha = int(2 + step * 0.8)
        draw.ellipse(
            (width * 0.58 - step * 20, -step * 14, width * 0.58 + step * 30, step * 34),
            fill=(BG[0] + alpha // 4, BG[1] + alpha, BG[2] + alpha // 2),
        )
    mark = _draw_mark(132)
    image.paste(mark, (80, 92), mark)
    draw.text((232, 108), "HCS Foundry", fill=INK, font=_font(52, bold=True))
    draw.text((234, 172), "LOCAL AI MODEL LAB", fill=ACCENT, font=_font(20))
    draw.text((80, 292), "Train, merge, quantize and run models", fill=INK, font=_font(40, bold=True))
    draw.text((80, 348), "on your own machine — no cloud, no telemetry.", fill=INK, font=_font(40, bold=True))
    draw.text((80, 432), "INT4 training on disk · GGUF export · local inference · MCP server", fill=ACCENT, font=_font(24))
    draw.line((80, 500, width - 80, 500), fill=(35, 42, 51), width=2)
    draw.text((80, 524), "Windows · Linux · macOS   ·   MIT   ·   timfromhcs & hcsmedia", fill=MUTED, font=_font(22))
    return image


def main() -> int:
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)

    (SITE_ASSETS / "logo-mark.svg").write_text(MARK_SVG, encoding="utf-8")
    (SITE_ASSETS / "logo.svg").write_text(LOGO_SVG, encoding="utf-8")
    (WEB_PUBLIC / "logo-mark.svg").write_text(MARK_SVG, encoding="utf-8")

    sizes = {"favicon-32.png": 32, "favicon-192.png": 192, "apple-touch-icon.png": 180}
    for name, size in sizes.items():
        _draw_mark(size).save(SITE_ASSETS / name)
        _draw_mark(size).save(WEB_PUBLIC / name)

    _og_image().save(SITE_ASSETS / "og.png")
    print("wrote", sorted(path.name for path in SITE_ASSETS.glob("*logo*")), "and favicons + og.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
