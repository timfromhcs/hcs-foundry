import type { SVGProps } from 'react'

export type IconName =
  | 'grid'
  | 'cpu'
  | 'box'
  | 'chat'
  | 'database'
  | 'fingerprint'
  | 'graduation'
  | 'chart'
  | 'sparkles'
  | 'bot'
  | 'history'
  | 'settings'
  | 'sun'
  | 'moon'
  | 'globe'
  | 'menu'
  | 'close'
  | 'play'
  | 'stop'
  | 'check'
  | 'alert'
  | 'download'
  | 'upload'
  | 'plus'
  | 'trash'
  | 'refresh'
  | 'external'
  | 'chevron'
  | 'copy'
  | 'edit'
  | 'search'
  | 'shield'
  | 'bolt'
  | 'layers'
  | 'terminal'

const PATHS: Record<IconName, string> = {
  grid: 'M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z',
  cpu: 'M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3M6 6h12v12H6zM10 10h4v4h-4z',
  box: 'M21 8l-9-5-9 5 9 5 9-5zM3 8v8l9 5 9-5V8M12 13v8',
  chat: 'M21 12a8 8 0 01-8 8H7l-4 3V12a8 8 0 018-8h2a8 8 0 018 8z',
  database: 'M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3',
  fingerprint: 'M12 4a8 8 0 00-8 8v3M12 4a8 8 0 018 8v2M8 12a4 4 0 018 0v4M12 12v6M8 17v2M16 17v3',
  graduation: 'M3 9l9-4 9 4-9 4-9-4zM7 11v5c0 1.7 2.2 3 5 3s5-1.3 5-3v-5',
  chart: 'M4 20V10M10 20V4M16 20v-7M22 20H2',
  sparkles: 'M12 3l1.8 4.7L18.5 9.5 13.8 11.3 12 16l-1.8-4.7L5.5 9.5l4.7-1.8L12 3zM19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15z',
  bot: 'M6 8h12a2 2 0 012 2v8a2 2 0 01-2 2H6a2 2 0 01-2-2v-8a2 2 0 012-2zM12 4v4M9 13h.01M15 13h.01M9 17h6',
  history: 'M3 12a9 9 0 109-9 9 9 0 00-7.5 4M3 4v5h5M12 8v4l3 2',
  settings: 'M12 15a3 3 0 100-6 3 3 0 000 6zM19.4 15a1.7 1.7 0 00.3 1.9l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.9-.3 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1.1-1.5 1.7 1.7 0 00-1.9.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00.3-1.9 1.7 1.7 0 00-1.5-1H3a2 2 0 110-4h.1A1.7 1.7 0 004.6 9a1.7 1.7 0 00-.3-1.9l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 001.9.3H9a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.9-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.9V9a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z',
  sun: 'M12 17a5 5 0 100-10 5 5 0 000 10zM12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4',
  moon: 'M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z',
  globe: 'M12 21a9 9 0 100-18 9 9 0 000 18zM3 12h18M12 3a14 14 0 010 18M12 3a14 14 0 000 18',
  menu: 'M4 7h16M4 12h16M4 17h16',
  close: 'M6 6l12 12M18 6L6 18',
  play: 'M8 5l11 7-11 7z',
  stop: 'M7 7h10v10H7z',
  check: 'M4 12l5 5L20 6',
  alert: 'M12 3l9 16H3l9-16zM12 9v5M12 17h.01',
  download: 'M12 3v12M7 10l5 5 5-5M4 21h16',
  upload: 'M12 21V9M7 14l5-5 5 5M4 3h16',
  plus: 'M12 5v14M5 12h14',
  trash: 'M4 7h16M9 7V4h6v3M6 7l1 14h10l1-14',
  refresh: 'M20 11a8 8 0 10-2.3 5.7M20 4v7h-7',
  external: 'M14 4h6v6M20 4l-9 9M18 14v6H4V6h6',
  chevron: 'M9 6l6 6-6 6',
  copy: 'M9 9h11v11H9zM5 15V4h11',
  edit: 'M4 20h4l10-10-4-4L4 16v4zM14 6l4 4',
  search: 'M11 19a8 8 0 100-16 8 8 0 000 16zM21 21l-4.3-4.3',
  shield: 'M12 3l8 3v6c0 5-3.4 8-8 9-4.6-1-8-4-8-9V6l8-3z',
  bolt: 'M13 2L4 14h7l-1 8 9-12h-7l1-8z',
  layers: 'M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5M3 18l9 5 9-5',
  terminal: 'M4 5h16v14H4zM7 9l3 3-3 3M13 15h4',
}

export function Icon({ name, size = 18, ...rest }: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      <path d={PATHS[name]} />
    </svg>
  )
}
