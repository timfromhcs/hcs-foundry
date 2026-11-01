import { NavLink } from 'react-router-dom'
import { useTranslation } from 'react-i18next'

import { Icon, type IconName } from './icons'

const ITEMS: Array<{ to: string; key: string; icon: IconName }> = [
  { to: '/', key: 'nav.overview', icon: 'grid' },
  { to: '/models', key: 'nav.models', icon: 'box' },
  { to: '/hub', key: 'nav.hub', icon: 'download' },
  { to: '/playground', key: 'nav.playground', icon: 'chat' },
]

export function BottomNav({ onMore }: { onMore: () => void }) {
  const { t } = useTranslation()
  return (
    <nav className="bottom-nav" aria-label="Mobile">
      {ITEMS.map((item) => (
        <NavLink key={item.to} to={item.to} end={item.to === '/'} className={({ isActive }) => `bottom-item ${isActive ? 'active' : ''}`}>
          <Icon name={item.icon} size={19} />
          <span>{t(item.key)}</span>
        </NavLink>
      ))}
      <button type="button" className="bottom-item" onClick={onMore}>
        <Icon name="menu" size={19} />
        <span>{t('nav.more')}</span>
      </button>
    </nav>
  )
}
