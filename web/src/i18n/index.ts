import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

import { de } from './locales/de'
import { en } from './locales/en'
import { fr } from './locales/fr'

export const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'de', label: 'Deutsch' },
  { code: 'fr', label: 'Français' },
] as const

export type LanguageCode = (typeof LANGUAGES)[number]['code']

const stored = (() => {
  try {
    return localStorage.getItem('foundry.lang')
  } catch {
    return null
  }
})()

const initial = LANGUAGES.some((l) => l.code === stored) ? (stored as LanguageCode) : 'en'

void i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    de: { translation: de },
    fr: { translation: fr },
  },
  lng: initial,
  fallbackLng: 'en',
  // The dictionaries use single-brace placeholders ({count}, {model}); i18next's default is
  // double braces, and a mismatch renders the placeholder literally instead of failing loudly.
  interpolation: { escapeValue: false, prefix: '{', suffix: '}' },
  returnNull: false,
})

document.documentElement.lang = initial

export function setLanguage(code: LanguageCode) {
  void i18n.changeLanguage(code)
  document.documentElement.lang = code
  try {
    localStorage.setItem('foundry.lang', code)
  } catch {
    /* ignore */
  }
}

export default i18n
