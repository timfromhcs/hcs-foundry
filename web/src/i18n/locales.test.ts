import { describe, expect, it } from 'vitest'

import { de } from './locales/de'
import { en } from './locales/en'
import { fr } from './locales/fr'

const locales = { en, de, fr }
const enKeys = Object.keys(en).sort()

describe('translations', () => {
  it('has identical key sets in every language', () => {
    for (const [code, dict] of Object.entries(locales)) {
      expect(Object.keys(dict).sort(), `keys in ${code}`).toEqual(enKeys)
    }
  })

  it('has no empty or placeholder values', () => {
    for (const [code, dict] of Object.entries(locales)) {
      for (const [key, value] of Object.entries(dict)) {
        expect(value.trim().length, `${code}:${key}`).toBeGreaterThan(0)
        expect(value, `${code}:${key}`).not.toMatch(/TODO|FIXME|lorem ipsum/i)
      }
    }
  })

  it('keeps interpolation placeholders consistent', () => {
    const placeholders = (value: string) => (value.match(/\{[a-zA-Z_]+\}/g) ?? []).sort()
    for (const key of enKeys) {
      const expected = placeholders(en[key as keyof typeof en])
      for (const [code, dict] of Object.entries(locales)) {
        expect(placeholders(dict[key as keyof typeof dict]), `${code}:${key}`).toEqual(expected)
      }
    }
  })

  it('covers the navigation labels in all languages', () => {
    for (const key of ['nav.overview', 'nav.machine', 'nav.models', 'nav.playground', 'nav.data', 'nav.identity', 'nav.training', 'nav.benchmarks', 'nav.upgrades', 'nav.agent', 'nav.runs', 'nav.settings']) {
      for (const [code, dict] of Object.entries(locales)) {
        expect(dict[key as keyof typeof dict], `${code}:${key}`).toBeTruthy()
      }
    }
  })
})
