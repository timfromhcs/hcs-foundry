import { expect, test, type Page } from '@playwright/test'

/**
 * These tests run against the built SPA served by the local Foundry service.
 * Start it first:  python -m foundry.cli serve --port 8770
 * Then:           npm run test:e2e
 */

const ROUTES: Array<{ path: string; heading: string }> = [
  { path: '/', heading: 'Overview' },
  { path: '/machine', heading: 'Machine' },
  { path: '/models', heading: 'Models' },
  { path: '/hub', heading: 'Model Hub' },
  { path: '/playground', heading: 'Playground' },
  { path: '/data', heading: 'Data Studio' },
  { path: '/identity', heading: 'Identity Maker' },
  { path: '/training', heading: 'Training Lab' },
  { path: '/benchmarks', heading: 'Benchmarks' },
  { path: '/upgrades', heading: 'Upgrade Lab' },
  { path: '/pipeline', heading: 'Pipeline' },
  { path: '/agent', heading: 'Agent Workspace' },
  { path: '/runs', heading: 'Runs & Reports' },
  { path: '/settings', heading: 'Settings' },
]

let token = ''

test.beforeAll(async ({ request }) => {
  const response = await request.post('/api/v1/system/pair', {
    data: { label: 'playwright' },
    headers: { 'X-Foundry-Client': '1' },
  })
  expect(response.ok()).toBeTruthy()
  token = (await response.json()).token
})

async function authenticate(page: Page) {
  await page.addInitScript((value: string) => {
    sessionStorage.setItem('foundry.token', value)
    if (!localStorage.getItem('foundry.theme')) localStorage.setItem('foundry.theme', 'dark')
    if (!localStorage.getItem('foundry.lang')) localStorage.setItem('foundry.lang', 'en')
  }, token)
}

test('every route renders without console errors', async ({ page }) => {
  await authenticate(page)
  const errors: string[] = []
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.text().startsWith('Failed to load resource')) {
      errors.push(message.text())
    }
  })
  page.on('pageerror', (error) => errors.push(error.message))

  for (const route of ROUTES) {
    await page.goto(route.path)
    await expect(page.getByRole('heading', { level: 1, name: route.heading })).toBeVisible()
    await expect(page.getByRole('complementary', { name: 'Primary' })).toBeAttached()
  }
  expect(errors, errors.join('\n')).toEqual([])
})

test('keyboard navigation reaches the skip link and the nav', async ({ page }) => {
  await authenticate(page)
  await page.goto('/')
  await page.keyboard.press('Tab')
  await expect(page.locator('.skip-link')).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(page.getByRole('link', { name: 'Overview' })).toBeFocused()
})

test('mobile layout has no horizontal overflow and exposes the drawer', async ({ page }) => {
  await authenticate(page)
  await page.setViewportSize({ width: 390, height: 844 })
  for (const path of ['/', '/models', '/hub', '/benchmarks', '/runs']) {
    await page.goto(path)
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
    expect(overflow, `horizontal overflow on ${path}`).toBeLessThanOrEqual(1)
  }
  const menu = page.getByRole('button', { name: 'Overview' })
  await expect(menu).toBeVisible()
  await menu.click()
  await expect(page.getByRole('link', { name: 'Settings' })).toBeVisible()
  await page.keyboard.press('Escape')
  await page.goto('/')
  const bottomNav = page.getByRole('navigation', { name: 'Mobile' })
  await expect(bottomNav).toBeVisible()
  await bottomNav.getByRole('link', { name: 'Model Hub' }).click()
  await expect(page.getByRole('heading', { level: 1, name: 'Model Hub' })).toBeVisible()
})

test('command palette navigates with the keyboard', async ({ page }) => {
  await authenticate(page)
  await page.goto('/')
  await page.keyboard.press('Control+k')
  const dialog = page.getByRole('dialog', { name: 'Open command palette' })
  await expect(dialog).toBeVisible()
  await page.keyboard.type('bench')
  await page.keyboard.press('Enter')
  await expect(page.getByRole('heading', { level: 1, name: 'Benchmarks' })).toBeVisible()
})

test('reduced motion is respected', async ({ page }) => {
  await authenticate(page)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/')
  const duration = await page.evaluate(() => {
    const card = document.querySelector('.card')
    if (!card) return '0s'
    return getComputedStyle(card).transitionDuration
  })
  expect(['0s', '0.001ms', '1e-06s']).toContain(duration)
})

test('language switch translates the interface', async ({ page }) => {
  await authenticate(page)
  await page.goto('/')
  await page.getByLabel('Language').selectOption('de')
  await expect(page.getByRole('link', { name: 'Übersicht' })).toBeVisible()
  await expect(page.getByRole('heading', { level: 1, name: 'Übersicht' })).toBeVisible()
})

test('light theme applies and persists', async ({ page }) => {
  await authenticate(page)
  await page.goto('/')
  await page.getByRole('button', { name: 'Theme' }).click()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')
  await page.reload()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')
})
