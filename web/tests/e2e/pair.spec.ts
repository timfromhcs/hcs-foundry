import { fileURLToPath, pathToFileURL } from 'node:url'
import path from 'node:path'

import { expect, test } from '@playwright/test'

const here = path.dirname(fileURLToPath(import.meta.url))
const siteUrl = pathToFileURL(path.resolve(here, '../../../site/index.html')).href

test('landing page pairing section explains the offline state and stores no token', async ({ page }) => {
  await page.goto(siteUrl)

  const section = page.locator('#pair')
  await section.scrollIntoViewIfNeeded()
  await expect(section.getByRole('heading', { level: 2 })).toHaveText('Pair this page with your machine')
  await expect(page.locator('#pair-base')).toHaveValue('http://127.0.0.1:8770')
  await expect(page.locator('#pair-connect')).toBeEnabled()

  // file:// has a null origin, so the probe must fail and say so instead of pretending.
  await expect(page.locator('#pair-status')).toContainText(/No local service answered|origin/i, { timeout: 10_000 })
  await expect(page.locator('#pair-facts')).toBeHidden()

  const tokens = await page.evaluate(() => ({
    local: Object.keys(localStorage).filter((key) => key.startsWith('fnd_')).length,
    session: Object.keys(sessionStorage).filter((key) => key.startsWith('fnd_')).length,
  }))
  expect(tokens.local).toBe(0)
  expect(tokens.session).toBe(0)
})

test('pairing section never claims success without a token', async ({ page }) => {
  await page.goto(siteUrl)
  await page.locator('#pair').scrollIntoViewIfNeeded()
  await page.locator('#pair-connect').click()
  await expect(page.locator('#pair-status')).not.toContainText(/Paired\./, { timeout: 10_000 })
  await expect(page.locator('#pair-status')).toContainText(/No local service answered|not allowlisted|Not connected/i)
})
