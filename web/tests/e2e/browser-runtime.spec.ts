import { expect, test, type Page } from '@playwright/test'

let token = ''

test.beforeAll(async ({ request }) => {
  const response = await request.post('/api/v1/system/pair', {
    data: { label: 'playwright-browser-runtime' },
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

test('browser runtime panel reports real capabilities and refuses without a model', async ({ page }) => {
  const consoleErrors: string[] = []
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text())
  })
  await authenticate(page)
  await page.goto('/playground')

  const panel = page.locator('.card').filter({ hasText: 'Browser runtime' }).first()
  await expect(panel).toBeVisible()

  // Chips come from a live detection, not from static text.
  await expect(panel.getByText(/WebGPU:/)).toBeVisible()
  await expect(panel.getByText(/WASM SIMD:/)).toBeVisible()
  await expect(panel.getByText(/cores/)).toBeVisible()

  // Headless Chromium usually has no adapter: the honest state is WASM-only or unsupported,
  // never a fake "ready".
  await expect(panel.locator('.badge').first()).toHaveText(/WASM|WEBGPU|unsupported/i)

  // Without an installed ONNX model the panel says so and offers no load button.
  await expect(panel.getByText(/No browser model installed/i)).toBeVisible()
  await expect(panel.getByRole('button', { name: /Load in browser/i })).toHaveCount(0)
  await expect(panel.getByText(/no CDN|no upload/i)).toBeVisible()

  expect(consoleErrors, consoleErrors.join('\n')).toEqual([])
})
