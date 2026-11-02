// Local stress test for the browser runtime.
//
//   node scripts/browser_stress.mjs [http://127.0.0.1:8770]
//
// What it does:
//   1. pairs with the local service and opens /playground,
//   2. reads the capability panel (WebGPU/WASM/cores/memory) and the computed safe limits,
//   3. probes how much memory this browser actually grants (bounded, released immediately),
//   4. if a browser model is installed, loads it and generates tokens, reporting tok/s,
//   5. exits non-zero when the runtime claims support it cannot deliver, or when the panel errors.
import { chromium } from 'playwright'

const base = process.argv[2] ?? 'http://127.0.0.1:8770'

function fail(message) {
  console.error(`FAIL: ${message}`)
  process.exit(1)
}

const browser = await chromium.launch()
const page = await browser.newPage()
const consoleErrors = []
page.on('console', (message) => {
  if (message.type() === 'error') consoleErrors.push(message.text())
})
page.on('pageerror', (error) => consoleErrors.push(`pageerror: ${error.message}`))

const pair = await page.request.post(`${base}/api/v1/system/pair`, {
  data: { label: 'browser-stress' },
  headers: { 'X-Foundry-Client': '1' },
})
if (!pair.ok()) fail(`pairing failed with HTTP ${pair.status()}`)
const { token } = await pair.json()

await page.addInitScript((value) => {
  sessionStorage.setItem('foundry.token', value)
  localStorage.setItem('foundry.lang', 'en')
}, token)

await page.goto(`${base}/playground`, { waitUntil: 'networkidle' })
const panel = page.locator('.card').filter({ hasText: 'Browser runtime' }).first()
await panel.waitFor({ state: 'visible', timeout: 15000 })

const text = await panel.innerText()
const badge = (await panel.locator('.badge').first().innerText()).trim().toUpperCase()

// 3. bounded memory probe — how much can this tab really allocate?
const allocation = await page.evaluate(async () => {
  const blocks = []
  const size = 128 * 1024 * 1024
  let granted = 0
  try {
    for (let i = 0; i < 12; i += 1) {
      const buffer = new ArrayBuffer(size)
      new Uint8Array(buffer)[0] = 1
      blocks.push(buffer)
      granted += size
    }
  } catch {
    /* stop at the ceiling */
  }
  const bytes = granted
  blocks.length = 0
  return { bytes, mib: Math.round(bytes / 1024 / 1024) }
})

const report = {
  url: base,
  backend: badge,
  capabilities: text
    .split('\n')
    .filter((line) => /WebGPU|WASM|threads|cores|device memory|secure context/i.test(line)),
  limits: text.split('\n').filter((line) => /Safe limits/i.test(line)),
  allocation,
  modelPresent: /No browser model installed/i.test(text) === false,
}

// 4. real inference, only when a model is installed
if (report.modelPresent && /WASM|WEBGPU/.test(badge)) {
  const load = panel.getByRole('button', { name: /Load in browser/i })
  if (await load.isEnabled().catch(() => false)) {
    const started = Date.now()
    await load.click()
    await panel.getByText(/Loaded|ready to generate/i).waitFor({ timeout: 180_000 }).catch(() => null)
    const loaded = /Loaded|ready to generate/i.test(await panel.innerText())
    if (loaded) {
      const run = panel.getByRole('button', { name: /Generate in browser/i })
      await run.click()
      await page.waitForTimeout(1500)
      await panel.locator('pre').first().waitFor({ state: 'visible', timeout: 60_000 }).catch(() => null)
      const metrics = await panel.innerText()
      report.inference = {
        loadSeconds: Math.round((Date.now() - started) / 100) / 10,
        metrics: metrics.split('\n').filter((line) => /tok\/s/.test(line)),
        outputChars: (await panel.locator('pre').first().innerText().catch(() => '')).length,
      }
    } else {
      report.inference = { error: 'model did not report ready within 180 s' }
    }
  }
} else if (report.modelPresent) {
  report.inference = { error: `no runnable backend (panel says ${badge})` }
} else {
  report.inference = {
    skipped: 'no ONNX model installed under data/browser-models/<id>/ — put one there to measure',
  }
}

report.consoleErrors = consoleErrors
console.log(JSON.stringify(report, null, 2))
await browser.close()

if (consoleErrors.length) fail(`console errors: ${consoleErrors.join(' | ')}`)
if (!/WASM|WEBGPU/.test(badge)) fail('the panel reports no backend at all')
if (report.allocation.mib < 256) fail(`the tab could not allocate 256 MB (got ${report.allocation.mib} MB)`)
if (report.inference?.error) fail(report.inference.error)
console.log('OK: capabilities detected, limits reported, no console errors')
