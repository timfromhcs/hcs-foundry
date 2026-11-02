// Real browser check of the pairing flow: the landing page served from a loopback origin
// talks to the Foundry service on 127.0.0.1:8770. This is what the deployed page would do
// if the browser allowed public-HTTPS pages to reach loopback.
const { chromium } = require('playwright')

const PAGE = process.env.PAIR_PAGE || 'http://127.0.0.1:4180/index.html#pair'

;(async () => {
  const browser = await chromium.launch()
  const page = await browser.newPage()
  const consoleErrors = []
  page.on('console', (msg) => { if (msg.type() === 'error') consoleErrors.push(msg.text()) })
  await page.goto(PAGE, { waitUntil: 'networkidle' })
  await page.locator('#pair').scrollIntoViewIfNeeded()
  await page.locator('#pair-connect').click()
  await page.waitForTimeout(3500)
  const status = await page.locator('#pair-status').innerText()
  const factsVisible = await page.locator('#pair-facts').isVisible()
  const version = factsVisible ? await page.locator('#pair-version').innerText() : null
  const model = factsVisible ? await page.locator('#pair-model').innerText() : null
  console.log(JSON.stringify({ status, factsVisible, version, model, consoleErrors }, null, 2))
  await browser.close()
})()
