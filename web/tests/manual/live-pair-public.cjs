// Live check: does the deployed page actually pair with the local service?
const { chromium } = require('playwright')

;(async () => {
  const browser = await chromium.launch()
  const page = await browser.newPage()
  const errors = []
  page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()) })
  await page.goto('https://hcs-foundry-site.vercel.app/#pair', { waitUntil: 'networkidle' })
  await page.locator('#pair').scrollIntoViewIfNeeded()
  const heading = await page.locator('#pair h2').innerText()
  const before = await page.locator('#pair-status').innerText()
  const button = page.locator('#pair-connect')
  const enabled = await button.isEnabled()
  if (enabled) {
    await button.click()
    await page.waitForTimeout(4000)
  }
  const after = await page.locator('#pair-status').innerText()
  const facts = await page.locator('#pair-facts').isVisible()
  console.log(JSON.stringify({ heading, before, enabled, after, factsVisible: facts, errors }, null, 2))
  await browser.close()
})()
