import { config } from 'dotenv'
config({ path: '.env.local' })
import { gateway } from '@ai-sdk/gateway'
import { generateText } from 'ai'

/**
 * "Normal" AI SDK path: the first-party Vercel AI Gateway provider.
 * Model strings use the gateway's provider/model format.
 */
const key = process.env.AI_GATEWAY_API_KEY
if (!key) {
  console.error('AI_GATEWAY_API_KEY is missing. Put it in .env.local (git-ignored).')
  process.exit(2)
}

try {
  const { text, usage, finishReason } = await generateText({
    model: gateway(process.env.AI_GATEWAY_MODEL ?? 'moonshotai/kimi-k3'),
    prompt: 'Invent a new holiday and describe its traditions in two sentences.',
    maxOutputTokens: 120,
  })
  console.log(text)
  console.log('finishReason:', finishReason)
  console.log('usage:', usage)
} catch (error) {
  const message = error instanceof Error ? error.message : String(error)
  console.error('gateway request failed:')
  console.error(message)
  if (message.includes('credit card') || message.includes('customer_verification_required')) {
    console.error('\nThe key is valid, but the Vercel account needs a payment method before inference is enabled.')
    console.error('Add one at https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai%3Fmodal%3Dadd-credit-card')
    process.exit(3)
  }
  process.exit(1)
}
