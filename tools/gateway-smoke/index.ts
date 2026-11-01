import { config } from 'dotenv'
config({ path: '.env.local' })
import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { generateText } from 'ai'

/**
 * OpenAI-compatible path: talks to https://ai-gateway.vercel.sh/v1 directly.
 * Kept separate from the local-first app; it only verifies the key.
 */
const key = process.env.AI_GATEWAY_API_KEY
if (!key) {
  console.error('AI_GATEWAY_API_KEY is missing. Put it in .env.local (git-ignored).')
  process.exit(2)
}

const gateway = createOpenAICompatible({
  name: 'vercel-ai-gateway',
  baseURL: 'https://ai-gateway.vercel.sh/v1',
  apiKey: key,
})

const model = gateway(process.env.AI_GATEWAY_MODEL ?? 'moonshotai/kimi-k3')

try {
  const { text, usage } = await generateText({
    model,
    prompt: 'Invent a new holiday and describe its traditions in two sentences.',
    maxOutputTokens: 120,
  })
  console.log(text)
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
