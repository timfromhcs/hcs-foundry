# Model Hub (Hugging Face)

HCS Foundry can search the Hugging Face Hub and pull files into the local model library. **Public repositories work without a token.** A token only adds gated and private repositories.

## Search and inspect

- Search by text, filter by pipeline task, sort by downloads, likes or recency.
- Selecting a result lists every file with size and, where the Hub publishes one, the LFS SHA-256.
- The default selection is the smallest GGUF plus its vision projector, or the safetensors weights when the repo is not a GGUF repo.
- You can select any subset of files before pulling.

## Download behaviour

- Streamed to a `.part` file, then atomically renamed.
- Resumable: an interrupted `.part` continues with an HTTP `Range` request.
- If the server ignores the range and returns `200`, the download restarts cleanly instead of appending duplicates.
- SHA-256 is verified when the Hub publishes a digest; a mismatch deletes the file.
- Free disk space is checked before a file starts.
- Progress, cancellation and the log tail appear in the job tray.
- After a successful pull the destination is scanned and the model appears in **Models**.

Destination: `<data>/models/<org>/<repo>/` unless you pass another configured model root. Path traversal is stripped by `sanitize_repo_dir`.

## Tokens

| Mode | Works for |
|---|---|
| Anonymous | all public repos |
| Token | public + gated + private repos you can access |

The token is stored in `data/state/hf-token.json`, which is git-ignored, and is masked in every API response. It is never logged and never sent anywhere except `huggingface.co`.

Set it in **Model Hub → Hugging Face token**, or:

```bash
foundry hf token set --token hf_xxx
foundry hf token status
foundry hf token clear
```

## CLI

```bash
foundry hf search "qwen3 0.6b" --limit 5
foundry hf files Qwen/Qwen3-0.6B
foundry hf pull Qwen/Qwen3-0.6B --file model.safetensors --file tokenizer.json
foundry hf pull TheBloke/Mistral-7B-GGUF --file mistral-7b.Q4_K_M.gguf
```

`--json` prints a machine-readable result.

## Errors you may see

| Code | Meaning |
|---|---|
| `auth_required` | Gated or private repo; add a token with access |
| `not_found` | Wrong repo id or file path |
| `rate_limited` | Hub rate limit; retry later or add a token |
| `no_space` | Not enough free disk for the file |
| `hash_mismatch` | Downloaded bytes do not match the published digest; file discarded |

## Licensing

HCS Foundry shows the model id and never assumes a license. Check the repository card before downloading or training. Gated models additionally require accepting the provider's terms on the Hub.
