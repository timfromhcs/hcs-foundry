# Evidence — v7.4.0 (disk-light, opt-in update check, panel a11y)

**Date:** 2026-11-02
**Plan:** `docs/plan/V8_3_DISKLIGHT.md` (D1–D3)
**Rule applied:** narrow test → module → suite, max 5 iterations, no weakened assertions

## D1 — disk-light training

| Before | After |
|---|---|
| optimizer, scheduler and RNG state written with every checkpoint | not written (`save_only_model=True`) |
| `save_steps = max(1, save_every)` | `max(10, save_every)` — no per-step checkpoint storms |
| write cost unmeasured | `manifest.extras["disk_bytes_written"]` |

Measured (`sft/fp32_lora`, 11 steps, seq 24, lora_r 2):

```
disk written   53,732 bytes
adapter        adapter_model.safetensors present
optimizer.pt   absent
scheduler.pt   absent
rng_state.pth  absent
```

`tests/train/test_disk_light.py` asserts all three absences and the byte count, and checks that no
more than two checkpoint directories exist.

## D2 — update check

```
default            {"enabled": false, "status": "disabled"}   zero network calls (test patches the checker and asserts it never runs)
enabled, newer     {"status": "ok", "available": true, "latest": "v9.9.9"}
enabled, same      {"status": "ok", "available": false}
network down       {"status": "unreachable", ...}             no exception reaches the UI
no tag_name        {"status": "invalid_response"}
```

There is no code path that writes a binary: the module returns a dict and the UI prints it.
Settings exposes the toggle plus "Check now"; the E2E test asserts the toggle starts unchecked and
the button is disabled until it is enabled.

## D3 — accessibility

| Element | Change |
|---|---|
| browser card | `role="region"`, `aria-label` = the panel title, `aria-busy` while loading |
| load stage | `role="status" aria-live="polite"` (pre-mounted, text only) |
| generation metrics | same live region pattern |
| output | already `aria-live="polite"` |

The focus ring is unchanged (`:focus-visible`, 2 px, 2 px offset) and the existing keyboard E2E
still passes.

## Gates

```
backend (windows, py3.14)   319 passed (10 new)
frontend (vitest)            17 passed
E2E (playwright)             11 passed (1 new)
build                        green
CI                           recorded in the release report
```

## Not claimed

- No delta-snapshot or page-aligned checkpoint writer (`docs/DISK_LIGHT.md` explains why).
- No automatic update download or install.
- No quality claim; S5 is still open.
