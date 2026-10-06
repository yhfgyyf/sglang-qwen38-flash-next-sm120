# Local coding-dialogue validation

`validate_coding_dialogues.py` checks readable text and a complete fixture-tool
round trip against a running local OpenAI-compatible service. It never executes
historical or generated commands. It uses only literal loopback HTTP addresses,
disables environment proxies and redirects, and creates a new private output
directory. Keep the dataset and output directory outside Git.

For Qwen3.8 use `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`. Requests
disable thinking and use temperature zero with normal EOS behavior. Configure the
desired cache policy on the server; the client does not change service settings.

The RTX PRO 6000 / Qwen3.8 NVFP4 four-dialogue validation profile uses the main
README's launch command with row caching enabled and these flag replacements:

```text
--max-running-requests 4
--max-total-tokens 114688
--max-mamba-cache-size 32
--mamba-ssm-dtype float32
--linear-attn-backend triton
--cuda-graph-max-bs-decode 4
--cuda-graph-bs-decode 1 2 3 4
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
```

Keep 8,192-token prefill chunks, the 16,384-token scheduling budget, NEXTN 3/1/4,
and Radix enabled. This profile is for roughly 25K-token dialogues at concurrency
four, not a claim of 256K simultaneous KV capacity. The initial 147,456-token KV
allocation left insufficient headroom when grammar-backed tool sampling first
initialized NCCL: a 512 MiB GPU allocation failed with about 0.45 GiB free.
Reducing KV allocation to 114,688 leaves more room without changing model
precision, tool checks, concurrency, or row-cache bytes. Size GPU headroom for
your own workload; the 512 MiB PLE budget is host-cache accounting, not GPU memory.

```bash
python benchmark/qwen38_ple/validate_coding_dialogues.py \
  --dataset /private/datasets/coding/dataset.jsonl \
  --output /private/results/row512-calibration \
  --endpoint http://127.0.0.1:30001 \
  --subset calibration --concurrency 1

# Start a fresh server with the same settings before the complete run.
python benchmark/qwen38_ple/validate_coding_dialogues.py \
  --dataset /private/datasets/coding/dataset.jsonl \
  --output /private/results/row512-full \
  --endpoint http://127.0.0.1:30001 \
  --subset full --concurrency 4
```

## Input contract

Each JSONL row has a unique safe `sample_id`, a distinct `source_sha256`,
`length_band` (`short`, `medium`, or `long`), `messages`, and `content_sha256`.
Normalized content hashes must also be distinct: different source files can
contain identical conversation windows.
The content hash is SHA256 of UTF-8 JSON for `messages`, with `ensure_ascii=False`,
sorted keys, and separators `(',', ':')`. Additional local provenance fields are
allowed. Messages start/end with real user turns, include at least two user turns,
and contain only `user`, `assistant`, and paired `tool` messages. Historical tool
arguments may be mappings or JSON object strings, as supported by the service.

Build the corpus from authorized histories only. Exclude system/developer/private
reasoning, control/orchestration messages and subagent messages. Redact credentials
and identifying paths before use. Document any visible truncation and preserve
local source hashes/spans. The repository intentionally includes no private corpus.

## What runs

For each sample, three dependent requests share the historical conversation:

1. A brief Chinese summary of the actual coding context and next check, without tools.
2. One read-only `read_code_excerpt` call, selecting it among three fixture tools.
   Parameters test strings, Unicode paths, quotes/backslashes, integers, booleans,
   and arrays. Calls alternate `auto`, named, and `required` selection.
3. The actual generated tool call ID receives an in-memory result containing a
   historical excerpt and an undisclosed Unicode marker. The model must continue
   normally and quote that marker, with no further tool call.

Generated responses are appended to the next request. The original dataset is
never rewritten. With 100 samples, the selection mix is 50 auto / 25 named /
25 required, with both streaming and nonstreaming. Named tool choice can change
the rendered tool prefix because the service narrows the tool list; this is not
by itself a cache-reuse failure. Workers run independent dialogue loops, without
per-stage barriers. No failed request is silently retried.

## Evidence and limits

The output retains each full request, raw response/SSE lines, reconstructed
message, timestamps, hashes, errors, a frozen plan, dialogue results, and summary.
Checks cover strict UTF-8/JSON/SSE, tool identity/arguments/finish reasons, exact
tool-result continuation markers, replacement/control/noncharacter corruption,
missing/duplicate requests, and client concurrency. The first failure is retained.
An HTTP-200 nonstream response with invalid UTF-8 retains its exact bytes in
`raw_body_base64` while still failing validation. JSON numeric overflow is rejected
before nonfinite values can corrupt the saved evidence.

`automated_pass` and exit status cover automated gates only. **They do not approve
publication.** Review every flagged output (mojibake signatures, repetitions,
parser markup, bidi/private-use characters or unclosed fences) and a deterministic
sample of otherwise clean responses, including long prompts and automatic calls.
Also independently verify the loaded code/artifact, actual row-cache budget,
server-side concurrency, no OOM/retraction, and owned-service cleanup.

This checks a bounded text/tool protocol workload, not general coding accuracy,
successful execution of real tools, output equivalence with another cache policy,
or a universal absence of corruption. Concurrency counts from the client alone
do not prove scheduler concurrency. No service performance claim follows from
these functional checks.

CPU harness tests:

```bash
python -m pytest -q test/registered/unit/models/test_qwen4_ple_dialogue_validation.py
```
