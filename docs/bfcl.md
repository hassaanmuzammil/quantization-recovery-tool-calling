# BFCL v4 reference

Reference notes on the Berkeley Function Calling Leaderboard (BFCL) v4 as used
in this project. The model under study is Qwen3-4B-Instruct-2507.

## What BFCL measures

BFCL gives the model a user query and a set of function schemas, then checks
the function call it produces. Single-turn categories are scored with an AST
checker: the call must parse, use the right function name, include every
required parameter, and give each argument a value from the list of accepted
answers. Multi-turn and agentic categories run the calls against simulated
environments and check the resulting state.

### FC mode vs prompting mode

BFCL can run the same model in two ways:

| Mode | Model name in BFCL | How tools are presented |
|---|---|---|
| Function calling (FC) | `Qwen/Qwen3-4B-Instruct-2507-FC` | The model's native chat template, with `<tools>` in the system turn and `<tool_call>…</tool_call>` in the output |
| Prompting | `Qwen/Qwen3-4B-Instruct-2507` | BFCL's own system prompt asks for calls in a Python-like text format |

This project uses FC mode only, because it is the model's native
tool-calling format and the format the xLAM recovery data is rendered in.

## Categories

Case counts come from `bfcl_eval/data/BFCL_v4_*.json` in this repo.
"Live" categories are user-contributed real queries; "non-live" categories
are curated by the BFCL authors.

Difficulty is for Qwen3-4B-Instruct-2507 in FC mode, based on its leaderboard
scores below: easy is about 85% or higher, intermediate is about 60–85%, and
hard is below about 30%.

| Category | Group | Cases | What it tests | Qwen3-4B FC score | Difficulty |
|---|---|---:|---|---:|---|
| `simple_python` | Non-live | 400 | One Python function, one call | Not reported alone (see note 1) | Easy |
| `simple_java` | Non-live | 100 | One Java function, one call | Not reported alone | Easy–intermediate |
| `simple_javascript` | Non-live | 50 | One JavaScript function, one call | Not reported alone | Easy–intermediate |
| `multiple` | Non-live | 200 | Pick the right function from several | 93.50% | Easy |
| `parallel` | Non-live | 200 | Several calls to one function in one turn | 92.50% | Easy |
| `parallel_multiple` | Non-live | 200 | Several calls across several functions | 90.00% | Easy |
| `irrelevance` | Non-live | 240 | Make no call when no function applies | Not reported alone (see note 2) | Easy–intermediate |
| `live_simple` | Live | 258 | One function, real user queries | 79.07% | Intermediate |
| `live_multiple` | Live | 1,053 | Pick the right function, real queries | 76.16% | Intermediate |
| `live_parallel` | Live | 16 | Parallel calls, real queries | 62.50% | Intermediate (too small to rely on) |
| `live_parallel_multiple` | Live | 24 | Parallel calls across functions, real queries | 66.67% | Intermediate (too small to rely on) |
| `live_irrelevance` | Live | 884 | Make no call when no function applies, real queries | Not reported alone (see note 2) | Intermediate |
| `live_relevance` | Live | 16 | Make a call when a function does apply | 87.50% | Easy (too small to rely on) |
| `multi_turn_base` | Multi-turn | 200 | Multi-step tasks in a simulated environment | 26.50% | Hard |
| `multi_turn_miss_func` | Multi-turn | 200 | Multi-turn with a needed function missing | 21.00% | Hard |
| `multi_turn_miss_param` | Multi-turn | 200 | Multi-turn with a needed parameter missing | 15.50% | Hard |
| `multi_turn_long_context` | Multi-turn | 200 | Multi-turn with long context | 25.50% | Hard |
| `web_search_base` | Agentic | 100 | Answer questions by calling a search tool | 4.00% | Hard |
| `web_search_no_snippet` | Agentic | 100 | Web search without result snippets | 2.00% | Hard |
| `memory_kv` | Agentic | 155 | Read and write a key-value memory | 16.13% | Hard |
| `memory_vector` | Agentic | 155 | Read and write a vector memory | 12.26% | Hard |
| `memory_rec_sum` | Agentic | 155 | Read and write a recursive-summary memory | 24.52% | Hard |
| `format_sensitivity` | Not scored | 10 | Robustness to system-prompt format | Prompting mode only | – |

Notes:

1. The leaderboard's "Non-Live Simple AST" (75.50% FC) is the unweighted
   mean of `simple_python`, `simple_java` and `simple_javascript`, so it does
   not give `simple_python` on its own.
2. The leaderboard's "Irrelevance Detection" (84.93% FC) is the unweighted
   mean of `irrelevance` and `live_irrelevance`, so neither is reported on its
   own.

## Official leaderboard results for Qwen3-4B-Instruct-2507

Source: [BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html),
last updated 2026-04-12 ([data_overall.csv](https://gorilla.cs.berkeley.edu/data_overall.csv)).

| Metric | FC | Prompting |
|---|---:|---:|
| Rank | 54 | 55 |
| **Overall accuracy** | **35.68%** | **35.52%** |
| Non-live AST (overall) | 87.88% | 86.44% |
| Non-live simple AST | 75.50% | 77.25% |
| Non-live multiple AST | 93.50% | 91.00% |
| Non-live parallel AST | 92.50% | 88.00% |
| Non-live parallel multiple AST | 90.00% | 89.50% |
| Live (overall) | 76.39% | 74.69% |
| **Live simple AST** | **79.07%** | **77.91%** |
| Live multiple AST | 76.16% | 74.17% |
| Live parallel AST | 62.50% | 81.25% |
| Live parallel multiple AST | 66.67% | 58.33% |
| Multi-turn (overall) | 22.12% | 20.50% |
| Multi-turn base | 26.50% | 24.50% |
| Multi-turn miss func | 21.00% | 21.50% |
| Multi-turn miss param | 15.50% | 16.00% |
| Multi-turn long context | 25.50% | 20.00% |
| Web search (overall) | 3.00% | 4.50% |
| Web search base | 4.00% | 4.00% |
| Web search no snippet | 2.00% | 5.00% |
| Memory (overall) | 17.63% | 23.87% |
| Memory KV | 16.13% | 12.90% |
| Memory vector | 12.26% | 14.19% |
| Memory recursive summarization | 24.52% | 44.52% |
| Relevance detection | 87.50% | 87.50% |
| Irrelevance detection | 84.93% | 75.87% |
| Format sensitivity max delta | N/A | 18.0 |
| Format sensitivity std | N/A | 5.22 |

**Note for this project:** the 77.91% `live_simple` baseline used in
`experiments.md` matches the leaderboard's **prompting-mode** score, not FC
mode (79.07%). For FC-only comparisons, re-run the BF16 model in FC mode in
this setup and use that number as the baseline.

## How the leaderboard aggregates scores

From `bfcl_eval/eval_checker/eval_runner_helper.py`:

- **Overall accuracy** is a weighted mean of five groups: non-live 10%, live
  10%, irrelevance 10%, multi-turn 30%, agentic (web search + memory) 40%.
  This is why Qwen3-4B's overall score (~36%) is low even though its
  single-turn scores are 75–94%.
- **Non-live** averages simple, multiple, parallel and parallel multiple
  equally (unweighted).
- **Live** is weighted by case count, so `live_multiple` (1,053 cases)
  dominates it.
- **Irrelevance detection** is the unweighted mean of `irrelevance` and
  `live_irrelevance`.

## Categories used in this project

The full plan is in [todo.md](todo.md).

1. `live_simple` first, on all models, as an intermediate-difficulty first
   pass.
2. Then 5 fast single-turn categories mixing easy and intermediate:
   `simple_python`, `multiple`, `parallel`, `irrelevance`, `live_simple`.
3. Hard categories (multi-turn, web search, memory) are skipped: the BF16
   model already scores 2–27% on them, so quantized models score near zero
   and the categories cannot separate the configurations.

The proposal also lists `live_multiple` and `live_irrelevance` as essential;
they are larger and are run after the 5-category set.

## Running BFCL in this repo

Run from `benchmarks/gorilla/berkeley-function-call-leaderboard` in the
`bfcl` conda env (vLLM backend, FC mode):

```bash
bfcl generate \
  --model Qwen/Qwen3-4B-Instruct-2507-FC \
  --test-category live_simple \
  --backend vllm \
  --num-gpus 1 \
  --local-model-path <model directory> \
  --gpu-memory-utilization 0.8 \
  --result-dir result/<run name> \
  --allow-overwrite

bfcl evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507-FC \
  --test-category live_simple \
  --result-dir result/<run name> \
  --score-dir score/<run name>
```

Several categories can be passed at once, separated by spaces
(`--test-category simple_python multiple parallel irrelevance live_simple`).
