# autoresearch — Brian's setup

Adapted from karpathy/autoresearch via the jsegov/autoresearch-win-rtx Windows fork.
The differences from upstream are not cosmetic — read the **House rules** section
before you run anything. This machine is not a dedicated research box.

You are an autonomous researcher. Your job is to lower **val_bpb** on a fixed
5-minute training budget, by editing exactly one file, and to keep an honest
record of what worked and what did not.

---

## House rules (these override upstream instructions)

**1. Never run `uv run train.py` directly. Always run:**

```
python C:\Brian\02_Projects\brian-os-fleet\runtime\autoresearch_run.py
```

This box has one 12 GB card and Brian's agent fleet shares it — 21 scheduled jobs
a day load qwen3.5:9b through Ollama. The runner waits for a window where no
GPU-using fleet job is due, frees VRAM if Ollama is holding a model, runs the
training, and releases the lock even if the run crashes. Running `train.py`
directly will eventually OOM a fleet job at 6:45am, and a fleet failure costs
Brian more than an experiment is worth.

**The fleet always wins a tie.** If the runner says it is waiting, let it wait.

To see when windows are available: `python ...\runtime\gpu_guard.py --plan`

**2. Nothing goes public. Ever.** Do not push a branch, open a PR, create a gist,
or post results anywhere. This repo is local. If you think something is worth
sharing, say so in your log and let Brian decide.

**3. Report the number or say you do not have it.** If a run crashes, log the
crash — do not quietly retry until something works and present that as the
result. If you are unsure whether an improvement is real, say so.

**4. Do not install packages, and do not modify `prepare.py`.** The evaluation
harness is the ground truth. Changing it invalidates every number in the table.

---

## The hardware you are actually on

| | |
|---|---|
| GPU | RTX 4070 SUPER, 12.0 GB VRAM, Ada (compute 8.9) |
| Attention | PyTorch SDPA — no FlashAttention-3, no Triton |
| `torch.compile` | **disabled** — it is unreliable on Windows; do not turn it on |
| Batch size | chosen by the fork's autotune pass, not by you directly |
| Dataset | TinyStories GPT-4 clean (narrow, so small models give sane results) |
| Baseline model | 50.3M params, depth 8, n_embd 512, seq_len 2048, vocab 8192 |

### Measured baseline on this machine (2026-08-19)

```
val_bpb:          0.938695     <- the number to beat
training_seconds: 303.7
total_seconds:    525.6        <- eval costs as much as training
peak_vram_mb:     1777.1       <- 1.7 GB of 12.0 GB
mfu_percent:      10.45
total_tokens_M:   24.1
num_steps:        46           <- only 46 optimizer steps in the whole budget
num_params_M:     50.3
train_batch_size: 4  (activation checkpointing: enabled)
```

A full experiment is **~8.8 minutes** wall clock, not 5 — eval is roughly as
expensive as training. So expect ~6 experiments an hour, not 12, and roughly
**20 overnight** in the clean 02:09–06:28 window.

An H100 running upstream gets ~40% MFU and 45 GB peak. Do not compare your
numbers to anything you have read online. **The only valid comparison is the
baseline row in `results.tsv` on this machine.**

**VRAM ceiling: keep peak under 9.0 GB.** You have 12.0 GB but Ollama needs
~6.6 GB the moment a fleet job fires, and the guard's horizon is not infinite.
There is real headroom above the current 1.8 GB — using it is one of the most
obvious places to look for a win.

---

## Setup (first run of a new session)

1. Confirm the branch: `git rev-parse --abbrev-ref HEAD` should be
   `autoresearch/<tag>`. If you are on `master`, create a fresh branch
   `autoresearch/<tag>` where `<tag>` is today's date, e.g. `aug19`.
2. Read `README.md`, `prepare.py` (read-only), and `train.py` (the file you edit).
3. Confirm the data is prepared — `prepare.py` has already been run and the
   tokenizer lives under `%LOCALAPPDATA%\autoresearch\datasets\tinystories`.
4. Confirm `results.tsv` exists with the header row. Do not commit it.
5. Establish the baseline before changing anything: run the runner unmodified and
   log that row first. Everything else is measured against it.

---

## What is fair game

Everything in `train.py`: architecture, depth, width, attention pattern,
optimizer, learning rates, schedule, weight decay, batch size, activation
functions, normalization, initialization, gradient accumulation.

The knobs at the top of `train.py` are the obvious starting points:

```
ASPECT_RATIO, HEAD_DIM, WINDOW_PATTERN, TOTAL_BATCH_SIZE,
EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR,
WEIGHT_DECAY, ADAM_BETAS, WARMUP_RATIO, WARMDOWN_RATIO, FINAL_LR_FRAC,
DEPTH, DEVICE_BATCH_SIZE, EVAL_BATCH_SIZE
```

But the knobs are not the interesting part. The interesting part is the model and
the optimizer, and you are allowed to rewrite them.

**Simplicity criterion.** All else equal, simpler wins. A 0.001 improvement that
adds 20 lines of hacky code is not worth it. A 0.001 improvement from *deleting*
code is an excellent outcome. Equal results with less code — keep it.

---

## The loop

```
LOOP FOREVER:
  1. git status — know which branch and commit you are on
  2. edit train.py with ONE clear idea (not five at once — you will not know
     which one worked)
  3. git commit -m "<the idea, in one line>"
  4. python C:\Brian\02_Projects\brian-os-fleet\runtime\autoresearch_run.py
  5. read the result it prints (it also writes run.log)
  6. if it crashed: read the tail it printed. Dumb bug (typo, import) -> fix and
     re-run. Fundamentally broken idea -> log it as a crash and move on.
  7. append a row to results.tsv
  8. improved (lower val_bpb)?  keep the commit, the branch advances
     equal or worse?             git reset --hard HEAD~1
  9. every ~10 experiments, run:
     python C:\Brian\02_Projects\brian-os-fleet\runtime\autoresearch_agent.py
     so comms\autoresearch.md reflects reality and the morning brief can see it
```

### results.tsv

Tab-separated. Commas break the descriptions. Five columns:

```
commit	val_bpb	memory_gb	status	description
```

- `commit` — short hash, 7 chars
- `val_bpb` — 6 decimals; `0.000000` for a crash
- `memory_gb` — `peak_vram_mb / 1024`, one decimal; `0.0` for a crash
- `status` — `keep`, `discard`, or `crash`
- `description` — what you tried, in a few words

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.234567	1.7	keep	baseline
b2c3d4e	1.229100	2.4	keep	raise DEVICE_BATCH_SIZE to 8
c3d4e5f	1.240000	1.7	discard	swap Muon for plain AdamW
d4e5f6g	0.000000	0.0	crash	depth 24 (OOM)
```

---

## Where to look for wins on THIS machine

These are hypotheses, not instructions. Test them; discard the ones that fail.

- **You are using 1.7 GB of 12 GB.** The autotune picked batch 4 *with*
  activation checkpointing — it traded memory it did not need for compute it
  could not spare. Turning checkpointing off, or raising the batch, is the very
  first thing to measure.
- **You only got 46 optimizer steps.** `TOTAL_BATCH_SIZE = 2**19` with 64
  gradient-accumulation micro-steps means each step costs ~8.6 s of the 300 s
  budget. Upstream's own small-GPU guidance says drop `TOTAL_BATCH_SIZE` a lot
  (to `2**14`–`2**16`). More, cheaper steps is the highest-prior hypothesis here.
- **MFU is ~10%.** On a consumer card with SDPA and no compile, you are probably
  launch-bound rather than FLOP-bound.
- **`WINDOW_PATTERN = "SSSL"`** is banded attention designed for long context on
  datacenter hardware. Upstream's own guidance says plain `"L"` may be faster on
  small setups. Measure it.
- **TinyStories is narrow.** The optimal model here is probably smaller and
  trained longer than the 50M/depth-8 default, not bigger. Depth 4 or 6 with more
  steps in the same 5 minutes is worth a run.
- **`vocab_size = 8192` on TinyStories** may be larger than the data needs. This
  lives in `prepare.py` which you may not edit — so note it as a finding for
  Brian rather than acting on it.

---

## Never stop

Once the loop has begun, do not pause to ask whether to continue. Brian may be
asleep. If you run out of ideas: re-read `train.py` properly, combine two
near-misses, or try something structurally different rather than another
hyperparameter nudge. The loop ends when Brian ends it.

The one exception is the guard. If `autoresearch_run.py` reports it cannot get a
GPU window for a long stretch, that is not a reason to bypass it — that is the
system working. Wait, or stop and report.
