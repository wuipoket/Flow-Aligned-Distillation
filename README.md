<div align="center">

# FAD · Flow-Aligned Distillation

### Reproducible cross-layer LLM compression with teacher-guided functional recovery and calibrated adaptive exit.

[![Verify repository](https://github.com/hanklin9188/Flow-Aligned-Distillation/actions/workflows/verify.yml/badge.svg)](https://github.com/hanklin9188/Flow-Aligned-Distillation/actions/workflows/verify.yml)
[![Research artifact](https://img.shields.io/badge/artifact-reproducible-2f6f62)](REPRODUCIBILITY.md)
[![Protocol audit](https://img.shields.io/badge/protocol-audited-6b5ca5)](docs/AUDIT.md)

[繁體中文](README_zh-TW.md) · [Method](docs/METHOD.md) · [Results](docs/RESULTS.md) · [Reproduce](reproduction/README.md) · [Deploy](deployment/README.md)

<img src="assets/figures/main.png" alt="FAD teacher-to-student transport alignment" width="100%">

</div>

---

FAD reduces the number of independent feed-forward weight sets in Llama-3.2-3B, then trains layer-private low-rank adapters to recover the teacher's projected FFN residual updates. A separate four-feature controller can exit at layer 16, 20, 24, or 28 at inference time.

This repository is organized as an **auditable research artifact**, not only a paper-code dump: selected raw outputs are retained, processed tables can be recomputed, protocol differences are disclosed, and a weight-free mock deployment can be exercised in CI.

## What this project demonstrates

- **Structural compression:** 28 decoder layers are mapped to fewer independent FFN weight sets.
- **Functional recovery:** the student is supervised in a teacher-derived projected transport space with a teacher-defined per-direction scale.
- **Deployment control:** adaptive exit trades computation for accuracy using an explicitly saved controller.
- **Evidence discipline:** public claims are re-derived from archived artifacts by `scripts/verify_data.py`.
- **Honest scope:** historical baseline results are separated from strictly current protocol claims.

## Evidence snapshot

The table below reports the selected FAD artifacts under the repository's seven-task zero-shot macro-accuracy protocol: PIQA, Social-IQA, WinoGrande, ARC-Challenge, ARC-Easy, HellaSwag, and OpenBookQA.

| Compression | Independent FFNs | Static FAD | FAD + adaptive exit | Accuracy change | Average exit | Layer saving |
|---:|---:|---:|---:|---:|---:|---:|
| 15% | 22 / 28 | **85.38%** | **85.41%** | +0.04 pp | 17.32 | 38.16% |
| 20% | 19 / 28 | **85.66%** | **85.53%** | −0.14 pp | 18.16 | 35.16% |
| 25% | 17 / 28 | **85.03%** | **84.94%** | −0.10 pp | 16.39 | 41.47% |
| 30% | 15 / 28 | **83.20%** | **82.89%** | −0.31 pp | 17.16 | 38.70% |

For the archived 25% H100 batch-1 runtime decomposition, the repository distinguishes two different summaries:

- **1.465× aggregate throughput ratio** versus the merged teacher.
- **1.388× task-wise geometric-mean speedup** versus the merged teacher.

Both values are recomputed from [`data/processed/runtime_25pct.csv`](data/processed/runtime_25pct.csv) and its source artifacts; they must not be interchanged.

## Hardware portability and Gemma 2 extension

A downstream study reproduced the compression and evaluation workflow on an AMD Radeon RX 7900 XTX and extended FAD to Gemma 2 9B. The 9.46B-parameter Teacher was reduced to a 7.58B Student, lowering allocated VRAM from 18.02 to 14.44 GiB while achieving 69.86% five-fold micro accuracy versus 74.32% for the merged Teacher. The matched BF16 benchmark showed that the memory reduction did not produce a latency speedup.

These results use a separate five-fold QA protocol and are not directly comparable with the seven-task Llama-3.2-3B table above. See [`experiments/gemma2-9b-rx7900/`](experiments/gemma2-9b-rx7900/) for the complete public summary and report.

## Protocol status

### Current FAD artifact protocol

Current FAD evaluator artifacts explicitly record `length_norm=none`, and `scripts/verify_data.py` rejects a current FAD artifact that does not preserve this field.

### Historical baseline context

The archived FLAP, Týr-the-Pruner, and LLM-Streamline runs were produced during the paper-era workflow. Their wrappers historically defaulted to answer-length normalization while the current public wrappers use `none`. The comparison remains useful as historical context, but it is **not presented as a fully matched current rerun** until every method is regenerated under one frozen protocol.

See [`docs/AUDIT.md`](docs/AUDIT.md) and [`LIMITATIONS.md`](LIMITATIONS.md) before citing a cross-method conclusion.

## Verify without a GPU

The repository's public integrity checks use only the standard Python library:

```bash
python scripts/verify_data.py
```

This recomputes the published macro scores and runtime ratios, validates paired-question accounting, checks local website links, and scans public text files for common private-path markers.

The mock deployment can also be exercised without weights:

```bash
bash deployment/web-demo/start-mock.sh 8765 &
server_pid=$!
python deployment/web-demo/smoke_test.py --base_url http://127.0.0.1:8765
kill "$server_pid"
```

GitHub Actions runs both paths on every push and pull request.

## Reproduce the research workflow

A complete GPU run requires separately obtained model weights and benchmark data. The repository provides launchers, configuration snapshots, baseline adapters, and a Slurm entry point; GPU experiments should be submitted through the scheduler rather than executed on a login node.

```bash
sbatch reproduction/fad/slurm/fad_budgets.sbatch
```

Start with [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for boundaries and [`reproduction/README.md`](reproduction/README.md) for the executable workflow.

## Repository map

```text
assets/                     figures and public research-artifact material
data/raw/                   compact source artifacts retained for audit
data/processed/             analysis-ready CSV tables
deployment/web-demo/        weight-free mock and optional real deployment
reproduction/fad/           FAD policy, training, evaluation, and Slurm entry points
reproduction/baselines/     FLAP, Týr, and LLM-Streamline adapters
scripts/verify_data.py       executable authority for public numeric claims
docs/                       method, results, protocol audit, and file guide
```

For a file-by-file walkthrough, read [`docs/FILE_GUIDE.md`](docs/FILE_GUIDE.md).

## Five-minute reviewer path

1. Inspect the architecture figure above.
2. Read [`docs/METHOD.md`](docs/METHOD.md) and [`docs/AUDIT.md`](docs/AUDIT.md).
3. Run `python scripts/verify_data.py`.
4. Inspect [`data/processed/runtime_25pct.csv`](data/processed/runtime_25pct.csv).
5. Read [`LIMITATIONS.md`](LIMITATIONS.md) before interpreting cross-method or deployment claims.

## Reproducibility boundary

Included:

- code and launch configuration;
- compact raw and processed result artifacts;
- provenance checks and protocol notes;
- figures and a weight-free mock interface;
- expected checksums for externally supplied deployment artifacts.

Not redistributed:

- Meta Llama weights or merged checkpoints;
- the 25% deployable student bundle;
- full benchmark corpora;
- the complete per-question benchmark record;
- private cluster paths, credentials, tokens, or scheduler outputs.

## Project status

| Area | Status |
|---|---|
| Public data-consistency audit | **Executable and CI-enforced** |
| Weight-free deployment smoke test | **Executable and CI-enforced** |
| Selected FAD budget artifacts | **Published** |
| Historical baseline context | **Published with protocol caveat** |
| Fully matched baseline rerun | **Pending** |
| External model/data redistribution | **Intentionally excluded** |

## License and attribution

Original unpublished FAD material remains subject to [`NOTICE.md`](NOTICE.md). Third-party projects, models, and datasets retain their own terms; see [`THIRD_PARTY.md`](THIRD_PARTY.md). Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

