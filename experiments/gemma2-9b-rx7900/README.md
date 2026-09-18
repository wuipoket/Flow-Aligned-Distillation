# Gemma 2 9B Compression on AMD Radeon RX 7900 XTX

This directory documents a downstream reproduction and deployment study that extends Flow-Aligned Distillation (FAD) to Gemma 2 9B and evaluates the compressed model on a single 24 GiB AMD Radeon RX 7900 XTX.

The study was conducted by Yen-Han Lin and Wei-Ting Liu. It is separate from the repository's original seven-task Llama-3.2-3B results and uses a different five-fold QA evaluation protocol; the two result tables should not be compared directly.

## Study goals

- Reproduce the existing LLM compression and evaluation workflow on AMD ROCm hardware and compare its quality behavior with archived NVIDIA H100 reference runs.
- Assemble and fine-tune a compressed Gemma 2 9B student within a 24 GiB memory envelope.
- Measure the quality, parameter, VRAM, and latency trade-offs of the compressed student.
- Preserve a complete five-fold training, best-checkpoint selection, reload, benchmark, and backup workflow.

## Experimental setup

| Item | Configuration |
|---|---|
| GPU | AMD Radeon RX 7900 XTX, 23.98 GiB |
| Framework | PyTorch 2.6.0 with ROCm 6.4.1 |
| Precision | BF16 |
| Evaluation | Group-safe five-fold English QA evaluation |
| Evaluated rows | 1,589 supported rows; 39 unsupported rows retained in accounting |
| Fine-tuning | LoRA rank 64, alpha 64, dropout 0.05 |
| Training | 8 epochs, batch size 1, gradient accumulation 16 |
| Benchmark | Batch 1; sequence lengths 128, 256, 512, and 768 |

## Main results

| Metric | Merged Teacher | FAD Student | Change |
|---|---:|---:|---:|
| Total parameters | 9.46B | 7.58B | -19.88% |
| Allocated VRAM | 18.02 GiB | 14.44 GiB | -3.58 GiB (-19.87%) |
| Five-fold micro accuracy | 74.32% | 69.86% | -4.47 percentage points |
| Correct supported rows | 1,181 / 1,589 | 1,110 / 1,589 | -71 |
| Relative inference speed | 1.000x | 0.988-0.994x | No speedup |

The student retained 93.99% of the merged Teacher's micro accuracy while reducing both parameter count and allocated VRAM by approximately 20%. Latency remained nearly unchanged because FAD reduces independent FFN parameters but does not shorten the decoder path.

## Engineering work

### CPU-first student assembly

The original GPU-first path created temporary copies during FAD assembly and exhausted the 24 GiB device before training. The revised workflow assembles the student on CPU, attaches the task adapter, and moves only the completed student to the GPU. The resulting student allocated 14.44 GiB and completed forward, backward, save, reload, and evaluation smoke tests.

### Model-family pipeline correction

An early trainer configuration imported a Llama-specific pipeline while constructing Gemma 2. The workflow was corrected to use the formal Gemma 2 reproduction source and was gated by a small end-to-end smoke test before launching the five-fold run.

### Adapter checkpoint filtering

The frozen FAD structure and the trainable PEFT adapter both used names with the `lora_` prefix. This caused 24 unrelated FAD keys to enter the task-adapter checkpoint and fail during reload. The save path was changed to retain only the trainable PEFT state, after which all five best adapters reloaded with zero unexpected keys.

### Best-checkpoint selection

Each fold selected the checkpoint with the lowest validation loss rather than the final epoch. All five folds selected epoch 3, preventing later training-loss improvements from being mistaken for better validation performance.

## Interpretation

- The AMD deployment reproduced the memory-first behavior of the earlier 8B study: approximately 20% lower allocated VRAM with essentially unchanged runtime.
- The Gemma 2 Teacher achieved stronger baseline quality, but the compressed student paid a larger accuracy cost than the earlier 8B student.
- Parameter sharing alone should not be described as an inference speedup when the full decoder depth still executes.

## Public artifact boundary

This public directory intentionally excludes ITRI question text, per-question predictions, model weights, merged checkpoints, FAD deployment bundles, LoRA adapters, private filesystem paths, credentials, tokens, and machine-specific archives.

The numerical summary is a documentation snapshot derived from the completed study. Full reproduction requires separately authorized model and dataset assets.

## Report

- [Gemma 2 9B Compression and RX 7900 Evaluation](Gemma2_9B_RX7900_Experiment_Report.pdf)

