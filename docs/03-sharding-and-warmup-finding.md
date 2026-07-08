# Finding: EXO's CPU Warmup Is a Single-Threaded MLX Inference Pass

**TL;DR** — On CPU-only x86 hardware, EXO shards the model correctly across
nodes but never produces tokens: it stays in an internal `StartWarmup`
phase that runs as a *single-threaded* MLX inference pass and did not
complete even after 62 minutes. llama.cpp, using its multi-threaded GGML
backend, runs the identical model on the same machine in seconds. The
divergence is at the **execution-backend** level, not in distribution or
networking.

## Hardware

| Node | CPU | RAM | Role |
|------|-----|-----|------|
| Dell | i7-1185G7 (4c/8t) | 15 GB | CPU node |
| Mini | i5-6500 (4c) | 15 GB | CPU node |

x86-64, CPU-only. Same base model both frameworks: `Llama-3.2-3B-Instruct`
at matched 4-bit precision (MLX `mlx-community` 4-bit for EXO; GGUF
`Q4_K_M` for llama.cpp). EXO commit `09f9ea31`; MLX 0.31.2 on the CPU node.

## 1. Sharding works — EXO splits layers correctly

Forcing a two-node pipeline via the HTTP placement API
(`POST /place_instance` with `min_nodes=2`), `/state/instances` reported
`worldSize=2` with an even 14/14 split of the 28 transformer layers:

| Node | device_rank | Assigned layers |
|------|-------------|-----------------|
| Mini | 0 | `[0, 14)` — 14 layers |
| Dell | 1 | `[14, 28)` — 14 layers |

Source inspection (`placement.py`) shows the split is RAM-proportional
(`allocate_layers_proportionally()`, largest-remainder method). Backend
type (GPU vs CPU) is used only as an *eligibility filter*, never as a
ranking signal — there is no compute-speed-aware placement.

For contrast, llama.cpp exposes explicit control: `-ngl 14` placed
layer-slots 0–14 on the local Dell CPU and 15–27 on the Mini RPC worker
(llama.cpp counts 29 slots = 28 transformer layers + 1 output slot). That
run completed and generated tokens.

## 2. But warmup never completes on CPU

The two-node run stayed in `StartWarmup` and produced 0 tokens in 30
minutes. To isolate distribution from backend, a **solo-node** run was
done (`world_size=1`, all other nodes stopped and disabled). It reproduced
the stall: still `WARMING UP` at 62 minutes, 0 tokens, then terminated.

This eliminates topology and synchronization as the cause — with a single
node there is no ring to synchronize with.

## 3. Profiling: single-threaded MLX compute

`top -H` on the runner during warmup:

```
Threads: 28 total, 1 running, 27 sleeping
one thread at 99.9% CPU, system 87% idle (8-thread machine)
```

Three `py-spy dump` samples (spread across the run) showed the same
main-thread stack, terminating inside MLX generation:

```
warmup
 └─ warmup_inference
    └─ mlx_generate
       └─ prefill
          └─ stream_generate
             └─ generate_step  (mlx_lm/generate.py)
```

Key observations:

- **No** polling, lock-wait, or network-receive frames in the compute
  path — this is genuine compute, not a busy-wait or deadlock.
- Between samples the line numbers inside the generation frames advanced
  (`generate.py:442` → `:460`) — the thread is *progressing* through the
  generation loop, just extremely slowly.
- EXO's warmup is effectively a **full inference pass** through `mlx_lm`,
  executed on a single core.

## 4. Same machine, same model, llama.cpp works

On the identical Dell, llama.cpp (GGML backend) loaded the same 3B model
and answered a prompt at **~10 tok/s generation**, with `mpstat` showing
3–4 CPU cores near 100% simultaneously — the multi-threaded contrast to
MLX's single hot thread.

| | GGML (llama.cpp) | MLX-CPU (EXO) |
|---|---|---|
| Threading | Multi-threaded, 3–4 cores | Single-threaded, 1 core |
| Same 3B model | ~36 s load, ~10 tok/s | Warmup > 62 min, 0 tok/s |
| On a GPU | Fast (CUDA/Metal) | Excellent (83 tok/s via MlxCuda) |

MLX is not a poor engine — on the GPU node it hit 83 tok/s via `MlxCuda`.
The issue is specifically the **MLX CPU fallback path** on non-Apple,
non-CUDA hardware.

## Why this matters

For distributed LLM inference on heterogeneous **CPU** clusters, the
choice of execution backend — not the sharding strategy or network
topology — determines whether inference completes. A framework can shard
perfectly and still never generate a token if its CPU backend is
single-threaded.

## Suggested directions (for EXO)

- Multi-threaded CPU kernels in the MLX-CPU path, or an alternative
  CPU-optimized backend (GGML-style) for non-Apple/non-CUDA nodes.
- Cache warmup artifacts so warmup cost is paid once per model/backend.
- Surface warmup progress in the API/dashboard (currently opaque).
- Add compute-speed-aware shard placement (placement currently ranks by
  memory only).

## Reproduction

```bash
# Solo EXO on the CPU node, other nodes stopped:
DEBUG=9 exo            # then load Llama-3.2-3B-Instruct-4bit from the UI
# In a second terminal, during StartWarmup:
top -H -p $(pgrep -f "exo.main" | head -1)
py-spy dump --pid <runner_pid>

# llama.cpp on the same machine for contrast:
./llama-cli -m Llama-3.2-3B-Instruct-Q4_K_M.gguf -p "test" -n 32
```

*Findings from an independent-study project comparing EXO and llama.cpp
on a heterogeneous CPU/GPU home cluster. Numbers are from academic
coursework; not affiliated with either project.*
