# exo-cluster-research
Independent study project exploring EXO for distributed LLM inference on heterogeneous GPU/CPU clusters.
## Course Information

Independent Study Project  
Professor: Dr. Iraklis Anagnostopoulos  
Summer 2026
## Current Status

Completed:
- GitHub repository created
- EXO installed and launched
- Dashboard/API verified
- Initial heterogeneous cluster testing performed
- Bug report submitted to EXO developers

In Progress:
- Installation guide documentation
- Benchmarking model inference performance

Known Issues:
- CPU worker selected over CUDA GPU in heterogeneous cluster
- WSL networking / synchronization challenges
- ## Describe the bug

In a two-node heterogeneous EXO cluster with one CUDA-capable GPU node and one CPU-only node, EXO repeatedly assigned full model execution to the CPU worker while the GPU worker remained idle. This resulted in severe inference slowdown or apparent stalls.

## To Reproduce

Steps to reproduce the behavior:
1. Start EXO on a WSL2 node with NVIDIA GPU and CUDA backend available
2. Connect a second CPU-only Linux node using Zenoh
3. Load a model and begin inference
4. Observe worker assignment in logs

## Expected behavior

The scheduler should prefer the CUDA-capable GPU node for inference, or distribute workload according to hardware capability.

## Actual behavior

Cluster formation succeeds, but logs indicate:
- `world_size = 1`
- full model assigned to CPU worker
- GPU node remains idle during inference

Inference becomes extremely slow or appears stalled.

## Environment

- macOS Version: N/A (using Windows 11 + WSL2 and Linux)
- EXO Version: commit `09f9ea31` (Zenoh branch)
- Hardware:
  - Device 1: MSI laptop, x86_64, NVIDIA GPU (CUDA available), Windows 11 + WSL2
  - Device 2: Dell laptop, x86_64, Linux, CPU-only
  - Additional devices: none
- Interconnection:
  - Gigabit Ethernet / local network via Zenoh

## Additional context

Leader election appears deterministic and based on Zenoh candidate ordering:

`elected = max(candidates)`

This may influence cluster topology and downstream scheduling behavior.

Workaround used:
- GPU node: `--force-master`
- CPU node: `--no-worker`

This preserved cluster connectivity while preventing full model execution on the CPU-only node.

Possible improvement:
Consider hardware-aware worker selection based on:
- GPU availability
- backend capability
- available VRAM
- historical throughput
- ## External Contributions

- Reported EXO scheduling bug involving incorrect CPU worker selection in heterogeneous CUDA/CPU cluster
- Issue reference: https://github.com/exo-explore/exo/issues/2180
