# Scripts

This folder contains helper scripts used for EXO and llama.cpp experiments.

Scripts:
- `start-exo-master.sh` — starts EXO on the GPU/master node
- `start-exo-worker.sh` — starts EXO on CPU worker nodes
- `start-rpc-worker.sh` — starts a llama.cpp RPC worker
- `llama-bench-rpc.sh` — runs the llama.cpp RPC benchmark
- `profile_exo_warmup.sh` — diagnoses whether an EXO node is stuck in single-threaded MLX-CPU warmup (captures top -H, py-spy, mpstat; prints a compute-bound vs sync-bound verdict)
