## Current Results Summary

| Exp | Framework | Nodes | Key Result | Outcome |
|---|---|---|---|---|
| E1 | EXO | MSI GPU | ~84 tok/s | GPU baseline |
| E2 | EXO | Dell + Mini | Prefill stalled | CPU-only failure |
| E3 | EXO | MSI + Dell | ~86 tok/s | GPU restored with workaround |
| E4 | llama.cpp | Dell | 2.44 tok/s | CPU baseline |
| E5 | llama.cpp | Dell + Mini | +16.0% generation | Distributed gain |
| E6 | llama.cpp | Dell + Mini + Pi 5 | Prefill recovered | Mixed scaling |
