**Table A — regression, held-out (LOSO, nested run-grouped CV, twin-style inputs)**

| UPF | Output | Runs/samples | Load (Gbps) | R² | MAE | RMSE | NMAE (%) | NRMSE (%) | MedAE | P95AE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DPDK | Power (W) | 110/4,388 | ≈0–5.0 | 0.175 | 0.00331 | 0.00415 | 0.4 | 0.5 | 0.00286 | 0.00800 |
| DPDK | Delay (µs) | 110/4,374 | ≈0–5.0 | -0.215 | 471 | 2,192 | 151.8 | 706.0 | 110 | 2,179 |
| DPDK | Loss (pkts/3 s) | 110/4,388 | ≈0–5.0 | n/a | 0 | 0 | n/a | n/a | 0 | 0 |
| OAI/USR | Power (W) | 110/4,393 | ≈0–5.0 | 0.971 | 0.0846 | 0.260 | 5.9 | 18.1 | 0.0114 | 0.351 |
| OAI/USR | Delay (µs) | 110/4,386 | ≈0–5.0 | 0.913 | 1,088 | 2,487 | 21.6 | 49.4 | 13 | 6,365 |
| OAI/USR | Loss (pkts/3 s) | 110/4,393 | ≈0–5.0 | 0.995 | 3,765 | 13,777 | 4.9 | 17.8 | 0.014 | 20,327 |

**Table B — QoS safety classification, held-out (unsafe = loss > 5 pkts/3 s or delay > 200 µs; positive = unsafe)**

| Protocol | UPF / load subset | Samples | TP | FN | FP | TN | False-safe (%) | False-unsafe (%) | Precision (unsafe) | Recall (unsafe) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LOSO | DPDK, all loads | 4,374 | 251 | 41 | 224 | 3,858 | 14.0 | 5.5 | 0.53 | 0.86 |
| LOSO | DPDK, 50–447 Mbps | 998 | 0 | 0 | 0 | 998 | n/a | 0.0 | n/a | n/a |
| LOSO | OAI/USR, all loads | 4,386 | 1,475 | 37 | 257 | 2,617 | 2.4 | 8.9 | 0.85 | 0.98 |
| LOSO | OAI/USR, 50–447 Mbps | 996 | 77 | 28 | 170 | 721 | 26.7 | 19.1 | 0.31 | 0.73 |
| LOLO | DPDK, all loads | 4,374 | 156 | 136 | 43 | 4,039 | 46.6 | 1.1 | 0.78 | 0.53 |
| LOLO | DPDK, 50–447 Mbps | 998 | 0 | 0 | 0 | 998 | n/a | 0.0 | n/a | n/a |
| LOLO | OAI/USR, all loads | 4,386 | 1,507 | 5 | 1,485 | 1,389 | 0.3 | 51.7 | 0.50 | 1.00 |
| LOLO | OAI/USR, 50–447 Mbps | 996 | 100 | 5 | 499 | 392 | 4.8 | 56.0 | 0.17 | 0.95 |

LOSO = leave-one-sweep-out (a held-out run's load level is still profiled by other sweeps). LOLO = leave-load-level-out (all runs at the held-out level withheld; tests interpolation to unprofiled loads). Near-boundary band = within a factor of 3 of the twin's USR QoS limit (149 Mbps). NMAE/NRMSE = MAE/RMSE ÷ mean measured value (n/a when the mean is 0). Full metric set, 95 % run-bootstrap CIs, other configurations and the non-held-out reference rows are in twin_validation_table.csv.
