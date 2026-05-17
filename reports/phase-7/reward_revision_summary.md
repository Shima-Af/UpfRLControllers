# Reward revision — v1 (flat) vs v2 (physics-grounded)

Reward magnitudes are NOT directly comparable across v1 and v2 (switching cost moved from a flat constant to a load-scaled physical quantity). What IS comparable is the *ranking* and the *gap structure*.

## Test-split results, mean ± std over 4 seeds

| Controller | Reward v1 (flat) | Reward v2 (physics) | Unsafe v1 % | Unsafe v2 % | Flips v1 | Flips v2 |
|---|---:|---:|---:|---:|---:|---:|
| MAPPO | -5610 ± 43 | -5667 ± 136 | 0.51 | 0.55 | 282 | 294 |
| IPPO-ensemble | -5787 ± 66 | -5757 ± 100 | 0.56 | 0.56 | 190 | 198 |
| centralised-PPO | -12269 ± 5186 | -14458 ± 4509 | 2.73 | 3.73 | 67 | 169 |
| always-DPDK | -7201 ± 0 | -7201 ± 0 | 0.38 | 0.38 | 0 | 0 |
| threshold(derived=81.0Mbps) | -7626 ± 0 | -7623 ± 0 | 1.42 | 1.42 | 425 | 425 |
| hysteresis(t_up=81,t_down=0,cd=1) | -7201 ± 0 | -7201 ± 0 | 0.38 | 0.38 | 0 | 0 |
| hysteresis(t_up=81,t_down=61,cd=1) | -6544 ± 0 | -6543 ± 0 | 0.94 | 0.94 | 215 | 215 |

## Ranking preservation

| Rank | v1 (flat reward) | v2 (physics reward) |
|---|---|---|
| 1 | MAPPO | MAPPO |
| 2 | IPPO-ensemble | IPPO-ensemble |
| 3 | hysteresis(t_up=81,t_down=61,cd=1) | hysteresis(t_up=81,t_down=61,cd=1) |
| 4 | always-DPDK | hysteresis(t_up=81,t_down=0,cd=1) |
| 5 | hysteresis(t_up=81,t_down=0,cd=1) | always-DPDK |
| 6 | threshold(derived=81.0Mbps) | threshold(derived=81.0Mbps) |
| 7 | centralised-PPO | centralised-PPO |

## Cooldown sensitivity (MAPPO, physics-grounded reward, seed 42)

| period | cost | test reward | energy Wh | n_switches | unsafe % |
|---:|---:|---:|---:|---:|---:|
| 4 (baseline) | 0.5 (baseline) | -5530.7 | 1972.9 | 238 | 0.47 |
| 2 | 0.5 | -5674.9 | 1955.2 | 344 | 0.58 |
| 4 | 0.1 | -6056.5 | 1969.5 | 376 | 0.71 |
| 4 | 1.0 | -5794.9 | 1958.3 | 220 | 0.62 |
| 6 | 0.5 | -5591.2 | 1967.0 | 292 | 0.51 |

Interpretation: cooldown is **not load-bearing** if all rows in this table lie within seed noise of the baseline (± ~50 reward units for MAPPO under v2).
