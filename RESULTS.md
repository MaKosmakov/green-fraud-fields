# Results snapshot

This is a compact pointer to the final results reproduced by the scripts. Full raw outputs are intentionally not committed.

## IEEE-CIS

Strict timestamp-block policy:

- all rows with the same `TransactionDT` are scored from a frozen pre-timestamp state;
- candidate transactions are excluded from their own graph features;
- labels enter released history only when available before scoring;
- train/validation/test roles are kept separate.

Headline graph-vs-history result:

| Comparison | Mean AUC-PR gain | AUC wins | Mean P@1% gain | P@1% wins |
|---|---:|---:|---:|---:|
| `M3 + H_raw + S_D` vs `M3 + H_raw` | +0.0261 | 4/5 | +0.130 | 5/5 |

Tail-ranking result:

| Model | Mean AUC-PR gain vs `M3` | Mean P@1% gain vs `M3` |
|---|---:|---:|
| adaptive two-stage | +0.1749 | +0.399 |
| cross-fit logistic tail | +0.1735 | +0.408 |

Delay sweep AUC-PR gains over raw history:

| Delay | Static Green | Adaptive two-stage | Cross-fit logistic tail |
|---:|---:|---:|---:|
| 0 | +0.0261 | +0.1103 | +0.1042 |
| 1 | -0.0056 | +0.0589 | +0.0631 |
| 3 | +0.0004 | +0.0563 | +0.0571 |
| 7 | -0.0097 | +0.0580 | +0.0575 |
| 14 | -0.0060 | +0.0622 | +0.0355 |

The delay-sweep figure is reproduced by:

```powershell
python scripts/plot_delay_auc_sweep.py --output-dir figures
```

## Elliptic++

The external check uses strict time-step blocks, excludes class-3 unknown labels from train/evaluation, and never uses official wallet labels as features.

| Model | AUC-PR | P@1% |
|---|---:|---:|
| strict transaction features | 0.5976 | 0.970 |
| raw released address history | 0.6299 | 1.000 |
| best combined exact-Green variant tested | 0.6088 | 0.970 |

Interpretation: Elliptic++ supports the released-history construction but not the Green-smoothing claim under the tested address-graph design.
