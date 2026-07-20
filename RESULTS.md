# Results snapshot

This is a compact pointer to the final results reproduced by the scripts. Full raw outputs are intentionally not committed.

## IEEE-CIS

Strict timestamp-block policy:

- all rows with the same `TransactionDT` are scored from a frozen pre-timestamp state;
- candidate transactions are excluded from their own graph features;
- labels enter released history only when available before scoring;
- train/validation/test roles are kept separate.
- tail cutoffs are frozen on validation and applied pointwise to test rows;
- alert-score ties are resolved by stable chronological arrival order.

Headline graph-vs-history result:

| Comparison | Mean AUC-PR gain | AUC wins | Mean P@1% gain | P@1% wins |
|---|---:|---:|---:|---:|
| `M3 + H_raw + S_D` vs `M3 + H_raw` | +0.0202 | 3/5 | +0.080 | 5/5 |

Tail-ranking result:

| Model | Mean AUC-PR gain vs `M3` | Mean P@1% gain vs `M3` |
|---|---:|---:|
| Green tail reranker | +0.1714 | +0.426 |
| cross-fitted Green tail reranker | +0.1735 | +0.420 |

Delay sweep AUC-PR gains over raw history:

| Delay | Static Green | Adaptive two-stage | Cross-fit logistic tail |
|---:|---:|---:|---:|
| 0 | +0.0202 | +0.0889 | +0.0910 |
| 1 | -0.0210 | +0.0430 | +0.0421 |
| 3 | +0.0164 | +0.0641 | +0.0656 |
| 7 | -0.0059 | +0.0596 | +0.0613 |
| 14 | +0.0060 | +0.0671 | +0.0395 |

The delay-sweep figure is reproduced by:

```powershell
python scripts/plot_delay_auc_sweep.py --output-dir figures
```

These are the stable-tie, pointwise-tail results. Earlier test-batch-ranked scores are diagnostic only and are not part of the frozen headline.

The final causality audit passes future-edge exclusion, candidate self-exclusion, strict label release, timestamp-block scoring, cache provenance, validation/test separation, and the label-permutation placebo. Under permutation, the adaptive graph marginal over raw history shrinks from `+0.0202` to `+0.0012` AUC-PR and from `+0.080` to `-0.009` P@1%.

## Elliptic++

The external check uses strict time-step blocks, excludes class-3 unknown labels from train/evaluation, and never uses official wallet labels as features.

| Model | AUC-PR | P@1% |
|---|---:|---:|
| strict transaction features | 0.5976 | 0.970 |
| raw released address history | 0.6299 | 1.000 |
| best combined exact-Green variant tested | 0.6088 | 0.970 |

Interpretation: Elliptic++ supports the released-history construction but not the Green-smoothing claim under the tested address-graph design.
