# Experiment manifests

This folder contains the machine-readable provenance used for the paper results.

- `ieee_green_block_causal_manifest.json` records the strict timestamp-block causal policy, split sizes, Green-field settings, model-selection grids, diagnostics, source scripts, and SHA-256 hashes of the generated result summaries.
- `environment.txt` records the Python and core package versions used for the public reproducibility check.

Generated experiment outputs are intentionally not committed. The hashes in the manifest refer to the generated summaries produced by the scripts under `scripts/`.
