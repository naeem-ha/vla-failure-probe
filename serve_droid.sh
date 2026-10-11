#!/bin/bash
# pi0.5-DROID (PolaRiS co-trained, joint-position actions) policy server on :8000.
# JAX capped at 50% of GPU memory so Isaac Sim fits on the same 16 GB card.
cd ~/spais-demo/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.5} uv run scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config pi05_droid_jointpos_polaris \
  --policy.dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris
