# Tuning Microbenchmarks

This directory contains parameter-extraction kernels for tuning Cyclotron against RTL.

Each test should emit one machine-readable line:

`TUNE_RESULT test=<name> base_latency=<n> bytes_per_cycle=<n> queue_capacity=<n> completions_per_cycle=<n> cycles=<n> ops=<n>`

Conventions:
- `base_latency`: dependency-chain latency estimate.
- `bytes_per_cycle`: throughput for byte-transported resources.
- `queue_capacity`: queue fill depth before reject.
- `completions_per_cycle`: sustained completion bandwidth estimate.

Non-applicable fields are set to `0` for a test.
