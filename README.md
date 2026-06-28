# MPI Strong Scaling Benchmark

MPI parallel scaling benchmark using mpi4py on a Linux ARM64 node.
Developed and tested on Ubuntu 24.04 (Lima VM on Apple M1).

## What it measures

**Strong scaling**: same total problem size, increasing number of MPI ranks.

The theoretical upper bound (Amdahl's law, serial fraction → 0) is a speedup
equal to the number of ranks: 2 ranks → at most 2×, 4 ranks → at most 4×.
Any reported speedup exceeding the rank count indicates a measurement artefact
(noise, timer resolution, oversubscription, or memory effects), not genuine
parallel acceleration.

## Methodology

A correct strong-scaling benchmark requires:

1. **Compute-bound kernel** — e.g. Monte Carlo estimation of π: each rank draws
   `N_total // size` random samples, computes a local partial sum, and the root
   gathers all partial sums with `comm.reduce`. This is CPU-bound and scales
   predictably, unlike a simple array sum which is memory-bandwidth-bound.

2. **Fixed total problem size** — `N_total` is constant; `local_N = N_total // size`
   shrinks as ranks increase. This is the definition of strong scaling.

3. **Proper timing** — `MPI.Wtime()` (not `time.time()`), with `comm.Barrier()`
   immediately before and after the timed region on every rank.

4. **Warm-up run** — one untimed iteration before the measurement loop to fill
   caches and stabilise the MPI runtime.

5. **Statistical robustness** — at least 5 timed repetitions; report the median
   wall time, not the minimum or a single run.

6. **No oversubscription** — `mpirun -np N` with N ≤ physical core count.

7. **Reference curve** — plot both measured speedup and the ideal linear line
   (`speedup = ranks`) to make efficiency losses immediately visible.

8. **Output** — write one CSV row per (ranks, repetition, wall_time) for
   reproducibility.

## Results

Benchmark methodology corrected. Quantitative results pending measurement on
validated hardware.

## How to run

Install dependencies:

    pip install mpi4py numpy

Run benchmark (replace N with 1, 2, 4 — never exceeding physical core count):

    mpirun -np 1 python3 mpi_scaling.py --samples 200000000
    mpirun -np 2 python3 mpi_scaling.py --samples 200000000
    mpirun -np 4 python3 mpi_scaling.py --samples 200000000

Results are written to `results/scaling_results.csv`.

## Environment

- OS: Ubuntu 24.04 LTS ARM64
- MPI: OpenMPI 4.x
- mpi4py: 4.1.2
- Lima VM on Apple M1
