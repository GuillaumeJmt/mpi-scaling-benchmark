"""
MPI Strong Scaling Benchmark — Monte Carlo π estimation.

Correct strong-scaling design:
  - Compute-bound kernel (CPU random sampling, not memory bandwidth)
  - Fixed total sample count divided across ranks
  - MPI.Wtime() + comm.Barrier() for timing
  - One untimed warm-up iteration
  - 5 timed repetitions, median reported
  - Results written to results/scaling_results.csv

Usage:
    mpirun -np 1 python3 mpi_scaling.py [--samples N]
    mpirun -np 2 python3 mpi_scaling.py [--samples N]
    mpirun -np 4 python3 mpi_scaling.py [--samples N]

Never exceed the physical core count with -np.
"""

import argparse
import csv
import os
import numpy as np
from mpi4py import MPI

REPETITIONS = 5


def monte_carlo_pi(local_n: int, rng: np.random.Generator) -> int:
    """Count how many of local_n uniform random points fall inside unit circle."""
    x = rng.uniform(-1.0, 1.0, local_n)
    y = rng.uniform(-1.0, 1.0, local_n)
    return int(np.sum(x**2 + y**2 <= 1.0))


def run_once(comm, total_samples: int, rng: np.random.Generator) -> float:
    """One timed measurement; returns wall time in seconds (root rank)."""
    size = comm.Get_size()
    rank = comm.Get_rank()
    local_n = total_samples // size

    comm.Barrier()
    t_start = MPI.Wtime()

    local_hits = monte_carlo_pi(local_n, rng)
    total_hits = comm.reduce(local_hits, op=MPI.SUM, root=0)

    comm.Barrier()
    t_end = MPI.Wtime()

    elapsed = t_end - t_start
    # Broadcast elapsed from root so all ranks have the same value
    elapsed = comm.bcast(elapsed, root=0)

    if rank == 0 and total_hits is not None:
        pi_approx = 4.0 * total_hits / total_samples
        # Suppress unused-variable warning: pi_approx available for debugging
        _ = pi_approx

    return elapsed


def main():
    parser = argparse.ArgumentParser(description="MPI strong scaling benchmark")
    parser.add_argument(
        "--samples",
        type=int,
        default=200_000_000,
        help="Total sample count (fixed across all rank counts)",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Each rank uses a distinct RNG seed for statistical independence
    rng = np.random.default_rng(seed=42 + rank)

    # Warm-up (not recorded)
    run_once(comm, args.samples, rng)

    # Timed repetitions
    times = []
    for _ in range(REPETITIONS):
        t = run_once(comm, args.samples, rng)
        times.append(t)

    if rank == 0:
        median_time = float(np.median(times))
        print(
            f"ranks={size}  samples={args.samples}  "
            f"median_time={median_time:.4f}s  "
            f"(over {REPETITIONS} repetitions)"
        )

        os.makedirs("results", exist_ok=True)
        csv_path = "results/scaling_results.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["ranks", "samples", "rep", "wall_time_s"])
            for i, t in enumerate(times):
                writer.writerow([size, args.samples, i + 1, f"{t:.6f}"])


if __name__ == "__main__":
    main()
