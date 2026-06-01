"""
MPI Strong Scaling Benchmark
Tests parallel efficiency using mpi4py on a shared-memory node.

Usage:
    mpirun -np 1 python3 mpi_scaling.py
    mpirun -np 2 python3 mpi_scaling.py
    mpirun -np 4 python3 mpi_scaling.py
"""

from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 10_000_000
local_N = N // size
data = np.random.rand(local_N)

comm.Barrier()
start = time.time()

local_sum = np.sum(data ** 2)
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

comm.Barrier()
elapsed = time.time() - start

if rank == 0:
    print(f"ranks={size} time={elapsed:.4f}s result={total_sum:.6f}")
