"""Custom Triton kernels.

Each kernel has:
    - A PyTorch reference implementation (correctness target)
    - The Triton kernel
    - A correctness test (asserts numerical equivalence within tolerance)
    - A benchmark sweep (performance characterization across input sizes)
"""
