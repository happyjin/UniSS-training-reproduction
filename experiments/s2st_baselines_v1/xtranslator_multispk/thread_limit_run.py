"""Run a server script or module with its CPU threads limited (infrastructure only; coordinator 2026-11-16 ~20:30 Prague:
the X-Voice servers took 500-900% CPU each, load ~777 on 192 cores). Sets OMP_NUM_THREADS / MKL_NUM_THREADS /
OPENBLAS_NUM_THREADS / TORCH_NUM_THREADS (inherited by child processes, e.g. vLLM's EngineCore) before torch is imported,
then torch.set_num_threads(n), and runs the target unchanged as __main__ with the remaining arguments:

    python thread_limit_run.py <n> <script.py> [args ...]
    python thread_limit_run.py <n> --module <module> [args ...]
"""
import os
import runpy
import sys


def main() -> None:
    n = int(sys.argv[1])
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
        os.environ[k] = str(n)
    import torch
    torch.set_num_threads(n)
    print(f"[thread_limit_run] threads={n} torch={torch.get_num_threads()} target={sys.argv[2:4]}", flush=True)
    if sys.argv[2] == "--module":
        mod = sys.argv[3]
        sys.argv = [mod] + sys.argv[4:]
        runpy.run_module(mod, run_name="__main__", alter_sys=True)
    else:
        script = sys.argv[2]
        sys.argv = [script] + sys.argv[3:]
        runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
