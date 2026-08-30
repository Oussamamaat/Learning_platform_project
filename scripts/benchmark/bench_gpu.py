"""
GPU / CUDA sanity check — run FIRST, inside the container's app venv.
    /app/.gguf_venv/bin/python scripts/benchmark/bench_gpu.py

Confirms the rented box is actually an RTX 5090 (Blackwell, compute capability
12.0) with a CUDA build that supports it. A wrong base image (CUDA < 12.8) shows
up here as either torch.cuda.is_available()==False or a "no kernel image" error
on the first CUDA op — this catches that before any model is loaded.

Exit code 0 = usable GPU; 2 = no CUDA; 3 = CUDA present but sm_120 unsupported.
Prints a JSON summary on the last line for bench_all.sh to capture.
"""
import json
import subprocess
import sys


def _nvidia_smi() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        return {"error": f"nvidia-smi unavailable: {e}"}
    name, driver, total, free, used = [x.strip() for x in out.split(",")]
    return {
        "name": name, "driver_version": driver,
        "memory_total_mib": int(float(total)),
        "memory_free_mib": int(float(free)),
        "memory_used_mib": int(float(used)),
    }


def main() -> int:
    summary = {"nvidia_smi": _nvidia_smi()}
    print("── nvidia-smi ──")
    for k, v in summary["nvidia_smi"].items():
        print(f"  {k}: {v}")

    print("\n── torch.cuda ──")
    try:
        import torch
    except ImportError:
        print("  torch not importable in this venv", file=sys.stderr)
        summary["torch"] = {"error": "not installed"}
        print(json.dumps(summary))
        return 2

    avail = torch.cuda.is_available()
    summary["torch"] = {"version": torch.__version__, "cuda_available": avail,
                        "cuda_build": torch.version.cuda}
    print(f"  torch {torch.__version__} (built for CUDA {torch.version.cuda})")
    print(f"  cuda_available: {avail}")
    if not avail:
        print("  ERROR: CUDA not available to torch — wrong base image or no GPU passthrough",
              file=sys.stderr)
        print(json.dumps(summary))
        return 2

    idx = torch.cuda.current_device()
    cc_major, cc_minor = torch.cuda.get_device_capability(idx)
    props = torch.cuda.get_device_properties(idx)
    summary["torch"].update({
        "device_name": props.name,
        "compute_capability": f"{cc_major}.{cc_minor}",
        "total_vram_gib": round(props.total_memory / (1024**3), 1),
        "arch_list": torch.cuda.get_arch_list(),
    })
    print(f"  device: {props.name}")
    print(f"  compute capability: {cc_major}.{cc_minor}  (RTX 5090 = 12.0)")
    print(f"  total VRAM: {summary['torch']['total_vram_gib']} GiB")
    print(f"  torch arch_list: {summary['torch']['arch_list']}")

    # Real op — surfaces a "no kernel image for sm_120" error a capability
    # string alone would hide.
    print("\n── live CUDA op (matmul on device) ──")
    try:
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        (a @ b).sum().item()
        torch.cuda.synchronize()
        print("  OK — matmul ran on the GPU")
        summary["live_op"] = "ok"
    except Exception as e:
        print(f"  ERROR: CUDA op failed — sm_{cc_major}{cc_minor} likely unsupported by this "
              f"torch build:\n  {e}", file=sys.stderr)
        summary["live_op"] = f"error: {e}"
        print(json.dumps(summary))
        return 3

    print("\nRESULT: GPU usable ✓")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
