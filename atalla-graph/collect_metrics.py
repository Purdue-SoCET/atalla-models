"""Collect per-kernel and end-to-end metrics for the Atalla pipeline."""
from __future__ import annotations
import json, sys, os, torch, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_model import run_pipeline

torch.manual_seed(42)
np.random.seed(42)


def run_basic():
    from model.basic import BasicModule
    model = BasicModule(dim=32, depth=2)
    inp = torch.randn(1, 32)
    return run_pipeline(model, inp, out_dir="out/metrics_basic", verbose=True)


def run_alexnet():
    from model.alexnet import AlexNetSmall
    model = AlexNetSmall(scale=0.01, num_classes=10)
    inp = torch.randn(1, 3, 32, 32)
    return run_pipeline(model, inp, out_dir="out/metrics_alexnet", verbose=True)


def print_kernel_table(km_list, model_name):
    print(f"\n{'='*90}")
    print(f"  Per-Kernel Metrics: {model_name}")
    print(f"{'='*90}")
    hdr = f"{'Node':<14} {'Op':<8} {'Backend':<8} {'Elems':>6} {'Range':>22} {'CosSim':>8} {'MaxDiff':>9} {'Cycles':>8} {'Instrs':>8} {'GEMMs':>6}"
    print(hdr)
    print("-" * len(hdr))
    for k in km_list:
        rng = f"[{k['min']:.4f}, {k['max']:.4f}]"
        cs = f"{k.get('cos_sim', float('nan')):.4f}" if 'cos_sim' in k else "n/a"
        md = f"{k.get('max_diff', float('nan')):.6f}" if 'max_diff' in k else "n/a"
        cy = str(k.get('cycles', '')) if k['backend'] == 'emulator' else ''
        ins = str(k.get('instructions', '')) if k['backend'] == 'emulator' else ''
        gm = str(k.get('gemm_ops', '')) if k['backend'] == 'emulator' else ''
        print(f"{k['name']:<14} {k['op']:<8} {k['backend']:<8} {k['elems']:>6} {rng:>22} {cs:>8} {md:>9} {cy:>8} {ins:>8} {gm:>6}")


def print_summary(results, model_name):
    print(f"\n--- End-to-End Summary: {model_name} ---")
    s = results["stats"]
    print(f"  Total nodes: {s['nodes_total']}, emulated: {s['nodes_emulated']}, "
          f"numpy: {s['nodes_numpy']}, passthrough: {s['nodes_passthrough']}")
    print(f"  Wall time: {results['elapsed_s']:.2f}s")
    if "cosine_sim" in results:
        print(f"  Final cosine sim: {results['cosine_sim']:.6f}")
        print(f"  Final max diff:   {results['max_diff']:.6f}")
        print(f"  Final mean diff:  {results['mean_diff']:.6f}")

    km = results.get("kernel_metrics", [])
    emu_kernels = [k for k in km if k["backend"] == "emulator"]
    if emu_kernels:
        total_cycles = sum(k.get("cycles", 0) for k in emu_kernels)
        total_instrs = sum(k.get("instructions", 0) for k in emu_kernels)
        total_gemms = sum(k.get("gemm_ops", 0) for k in emu_kernels)
        any_nan = any(k.get("has_nan", False) for k in emu_kernels)
        cos_sims = [k["cos_sim"] for k in emu_kernels if "cos_sim" in k]
        print(f"  Total cycles (emulated): {total_cycles}")
        print(f"  Total instructions: {total_instrs}")
        print(f"  Total GEMM ops: {total_gemms}")
        print(f"  Any NaN in output: {any_nan}")
        if cos_sims:
            print(f"  Per-kernel cos_sim range: [{min(cos_sims):.4f}, {max(cos_sims):.4f}]")


if __name__ == "__main__":
    print("=" * 90)
    print("  ATALLA PIPELINE METRICS COLLECTION")
    print("=" * 90)

    print("\n>>> Running BasicModule...")
    r_basic = run_basic()
    print_kernel_table(r_basic["kernel_metrics"], "BasicModule(dim=32, depth=2)")
    print_summary(r_basic, "BasicModule")

    print("\n\n>>> Running AlexNetSmall...")
    r_alex = run_alexnet()
    print_kernel_table(r_alex["kernel_metrics"], "AlexNetSmall(scale=0.01)")
    print_summary(r_alex, "AlexNetSmall")

    out = {"basic": {k: v for k, v in r_basic.items()
                     if k not in ("emulator_output", "reference_output")},
           "alexnet": {k: v for k, v in r_alex.items()
                       if k not in ("emulator_output", "reference_output")}}
    Path("out/metrics.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nMetrics saved to out/metrics.json")
