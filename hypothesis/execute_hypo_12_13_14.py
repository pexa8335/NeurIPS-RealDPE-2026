import os
import time
import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "hypothesis/hypo_12_optical_piv_shadow_and_scoring_bias.ipynb",
    "hypothesis/hypo_13_phase_shift_vs_amplitude_tke_decomposition.ipynb",
    "hypothesis/hypo_14_reynolds_extrapolation_and_zero_mean_invariance.ipynb"
]

print("=" * 80)
print("EXECUTING NEW HYPOTHESIS NOTEBOOKS (12 - 14)")
print("=" * 80)

execution_summary = []

for nb_path in NOTEBOOKS:
    t0 = time.time()
    print(f"\n[RUNNING] {nb_path} ...", flush=True)
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        client = NotebookClient(nb, timeout=600, kernel_name='python3')
        client.execute()

        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        elapsed = time.time() - t0
        print(f"[SUCCESS] {nb_path} executed in {elapsed:.2f}s", flush=True)
        execution_summary.append({'notebook': nb_path, 'status': 'SUCCESS', 'time': elapsed})
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[FAILED]  {nb_path} failed after {elapsed:.2f}s: {e}", flush=True)
        execution_summary.append({'notebook': nb_path, 'status': 'FAILED', 'time': elapsed, 'error': str(e)})

print("\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)
for item in execution_summary:
    print(f"{item['notebook']:<65} | {item['status']:<7} | {item['time']:.2f}s")
