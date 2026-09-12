import os
import time
import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "hypothesis/hypo_08_error_horizon_dynamics.ipynb",
    "hypothesis/hypo_09_spatial_shear_and_vorticity_transport.ipynb",
    "hypothesis/hypo_10_optimal_sps_calibration_front.ipynb",
    "hypothesis/hypo_11_model_blending_and_residual_orthogonality.ipynb"
]

print("=" * 80)
print("EXECUTING NEW HYPOTHESIS NOTEBOOKS (08 - 11)")
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
print("EXECUTION REPORT")
print("=" * 80)
for res in execution_summary:
    print(f"[{res['status']}] {res['notebook']:60s} ({res['time']:.2f}s)")
