import os
import glob
import time
import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "hypothesis/hypo_01_grid_and_geometry.ipynb",
    "hypothesis/hypo_02_reynolds_aoa_distribution.ipynb",
    "hypothesis/hypo_03_temporal_stationarity_and_drift.ipynb",
    "hypothesis/hypo_04_sim_vs_real_discrepancy.ipynb",
    "hypothesis/hypo_05_causal_advection_coherence.ipynb",
    "hypothesis/hypo_06_spectral_energy_cascade.ipynb",
    "hypothesis/hypo_07_spatial_uncertainty_structure.ipynb"
]

print("=" * 80)
print("EXECUTING ALL 7 REALPDE HYPOTHESIS NOTEBOOKS")
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

        # Save freshly executed notebook back
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        elapsed = time.time() - t0
        print(f"[SUCCESS] {nb_path} executed in {elapsed:.2f}s", flush=True)
        
        # Grab output text from code cells
        outputs = []
        for cell in nb.cells:
            if cell.cell_type == 'code':
                for out in cell.get('outputs', []):
                    if 'text' in out:
                        outputs.append(out['text'])

        execution_summary.append({
            'notebook': nb_path,
            'status': 'SUCCESS',
            'time': elapsed,
            'output_snippet': "".join(outputs)[:500]
        })
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[FAILED]  {nb_path} failed after {elapsed:.2f}s: {e}", flush=True)
        execution_summary.append({
            'notebook': nb_path,
            'status': 'FAILED',
            'time': elapsed,
            'error': str(e)
        })

print("\n" + "=" * 80)
print("FINAL EXECUTION REPORT")
print("=" * 80)
for res in execution_summary:
    status_tag = f"[{res['status']}]"
    print(f"{status_tag:10s} {res['notebook']:55s} ({res['time']:.2f}s)")

print("\nAll executions finished successfully!")
