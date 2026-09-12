import os
import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    print('Copying codebase from dataset...')
    run_cmd('cp -r /kaggle/input/realpde-code-v4/CCCCCC .')
    run_cmd('cp -r /kaggle/input/realpde-code-v4/research .')

    print('Copying checkpoint...')
    checkpoint_dir = '/kaggle/input/realpde-step-0-checkpoints'
    os.makedirs('CCCCCC/submissions/sub1_cno_sps', exist_ok=True)
    os.system(f'cp {checkpoint_dir}/sim_real_cno.pth CCCCCC/submissions/sub1_cno_sps/')

    competition_data = '/kaggle/input/realpde-competition-data'
    print('Setting up data symlinks for build_cache...')
    os.makedirs('Data/train_sim-002', exist_ok=True)
    os.makedirs('Data/train_real-001', exist_ok=True)
    os.system(f'ln -s {competition_data}/train_sim/train_sim Data/train_sim-002/train_sim')
    os.system(f'ln -s {competition_data}/train_real/train_real Data/train_real-001/train_real')

    print('--- BUILDING CACHE ---')
    run_cmd('python CCCCCC/src/build_cache.py --split real')
    run_cmd('python CCCCCC/src/build_cache.py --split sim')

    print('--- RUNNING STAGE 1 (H2 VERIFICATION) ---')
    run_cmd('python research/architecture_audit/stage1_transport_viability.py')
    os.system('cp -r research/architecture_audit/results /kaggle/working/stage1_results')

    print('--- RUNNING MODEL B BASELINE ---')
    run_cmd('python CCCCCC/src/train.py --model cno --init CCCCCC/submissions/sub1_cno_sps/sim_real_cno.pth --out runs/train/cno_baseline --epochs 50 --batch 4')

    print('--- RUNNING MODEL B+T TRANSPORT ---')
    run_cmd('python research/architecture_audit/train_transport.py --model cno --init CCCCCC/submissions/sub1_cno_sps/sim_real_cno.pth --out runs/train/cno_transport --epochs 50 --batch 4')

    print('--- RUNNING EVALUATION (H3 VERIFICATION) ---')
    run_cmd('python research/architecture_audit/evaluate_horizon.py')

    print('Copying results to /kaggle/working...')
    os.system('cp -r runs/train/cno_baseline /kaggle/working/')
    os.system('cp -r runs/train/cno_transport /kaggle/working/')
    if os.path.exists('research/architecture_audit/results'):
        os.system('cp -r research/architecture_audit/results /kaggle/working/final_results')

    print('ALL EXPERIMENTS COMPLETED!')

if __name__ == '__main__':
    main()
