"""
Run the top ensemble combinations with --run-name so results are saved
to Predictions/<run_name>/ and then copy to PredictionsModelTourney8_EnsembleTop/.
"""
import subprocess, shutil, sys, os
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
SCRIPT = str(Path(__file__).resolve().parent / 'ensemble3_loyo.py')
ENS_TOP = BASE / 'PredictionsModelTourney8_EnsembleTop'
ENS_TOP.mkdir(exist_ok=True)

T5_TOP = 'PredictionsModelTourney5to7_Top'
T8 = 'PredictionsModelTourney8'
PRED = 'Predictions'

# PKL aliases
PKLS = {
    'LDA_d2':   f'{T8}/13g_lda_d2full_mixup2_pca10/model.pkl',
    'LDA_core': f'{T5_TOP}/8c_lda_core_mixup2_pca20/model.pkl',
    'LR_core':  f'{T5_TOP}/8i_lr_core_mixup2_pca20_c08/model.pkl',
    'SVC_d2':   f'{T8}/13g_svc_d2full_C015_mixup2_pca8/model.pkl',
    'SVC_core': f'{T5_TOP}/11b_svc_core_C0.2_mixup2_pca20/model.pkl',
    'HGB':      f'{T5_TOP}/9c_hgb_lr001_pca20/model.pkl',
    'ET_core':  f'{T5_TOP}/9b_et_core_leaf10_mixup2_pca20/model.pkl',
    'SVC_d2_14': f'{T8}/14b_svc_d2full_C010_mixup2_pca8/model.pkl',
}

# (run_name, [pkl_keys], strategy)
# Ranked by LOYO avg test acc
ENSEMBLES = [
    ('ens3_hgb_et_lda',        ['HGB', 'ET_core', 'LDA_d2'],                   'hard'),  # 0.7587
    ('ens3_hgb_lda_svc',       ['HGB', 'LDA_d2', 'SVC_core'],                  'hard'),  # 0.7587
    ('ens5_hgb_lda_svc_lr_svcd2', ['HGB', 'LDA_d2', 'SVC_core', 'LR_core', 'SVC_d2'], 'hard'),  # 0.7571
    ('ens5_hgb_lda_svc_ldac_svcd2', ['HGB', 'LDA_d2', 'SVC_core', 'LDA_core', 'SVC_d2'], 'hard'),  # 0.7571
    ('ens3_hgb_svc_d2_svc_c',  ['HGB', 'SVC_d2', 'SVC_core'],                  'hard'),  # 0.7556
    ('ens3_hgb_lda_lr',        ['HGB', 'LDA_d2', 'LR_core'],                   'hard'),  # 0.7556
    ('ens5_lda_lr_svcd_svcc_et', ['LDA_d2', 'LR_core', 'SVC_d2', 'SVC_core', 'ET_core'], 'hard'),  # 0.7524
]


def run_ensemble(run_name, pkl_keys, strategy):
    dst = ENS_TOP / run_name
    if dst.exists():
        print(f'  SKIP (already exists): {run_name}')
        return True

    pkls = [PKLS[k] for k in pkl_keys]
    # verify all PKLs exist
    for p in pkls:
        full = BASE / p
        if not full.exists():
            print(f'  SKIP (PKL missing: {p}): {run_name}')
            return False

    cmd = [PYTHON, SCRIPT]
    for i, p in enumerate(pkls, 1):
        cmd += [f'--pkl{i}', p]
    cmd += ['--strategy', strategy, '--run-name', run_name]

    print(f'  Running: {run_name} ({len(pkls)} models, {strategy}) ...')
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE))

    pred_dir = BASE / 'Predictions' / run_name
    if pred_dir.exists():
        shutil.copytree(pred_dir, dst)
        print(f'  SAVED + COPIED -> PredictionsModelTourney8_EnsembleTop/{run_name}')
        # print LOYO acc from model_info.json
        import json
        mi = json.loads((pred_dir / 'model_info.json').read_text())
        print(f'    LOYO test acc: {mi.get("loyo_avg_test_acc")}')
        return True
    else:
        print(f'  ERROR: {run_name} - no output dir created')
        print(result.stdout[-1000:] if result.stdout else '')
        print(result.stderr[-500:] if result.stderr else '')
        return False


if __name__ == '__main__':
    print(f'Saving top {len(ENSEMBLES)} ensembles to PredictionsModelTourney8_EnsembleTop/\n')
    ok = 0
    for run_name, pkl_keys, strategy in ENSEMBLES:
        if run_ensemble(run_name, pkl_keys, strategy):
            ok += 1

    print(f'\nDone. {ok}/{len(ENSEMBLES)} ensembles saved.')
    print(f'Contents of PredictionsModelTourney8_EnsembleTop/:')
    for d in sorted(ENS_TOP.iterdir()):
        print(f'  {d.name}')
