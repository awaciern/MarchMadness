"""Temporary sweep script - find best model+feature combo by avg bracket score."""
import subprocess

PYTHON = '/Users/averyacierno/Desktop/Coding/MarchMadness/env/bin/python3'

models = ['random_forest', 'decision_tree', 'gp', 'svc_linear', 'adaboost', 'logistic_lbfgs']
feature_sets = [
    ['AdjEM', 'AdjO', 'AdjD', 'AdjT', 'WinPct'],
    ['AdjEM', 'AdjO', 'AdjD'],
    ['AdjEM', 'WinPct', 'Luck'],
    ['AdjEM', 'AdjO', 'AdjD', 'AdjT', 'WinPct', 'Conf', 'Seed'],
    ['AdjEM', 'Conf', 'Seed'],
    ['AdjEM', 'AdjO', 'AdjD', 'Conf', 'Seed'],
    ['AdjEM', 'AdjO', 'AdjD', 'AdjT', 'WinPct', 'SOS_AdjEM', 'Rk_NCSOS_AdjEM', 'Conf', 'Seed'],
    ['AdjEM'],
    ['AdjEM', 'Seed'],
    ['WinPct', 'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'SOS_AdjEM', 'Rk_NCSOS_AdjEM', 'Conf', 'Seed'],
]

results = []
total = len(models) * len(feature_sets)
done = 0
for m in models:
    for feats in feature_sets:
        done += 1
        cmd = [PYTHON, 'Python/predict_brackets.py', '-m', m, '--features'] + feats
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        for line in out.splitlines():
            if 'Results saved' in line:
                folder = line.split('Predictions/')[-1].strip()
                parts = folder.split('_')
                score = int(parts[2])
                results.append((score, folder))
                print(f'[{done}/{total}] {score:5d}  {folder}', flush=True)

results.sort(reverse=True)
print('\n--- TOP 15 ---')
for score, folder in results[:15]:
    print(f'  {score:5d}  {folder}')
