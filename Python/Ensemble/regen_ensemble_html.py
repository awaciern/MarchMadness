"""Regenerate HTML for all ensemble model directories."""
import sys, json
sys.path.insert(0, 'Python')
import predict_year as py
from pathlib import Path

ensemble_dirs = [
    'Predictions/ENS_HGB_ET_LDA',
    'Predictions/ENS_HGB_LDA_LR',
    'Predictions/ENS_HGB_LDA_SVC',
    'Predictions/ENS_HGB_LDA_SVC_LDA_C_SVC_D2',
    'Predictions/ENS_HGB_LDA_SVC_LR_SVC_D2',
    'Predictions/ENS_HGB_SVC_D2_SVC_C',
    'PredictionsModelTourney8_EnsembleTop/ens3_hgb_et_lda',
    'PredictionsModelTourney8_EnsembleTop/ens3_hgb_lda_lr',
    'PredictionsModelTourney8_EnsembleTop/ens3_hgb_lda_svc',
    'PredictionsModelTourney8_EnsembleTop/ens3_hgb_svc_d2_svc_c',
    'PredictionsModelTourney8_EnsembleTop/ens5_hgb_lda_svc_ldac_svcd2',
    'PredictionsModelTourney8_EnsembleTop/ens5_hgb_lda_svc_lr_svcd2',
]

data_root = Path('.')
ff = [(0, 1), (2, 3)]

for d in ensemble_dirs:
    pred_dir = Path(d)
    mi = json.loads((pred_dir / 'model_info.json').read_text())
    # Find all years that already have an HTML file in this directory
    html_years = sorted(int(p.stem) for p in pred_dir.glob('*.html')
                        if p.stem.isdigit() and 'bracket' not in p.stem)
    if not html_years:
        html_years = [2025]
    print(f'Regenerating {pred_dir.name} for years {html_years} ...')
    for year in html_years:
        py._run_ensemble(pred_dir, mi, data_root, year, ff_pairings=ff)

print('Done')
