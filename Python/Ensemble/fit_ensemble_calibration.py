"""Fit and store post-combination calibration temperature T for all ensemble models.

For each ensemble, loads all component model PKLs, collects combined ensemble
probabilities across all available historical bracket rounds, and uses the stretch
calibration algorithm (same as fit_temperature_stretch in predict_brackets.py) to
find T such that the 98th-percentile logit maps to logit(p_target).

T is stored in the ensemble's model_info.json under 'ensemble_calibrate_temperature'.
"""

import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'Python'))
import predict_year as py

DATA_ROOT = Path(__file__).resolve().parent.parent

ENSEMBLE_DIRS = [
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

P_TARGET = 0.97   # target max win probability after calibration


def load_components(pred_dir: Path, model_info: dict, data_root: Path) -> list:
    component_names = model_info.get('feature_bases', [])
    components = []
    for name in component_names:
        pkl_path = py.find_component_pkl(data_root, name)
        if pkl_path is None:
            raise FileNotFoundError(f'Cannot find model.pkl for component: {name}')
        with open(pkl_path, 'rb') as fh:
            payload = pickle.load(fh)
        fl = payload['feature_list']
        components.append({
            'model':              payload['model'],
            'feature_list':       fl,
            'model_feature_list': payload.get('model_feature_list', fl),
            'cat_encoders':       payload.get('cat_encoders', {}),
            'norm_info':          payload.get('norm_info', None),
            'delta_feats':        payload.get('delta_feats', False),
            'numeric_bases':      payload.get('numeric_bases', []),
            'pca_transformer':    payload.get('pca_transformer', None),
        })
    return components


if __name__ == '__main__':
    for d in ENSEMBLE_DIRS:
        pred_dir = DATA_ROOT / d
        info_path = pred_dir / 'model_info.json'
        model_info = json.loads(info_path.read_text())
        strategy = model_info.get('model_params', {}).get('strategy', 'hard')
        weights  = model_info.get('model_params', {}).get('weights', None)
        exclude  = model_info.get('exclude_years', [])

        print(f'\n{pred_dir.name} (strategy={strategy}, exclude={exclude})')
        print(f'  Loading {len(model_info["feature_bases"])} components ...')
        components = load_components(pred_dir, model_info, DATA_ROOT)

        T = py.fit_ensemble_calibration_T(
            components=components,
            data_root=DATA_ROOT,
            strategy=strategy,
            weights=weights,
            exclude_years=exclude,
            p_target=P_TARGET,
        )

        model_info['ensemble_calibrate_temperature'] = round(T, 6)
        model_info['ensemble_calibrate_target']      = P_TARGET
        info_path.write_text(json.dumps(model_info, indent=2))
        print(f'  Stored T={T:.4f} in {info_path}')

    print('\nDone.')
