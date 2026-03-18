"""
debug_ensemble.py — Compare standalone LOYO fold vs. predict_brackets results
"""
import sys, pickle
sys.path.insert(0, 'Python')
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from predict_brackets import (
    load_combined_games, apply_year_norm,
    apply_delta_transform, mirror_augment, fit_global_scaler_delta,
    build_and_train_model,
)

BASE = Path('.')
EXCLUDE = {2012, 2013, 2014}

# Load PKL config for LDA d2full pca10
with open('Predictions/13g_lda_d2full_mixup2_pca10/model.pkl', 'rb') as f:
    pkl = pickle.load(f)

feature_list       = pkl['feature_list']
model_feature_list = pkl['model_feature_list']
numeric_bases      = pkl['numeric_bases']
pca_n              = pkl['pca_transformer'].n_components_

print(f'features: {len(feature_list)} raw, {len(model_feature_list)} delta, pca={pca_n}')

# Load real data
df_all = load_combined_games(BASE)
df_all = df_all.dropna(subset=[c for c in feature_list if c in df_all.columns])
print(f'df_all rows: {len(df_all)} (expected: 819)')

# Fit norm globally on all years (matches predict_brackets.py --norm-all behavior)
norm_info = fit_global_scaler_delta(df_all, numeric_bases)

# Load Mixup2 sim data
df_sim = pd.read_csv('Data/SimulatedDataMixup2/All.csv')
print(f'sim rows total: {len(df_sim)}')

# Per-year accuracy comparison
expected = {
    2015: 0.8413, 2016: 0.7460, 2017: 0.7302, 2018: 0.6508,
    2019: 0.7937, 2021: 0.6825, 2022: 0.7302, 2023: 0.7143,
    2024: 0.7302, 2025: 0.8254,
}

print(f'\n{"Year":>4}  {"My LOYO":>8}  {"Expected":>9}  {"Match?":>7}')
print('=' * 40)

for test_year in [y for y in sorted(df_all['Year'].unique()) if y not in EXCLUDE]:
    train_mask = ~df_all['Year'].isin({test_year} | EXCLUDE)
    df_train = df_all[train_mask].copy()

    # Add sim data (exclude test_year)
    sim_mask = ~df_sim['Year'].isin({test_year} | EXCLUDE)
    df_sim_slice = df_sim[sim_mask].copy()
    shared_cols = [c for c in df_sim_slice.columns if c in df_train.columns]
    df_train = pd.concat([df_train, df_sim_slice[shared_cols]], ignore_index=True)

    # Preprocess training
    df_train = apply_year_norm(df_train, norm_info)
    df_train = apply_delta_transform(df_train, numeric_bases)
    df_train = mirror_augment(df_train, model_feature_list)

    X_tr = df_train[model_feature_list]
    y_tr = df_train['Win__1'].reset_index(drop=True)

    # PCA
    pc_cols = [f'PC{i}' for i in range(pca_n)]
    fold_pca = PCA(n_components=pca_n, random_state=42)
    X_tr_pca = pd.DataFrame(fold_pca.fit_transform(X_tr), columns=pc_cols)

    # Train
    model = build_and_train_model('lda', X_tr_pca, y_tr, {})

    # Test
    df_test = df_all[df_all['Year'] == test_year].copy()
    y_test  = df_test['Win__1'].values
    df_test_n = apply_year_norm(df_test.copy(), norm_info)
    df_test_d = apply_delta_transform(df_test_n, numeric_bases)
    X_te      = pd.DataFrame(fold_pca.transform(df_test_d[model_feature_list]), columns=pc_cols)

    acc = model.score(X_te, y_test)
    exp = expected.get(test_year, None)
    match = '✓' if exp and abs(acc - exp) < 0.01 else 'DIFF'
    print(f'{test_year:>4}  {acc:>8.4f}  {exp:>9.4f}  {match:>7}')
