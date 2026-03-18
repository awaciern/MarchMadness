"""Diagnose exactly why ensemble_loyo.py gives wrong per-year results for LDA d2full pca10."""
import sys, pickle
sys.path.insert(0, 'Python')
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from predict_brackets import (
    load_combined_games, apply_year_norm,
    apply_delta_transform, mirror_augment, fit_global_scaler_delta,
    build_and_train_model, ALL_YEARS,
)

BASE = Path('.')
EXCLUDE = {2012, 2013, 2014}
TEST_YEAR = 2015

with open('Predictions/13g_lda_d2full_mixup2_pca10/model.pkl', 'rb') as f:
    pkl = pickle.load(f)

feature_list       = pkl['feature_list']
model_feature_list = pkl['model_feature_list']
numeric_bases      = pkl['numeric_bases']
pca_n              = pkl['pca_transformer'].n_components_

print(f"numeric_bases: {numeric_bases}")
print(f"feature_list: {feature_list[:4]}...")
print(f"model_feature_list: {model_feature_list[:4]}...")
print()

# --- method A: debug_ensemble.py approach ---
print("=== Method A (debug_ensemble.py approach) ===")
df_all = load_combined_games(BASE)
df_all = df_all.dropna(subset=[c for c in feature_list if c in df_all.columns])
print(f"df_all rows: {len(df_all)}, index: {df_all.index.min()}..{df_all.index.max()}")

norm_info = fit_global_scaler_delta(df_all, numeric_bases)
df_sim = pd.read_csv('Data/SimulatedDataMixup2/All.csv')

train_mask = ~df_all['Year'].isin({TEST_YEAR} | EXCLUDE)
df_train = df_all[train_mask].copy()
sim_mask = ~df_sim['Year'].isin({TEST_YEAR} | EXCLUDE)
df_sim_s = df_sim[sim_mask].copy()
shared_cols = [c for c in df_sim_s.columns if c in df_train.columns]
df_train = pd.concat([df_train, df_sim_s[shared_cols]], ignore_index=True)
df_train = apply_year_norm(df_train, norm_info)
df_train = apply_delta_transform(df_train, numeric_bases)
df_train = mirror_augment(df_train, model_feature_list)
X_tr = df_train[model_feature_list]
y_tr = df_train['Win__1'].reset_index(drop=True)
pc_cols = [f'PC{i}' for i in range(pca_n)]
fold_pca = PCA(n_components=pca_n, random_state=42)
X_tr_pca = pd.DataFrame(fold_pca.fit_transform(X_tr), columns=pc_cols)
model_A = build_and_train_model('lda', X_tr_pca, y_tr, {})

df_test_A = df_all[df_all['Year'] == TEST_YEAR].copy()
y_test_A = df_test_A['Win__1'].values
df_test_n = apply_year_norm(df_test_A.copy(), norm_info)
df_test_d = apply_delta_transform(df_test_n, numeric_bases)
X_te_A = pd.DataFrame(fold_pca.transform(df_test_d[model_feature_list]), columns=pc_cols)
acc_A = model_A.score(X_te_A, y_test_A)
preds_A = model_A.predict(X_te_A).astype(int)
print(f"Method A acc: {acc_A:.4f}  y_true sum: {int(y_test_A.sum())} n_correct: {int((preds_A == y_test_A).sum())}")
print(f"  preds_A[:10]: {preds_A.tolist()[:10]}")
print(f"  y_true[:10]:  {y_test_A.astype(int).tolist()[:10]}")
print()

# --- method B: ensemble_loyo.py approach ---
print("=== Method B (ensemble_loyo.py approach) ===")
df_raw = load_combined_games(BASE)
df_all1 = df_raw.dropna(subset=[c for c in feature_list if c in df_raw.columns]).copy()
print(f"df_all1 rows: {len(df_all1)}, index: {df_all1.index.min()}..{df_all1.index.max()}")

norm1_global = fit_global_scaler_delta(df_all1, numeric_bases)

# train B
train_mask2 = ~df_all1['Year'].isin({TEST_YEAR} | EXCLUDE)
df_train2 = df_all1[train_mask2].copy()
sim_mask2 = ~df_sim['Year'].isin({TEST_YEAR} | EXCLUDE)
df_sim_s2 = df_sim[sim_mask2].copy()
shared_cols2 = [c for c in df_sim_s2.columns if c in df_train2.columns]
df_train2 = pd.concat([df_train2, df_sim_s2[shared_cols2]], ignore_index=True)
df_train2 = apply_year_norm(df_train2, norm1_global)
df_train2 = apply_delta_transform(df_train2, numeric_bases)
df_train2 = mirror_augment(df_train2, model_feature_list)
X_tr2 = df_train2[model_feature_list]
y_tr2 = df_train2['Win__1']
pc_cols2 = [f'PC{i}' for i in range(pca_n)]
fold_pca2 = PCA(n_components=pca_n, random_state=42)
X_tr2_pca = pd.DataFrame(fold_pca2.fit_transform(X_tr2), columns=pc_cols2)
y_tr2 = y_tr2.reset_index(drop=True)
model_B = build_and_train_model('lda', X_tr2_pca, y_tr2, {})

# test B: ensemble_loyo.py style (reset_index + predict_proba_cfg)
df_test1 = df_all1[df_all1['Year'] == TEST_YEAR].copy().reset_index(drop=True)
y_true_B = df_test1['Win__1'].values
df_t2 = df_test1.copy()
df_t2 = apply_year_norm(df_t2, norm1_global)
df_t2 = apply_delta_transform(df_t2, numeric_bases)
X_te_B = df_t2[model_feature_list]
X_te_B = pd.DataFrame(fold_pca2.transform(X_te_B), columns=pc_cols2)

preds_B = model_B.predict(X_te_B).astype(int)
acc_B = (preds_B == y_true_B.astype(int)).mean()
print(f"Method B acc: {acc_B:.4f}  y_true sum: {int(y_true_B.sum())} n_correct: {int((preds_B == y_true_B).sum())}")
print(f"  preds_B[:10]: {preds_B.tolist()[:10]}")
print(f"  y_true[:10]:  {y_true_B.astype(int).tolist()[:10]}")
print()

# --- compare ---
print("=== Comparisons ===")
print(f"Training rows:  A={len(X_tr)}, B={len(X_tr2)}")
print(f"max diff in training features A vs B: {np.max(np.abs(X_tr.values - X_tr2.values)):.8f}")
print(f"Test rows: A={len(X_te_A)}, B={len(X_te_B)}")
print(f"max diff in test PCA features A vs B: {np.max(np.abs(X_te_A.values - X_te_B.values)):.8f}")

# Check if y_true arrays are the same
print(f"\ny_true match: {np.array_equal(y_test_A.astype(int), y_true_B.astype(int))}")
print(f"preds match:  {np.array_equal(preds_A, preds_B)}")

# Check norm scalers
for base in numeric_bases[:2]:
    sc_A = norm_info['fallback'].get(base, {})
    sc_B = norm1_global['fallback'].get(base, {})
    c1 = f'{base}__1'
    if isinstance(sc_A, dict) and isinstance(sc_B, dict):
        scA1 = sc_A.get(c1)
        scB1 = sc_B.get(c1)
        if scA1 and scB1:
            print(f"  Scaler '{c1}': A mean={scA1.mean_[0]:.6f}, B mean={scB1.mean_[0]:.6f}, same={abs(scA1.mean_[0]-scB1.mean_[0])<1e-9}")
