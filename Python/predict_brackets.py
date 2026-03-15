"""
predict_brackets.py

For each historical year, trains a leave-one-year-out model (all other years as
training data) to prevent data leakage, evaluates it on that year's game data,
and simulates filling out that year's tournament bracket.  After all years are
processed, also trains a traditional 67/33 random-split model on the full
historical dataset and reports its accuracy for reference.

The current year's bracket (--this-year) is predicted using a model trained on
all available historical data; no test accuracy is reported for it.

Final Four pairings (Round 5)
- Past years: derived automatically from the actual Round 5 CSV by comparing its
  teams against the predicted Elite Eight winners.
- Current year: controlled by --final-four-pairings (default "0-1,2-3"), which
  specifies how the 4 predicted regional winners (indexed 0-3 in Elite Eight order)
  are matched up.  E.g. "0-2,1-3" means winner[0] vs winner[2] and winner[1] vs
  winner[3].

Usage:
    python3 predict_brackets.py --model logistic_lbfgs
    python3 predict_brackets.py -m random_forest --final-four-pairings "0-2,1-3"
"""

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.optimize import minimize_scalar
from bracket_html import format_bracket_html
try:
    from xgboost import XGBClassifier as _XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    from lightgbm import LGBMClassifier as _LGBMClassifier
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False
try:
    from neural_net import TorchClassifier, TORCH_MODEL_KEYS, make_torch_classifier
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    TORCH_MODEL_KEYS = frozenset()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_YEARS = [y for y in range(2012, 2026) if y != 2020]  # all completed years through 2025

# Common features present in both sources; always sourced from KenPom (KP__ prefix).
# AdjO/AdjD/AdjT can be selected from either source using KP_AdjO or BT_AdjO etc.
COMMON_BASES: List[str] = [
    'WinPct', 'Wins', 'Losses',
    'Conf',
]

# KenPom bases (always KP__ prefix).
# KP_AdjO/KP_AdjD/KP_AdjT are the KenPom versions of the shared Adj fields.
KP_ONLY_BASES: List[str] = [
    'KP_AdjO', 'KP_Rk_AdjO', 'KP_AdjD', 'KP_Rk_AdjD', 'KP_AdjT', 'KP_Rk_AdjT',
    'AdjEM', 'Rk_AdjEM',
    'Luck', 'Rk_Luck',
    'SOS_AdjEM', 'Rk_SOS_AdjEM',
    'SOS_AdjO', 'Rk_SOS_AdjO',
    'SOS_AdjD', 'Rk_SOS_AdjD',
    'NCSOS_AdjEM', 'Rk_NCSOS_AdjEM',
]

# BartTorvik bases (always BT__ prefix).
# BT_AdjO/BT_AdjD/BT_AdjT are the BartTorvik versions of the shared Adj fields.
BT_ONLY_BASES: List[str] = [
    'BT_AdjO', 'BT_Rk_AdjO', 'BT_AdjD', 'BT_Rk_AdjD', 'BT_AdjT', 'BT_Rk_AdjT',
    'Barthag', 'Rk_Barthag',
    'EFG%', 'Rk_EFG%', 'EFGD%', 'Rk_EFGD%',
    'TOR', 'Rk_TOR', 'TORD', 'Rk_TORD',
    'ORB', 'Rk_ORB', 'DRB', 'Rk_DRB',
    'FTR', 'Rk_FTR', 'FTRD', 'Rk_FTRD',
    '2P%', 'Rk_2P%', '2P%D', 'Rk_2P%D',
    '3P%', 'Rk_3P%', '3P%D', 'Rk_3P%D',
    '3PR', 'Rk_3PR', '3PRD', 'Rk_3PRD',
    'WAB', 'Rk_WAB',
]

# 2-week BartTorvik snapshot bases (always BT2W__ prefix).
# Same column schema as BartTorvikData; base names are prefixed with '2W_'.
BT2W_BASES: List[str] = [
    '2W_WinPct', '2W_Wins', '2W_Losses',
    '2W_AdjO', '2W_Rk_AdjO', '2W_AdjD', '2W_Rk_AdjD', '2W_AdjT', '2W_Rk_AdjT',
    '2W_Barthag', '2W_Rk_Barthag',
    '2W_EFG%', '2W_Rk_EFG%', '2W_EFGD%', '2W_Rk_EFGD%',
    '2W_TOR', '2W_Rk_TOR', '2W_TORD', '2W_Rk_TORD',
    '2W_ORB', '2W_Rk_ORB', '2W_DRB', '2W_Rk_DRB',
    '2W_FTR', '2W_Rk_FTR', '2W_FTRD', '2W_Rk_FTRD',
    '2W_2P%', '2W_Rk_2P%', '2W_2P%D', '2W_Rk_2P%D',
    '2W_3P%', '2W_Rk_3P%', '2W_3P%D', '2W_Rk_3P%D',
    '2W_3PR', '2W_Rk_3PR', '2W_3PRD', '2W_Rk_3PRD',
    '2W_WAB', '2W_Rk_WAB',
]

# Hotness BartTorvik bases (always BTHOT__ prefix).
# Each value is the difference (2-week minus season) for the same BartTorvik column;
# base names are prefixed with 'HOT_'.
BTHOT_BASES: List[str] = [
    'HOT_WinPct', 'HOT_Wins', 'HOT_Losses',
    'HOT_AdjO', 'HOT_Rk_AdjO', 'HOT_AdjD', 'HOT_Rk_AdjD', 'HOT_AdjT', 'HOT_Rk_AdjT',
    'HOT_Barthag', 'HOT_Rk_Barthag',
    'HOT_EFG%', 'HOT_Rk_EFG%', 'HOT_EFGD%', 'HOT_Rk_EFGD%',
    'HOT_TOR', 'HOT_Rk_TOR', 'HOT_TORD', 'HOT_Rk_TORD',
    'HOT_ORB', 'HOT_Rk_ORB', 'HOT_DRB', 'HOT_Rk_DRB',
    'HOT_FTR', 'HOT_Rk_FTR', 'HOT_FTRD', 'HOT_Rk_FTRD',
    'HOT_2P%', 'HOT_Rk_2P%', 'HOT_2P%D', 'HOT_Rk_2P%D',
    'HOT_3P%', 'HOT_Rk_3P%', 'HOT_3P%D', 'HOT_Rk_3P%D',
    'HOT_3PR', 'HOT_Rk_3PR', 'HOT_3PRD', 'HOT_Rk_3PRD',
    'HOT_WAB', 'HOT_Rk_WAB',
]

# Full list of valid unprefixed base names for --features.
ALL_FEATURE_BASES: List[str] = COMMON_BASES + KP_ONLY_BASES + BT_ONLY_BASES + BT2W_BASES + BTHOT_BASES + ['Seed']

# Default feature selection.
DEFAULT_FEATURE_BASES: List[str] = ['WinPct', 'KP_AdjO', 'KP_AdjD', 'SOS_AdjEM']

# Base names that require label encoding (resolved before prefix is applied).
CATEGORICAL_BASE_NAMES: frozenset = frozenset(['Conf', 'Seed'])

MODEL_REGISTRY = {
    'logistic_regression':     LogisticRegression,
    'knn':                     KNeighborsClassifier,
    'svc':                     SVC,
    'decision_tree':           DecisionTreeClassifier,
    'random_forest':           RandomForestClassifier,
    'gradient_boosting':       GradientBoostingClassifier,
    'adaboost':                AdaBoostClassifier,
    'gpc':                     GaussianProcessClassifier,
    'mlp':                     MLPClassifier,
    'extra_trees':             ExtraTreesClassifier,
    'hist_gradient_boosting':  HistGradientBoostingClassifier,
    'lda':                     LinearDiscriminantAnalysis,
    **({'xgboost':  _XGBClassifier}  if _HAS_XGB   else {}),
    **({'lightgbm': _LGBMClassifier} if _HAS_LGB   else {}),
    **({'torch_mlp':         TorchClassifier,
        'torch_resnet':      TorchClassifier,
        'torch_transformer': TorchClassifier} if _HAS_TORCH else {}),
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def parse_model_params(params_list: list) -> dict:
    """
    Parse a list of 'key=value' strings into a typed dict.
    Attempts int → float → bool → None → str in that order.
    Single/double quotes around the value are stripped.
    """
    result = {}
    for item in (params_list or []):
        key, _, val_str = item.partition('=')
        key = key.strip()
        val_str = val_str.strip().strip("'\"")
        low = val_str.lower()
        if low == 'true':
            result[key] = True
        elif low == 'false':
            result[key] = False
        elif low == 'none':
            result[key] = None
        else:
            try:
                result[key] = int(val_str)
            except ValueError:
                try:
                    result[key] = float(val_str)
                except ValueError:
                    result[key] = val_str
    return result

def load_combined_games(data_root: Path, exclude_year: int = None) -> pd.DataFrame:
    """Load GameCombinedData/All.csv, optionally excluding one year."""
    df = pd.read_csv(data_root / 'Data' / 'GameCombinedData' / 'All.csv')
    if exclude_year is not None:
        df = df[df['Year'] != exclude_year]
    return df


def load_bracket_round(data_root: Path, year: int, rnd: int) -> pd.DataFrame:
    return pd.read_csv(
        data_root / 'Data' / 'BracketCombinedData' / str(year) / f'Round{rnd}_{year}.csv'
    )


def load_kenpom(data_root: Path, year: int) -> pd.DataFrame:
    return pd.read_csv(data_root / 'Data' / 'KenPomData' / f'{year}.csv')


def load_barttorvik(data_root: Path, year: int) -> pd.DataFrame:
    return pd.read_csv(data_root / 'Data' / 'BartTorvikData' / f'{year}.csv')


def load_barttorvik_2week(data_root: Path, year: int) -> pd.DataFrame:
    return pd.read_csv(data_root / 'Data' / '2WeekBartTorvikData' / f'{year}.csv')


def load_barttorvik_hotness(data_root: Path, year: int) -> pd.DataFrame:
    return pd.read_csv(data_root / 'Data' / 'HotnessBartTorvikData' / f'{year}.csv')


def resolve_feature_col(base: str) -> str:
    """
    Map an unprefixed feature base name to its prefixed column base.
    Common features (WinPct, Wins, Losses, Conf) always use KP__.
    KP_AdjO/KP_AdjD/KP_AdjT map to KP__AdjO etc.; BT_AdjO etc. map to BT__AdjO etc.
    KenPom-only features always use KP__; BartTorvik-only always use BT__.
    2-week BartTorvik bases (2W_* prefix) always use BT2W__.
    Hotness BartTorvik bases (HOT_* prefix) always use BTHOT__.
    'Seed' is bracket metadata and carries no source prefix.
    """
    if base == 'Seed':
        return 'Seed'
    if base.startswith('KP_Adj'):
        return f'KP__{base[3:]}'
    if base.startswith('BT_Adj'):
        return f'BT__{base[3:]}'
    if base.startswith('2W_'):
        return f'BT2W__{base[3:]}'
    if base.startswith('HOT_'):
        return f'BTHOT__{base[4:]}'
    if base in COMMON_BASES:
        return f'KP__{base}'
    if base in KP_ONLY_BASES:
        return f'KP__{base}'
    return f'BT__{base}'


def fit_label_encoders(df: pd.DataFrame, cat_cols: List[str]) -> dict:
    """Fit one LabelEncoder per categorical column.  Returns {col: LabelEncoder}."""
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def apply_label_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Return a copy of df with each categorical column replaced by integer codes.
    Values not seen during fit (including NaN / blank seeds) are encoded as -1."""
    df = df.copy()
    for col, le in encoders.items():
        if col not in df.columns:
            continue
        class_map = {v: i for i, v in enumerate(le.classes_)}
        df[col] = df[col].astype(str).map(class_map).fillna(-1).astype(int)
    return df


# ---------------------------------------------------------------------------
# Per-year normalisation helpers
# ---------------------------------------------------------------------------

def fit_year_scalers(df: pd.DataFrame, num_cols: List[str]) -> dict:
    """
    Fit a StandardScaler for each year on the supplied numeric columns.

    Returns a norm_info dict:
        {
            'by_year':  {year: fitted_StandardScaler, …},
            'fallback': StandardScaler fitted on all data combined,
            'cols':     list of column names actually present
        }
    The fallback is used for any year not seen during fitting (e.g. a future year).
    """
    avail = [c for c in num_cols if c in df.columns]
    if not avail:
        return {'by_year': {}, 'fallback': None, 'cols': []}
    by_year: dict = {}
    for year, grp in df.groupby('Year'):
        sc = StandardScaler()
        sc.fit(grp[avail])
        by_year[year] = sc
    fallback = StandardScaler()
    fallback.fit(df[avail])
    return {'by_year': by_year, 'fallback': fallback, 'cols': avail}


def fit_global_scaler(df: pd.DataFrame, num_cols: List[str]) -> dict:
    """
    Fit a single StandardScaler across all years (global normalisation).

    Returns a norm_info dict in the same format as fit_year_scalers, but
    with an empty by_year dict so the global fallback scaler is always used.
    This is the --norm-all counterpart to --norm-years.
    """
    avail = [c for c in num_cols if c in df.columns]
    if not avail:
        return {'by_year': {}, 'fallback': None, 'cols': []}
    sc = StandardScaler()
    sc.fit(df[avail])
    return {'by_year': {}, 'fallback': sc, 'cols': avail}


def fit_year_scalers_delta(df: pd.DataFrame, numeric_bases: List[str]) -> dict:
    """
    Fit per-year scalers for delta-feature mode.
    Each base gets ONE scaler fitted on the stacked __1 + __2 values so
    both columns share identical scaling before the delta is computed.
    norm_info format:  {'by_year': {year: {base: scaler}},
                        'fallback': {base: scaler},
                        'cols': numeric_bases,
                        'is_delta': True}
    """
    by_year: dict = {}
    all_stacked: dict = {base: [] for base in numeric_bases}

    for year, grp in df.groupby('Year'):
        scalers: dict = {}
        for base in numeric_bases:
            c1 = f'{base}__1'
            c2 = f'{base}__2'
            vals = pd.concat([
                grp[c1].dropna() if c1 in grp.columns else pd.Series(dtype=float),
                grp[c2].dropna() if c2 in grp.columns else pd.Series(dtype=float),
            ]).values.reshape(-1, 1)
            if len(vals) == 0:
                scalers[base] = None
                continue
            sc = StandardScaler().fit(vals)
            scalers[base] = sc
            all_stacked[base].append(vals)
        by_year[year] = scalers

    # Global fallback scaler (all years stacked)
    fallback: dict = {}
    for base in numeric_bases:
        parts = all_stacked[base]
        if not parts:
            fallback[base] = None
            continue
        vals = np.vstack(parts)
        fallback[base] = StandardScaler().fit(vals)

    return {'by_year': by_year, 'fallback': fallback,
            'cols': numeric_bases, 'is_delta': True}


def fit_global_scaler_delta(df: pd.DataFrame, numeric_bases: List[str]) -> dict:
    """
    Fit a single global scaler for delta-feature mode (--norm-all --delta-feats).
    One scaler per base, fitted on all years' stacked __1 + __2 values.
    """
    fallback: dict = {}
    for base in numeric_bases:
        c1 = f'{base}__1'
        c2 = f'{base}__2'
        vals = pd.concat([
            df[c1].dropna() if c1 in df.columns else pd.Series(dtype=float),
            df[c2].dropna() if c2 in df.columns else pd.Series(dtype=float),
        ]).values.reshape(-1, 1)
        if len(vals) == 0:
            fallback[base] = None
            continue
        fallback[base] = StandardScaler().fit(vals)
    return {'by_year': {}, 'fallback': fallback,
            'cols': numeric_bases, 'is_delta': True}


def apply_delta_transform(df: pd.DataFrame, numeric_bases: List[str]) -> pd.DataFrame:
    """
    For each base in numeric_bases, compute:
        base__delta = base__1 - base__2
    then drop base__1 and base__2.  Returns a copy.
    Call this *after* normalisation so that deltas are in Z-score space when
    normalisation is active.
    """
    df = df.copy()
    for b in numeric_bases:
        c1, c2 = f'{b}__1', f'{b}__2'
        if c1 in df.columns and c2 in df.columns:
            df[f'{b}__delta'] = df[c1] - df[c2]
            df.drop(columns=[c1, c2], inplace=True)
    return df


def mirror_augment(df: pd.DataFrame, model_feature_list: List[str]) -> pd.DataFrame:
    """
    Append a mirrored copy of every row to balance the training set.

    For delta-feature mode: a mirrored row negates all ``__delta`` columns
    (team2 perspective) and swaps any remaining ``base__1``/``base__2``
    categorical pair columns, then flips ``Win__1``.

    This guarantees exactly 50/50 Win__1 class balance → intercept ≈ 0 and
    forces the model to learn correctly-signed coefficients (positive delta =
    team1 has better stats = team1 more likely to win).

    Works on the *post-transformation* DataFrame (after label-encoding,
    normalisation, and delta transform).  Only columns in model_feature_list
    plus ``Win__1`` and ``Year`` are relevant; all other columns are copied
    as-is for the mirrored rows.
    """
    mirror = df.copy()

    # Negate all delta columns present in the feature list.
    delta_cols = [c for c in model_feature_list if c.endswith('__delta') and c in mirror.columns]
    for c in delta_cols:
        mirror[c] = -mirror[c]

    # Swap any remaining __1 / __2 categorical column pairs present in the feature list.
    feat_set = set(model_feature_list)
    swapped: set = set()
    for c in model_feature_list:
        if c.endswith('__1') and c not in swapped:
            base = c[:-3]
            c2 = f'{base}__2'
            if c2 in feat_set and c in mirror.columns and c2 in mirror.columns:
                mirror[c], mirror[c2] = mirror[c2].copy(), mirror[c].copy()
                swapped.add(c)
                swapped.add(c2)

    # Flip the target label.
    if 'Win__1' in mirror.columns:
        mirror['Win__1'] = ~mirror['Win__1']

    return pd.concat([df, mirror], ignore_index=True)


def apply_year_norm(df: pd.DataFrame, norm_info: dict) -> pd.DataFrame:
    """Apply per-year normalisation in-place (training/batch path)."""
    by_year  = norm_info['by_year']
    fallback = norm_info['fallback']
    is_delta = norm_info.get('is_delta', False)
    df = df.copy()

    if is_delta:
        numeric_bases = norm_info['cols']
        for year, grp_idx in df.groupby('Year').groups.items():
            scalers = by_year.get(year, fallback)
            for base in numeric_bases:
                sc = scalers.get(base) if isinstance(scalers, dict) else None
                if sc is None:
                    sc = fallback.get(base) if isinstance(fallback, dict) else None
                if sc is None:
                    continue
                c1, c2 = f'{base}__1', f'{base}__2'
                if c1 in df.columns:
                    df.loc[grp_idx, c1] = sc.transform(
                        df.loc[grp_idx, c1].values.reshape(-1, 1)).ravel()
                if c2 in df.columns:
                    df.loc[grp_idx, c2] = sc.transform(
                        df.loc[grp_idx, c2].values.reshape(-1, 1)).ravel()
    else:
        num_cols = norm_info['cols']
        for year, grp_idx in df.groupby('Year').groups.items():
            sc = by_year.get(year, fallback)
            if sc is None:
                continue
            present = [c for c in num_cols if c in df.columns]
            if not present:
                continue
            df.loc[grp_idx, present] = sc.transform(df.loc[grp_idx, present])
    return df


def apply_year_norm_single(df: pd.DataFrame, year: int, norm_info: dict) -> pd.DataFrame:
    """
    Apply normalisation to a single-year DataFrame (simulation / bracket path).
    Works for both standard and delta-feature norm_info.
    """
    by_year  = norm_info['by_year']
    fallback = norm_info['fallback']
    is_delta = norm_info.get('is_delta', False)
    df = df.copy()

    if is_delta:
        numeric_bases = norm_info['cols']
        scalers = by_year.get(year, fallback)
        for base in numeric_bases:
            sc = scalers.get(base) if isinstance(scalers, dict) else None
            if sc is None:
                sc = fallback.get(base) if isinstance(fallback, dict) else None
            if sc is None:
                continue
            c1, c2 = f'{base}__1', f'{base}__2'
            if c1 in df.columns:
                df[c1] = sc.transform(df[c1].values.reshape(-1, 1)).ravel()
            if c2 in df.columns:
                df[c2] = sc.transform(df[c2].values.reshape(-1, 1)).ravel()
    else:
        num_cols = norm_info['cols']
        sc = by_year.get(year)
        if sc is None:
            sc = fallback if not isinstance(fallback, dict) else fallback.get(year)
        if sc is None:
            return df
        present = [c for c in num_cols if c in df.columns]
        if present:
            df[present] = sc.transform(df[present])
    return df


# ---------------------------------------------------------------------------
# Matchup attachment helpers
# ---------------------------------------------------------------------------

def attach_kenpom(df_matchups: pd.DataFrame, df_kp: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns Team__1 and Team__2, merge in KenPom stats
    with the KP__ source prefix and __1 / __2 team suffix.
    Seed is dropped — seeds are tracked separately via team_seed_map.
    """
    kp = df_kp.drop(columns=['Seed'], errors='ignore')
    rename_map = {c: f'KP__{c}' for c in kp.columns if c != 'Team'}
    kp = kp.rename(columns=rename_map)
    kp1 = kp.add_suffix('__1')
    kp2 = kp.add_suffix('__2')
    df = df_matchups.merge(kp1, on='Team__1', how='inner')
    df = df.merge(kp2, on='Team__2', how='inner')
    return df.reset_index(drop=True)


def attach_barttorvik(df_matchups: pd.DataFrame, df_bt: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns Team__1 and Team__2, merge in BartTorvik stats
    with the BT__ source prefix and __1 / __2 team suffix.
    Seed is dropped — seeds are tracked separately via team_seed_map.
    """
    bt = df_bt.drop(columns=['Seed', 'ConfRec', 'ConfWins', 'ConfLosses', 'ConfWinPct'], errors='ignore')
    rename_map = {c: f'BT__{c}' for c in bt.columns if c != 'Team'}
    bt = bt.rename(columns=rename_map)
    bt1 = bt.add_suffix('__1')
    bt2 = bt.add_suffix('__2')
    df = df_matchups.merge(bt1, on='Team__1', how='inner')
    df = df.merge(bt2, on='Team__2', how='inner')
    return df.reset_index(drop=True)


def attach_barttorvik_2week(df_matchups: pd.DataFrame, df_bt2w: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns Team__1 and Team__2, merge in 2-week BartTorvik
    stats with the BT2W__ source prefix and __1 / __2 team suffix.
    """
    bt2w = df_bt2w.drop(columns=['Seed', 'ConfRec', 'ConfWins', 'ConfLosses', 'ConfWinPct'], errors='ignore')
    rename_map = {c: f'BT2W__{c}' for c in bt2w.columns if c != 'Team'}
    bt2w = bt2w.rename(columns=rename_map)
    bt2w1 = bt2w.add_suffix('__1')
    bt2w2 = bt2w.add_suffix('__2')
    df = df_matchups.merge(bt2w1, on='Team__1', how='left')
    df = df.merge(bt2w2, on='Team__2', how='left')
    return df.reset_index(drop=True)


def attach_barttorvik_hotness(df_matchups: pd.DataFrame, df_hot: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns Team__1 and Team__2, merge in Hotness BartTorvik
    stats (difference: 2-week minus season) with BTHOT__ prefix and __1/__2 suffix.
    """
    hot = df_hot.drop(columns=['Seed', 'ConfRec', 'ConfWins', 'ConfLosses', 'ConfWinPct'], errors='ignore')
    rename_map = {c: f'BTHOT__{c}' for c in hot.columns if c != 'Team'}
    hot = hot.rename(columns=rename_map)
    hot1 = hot.add_suffix('__1')
    hot2 = hot.add_suffix('__2')
    df = df_matchups.merge(hot1, on='Team__1', how='left')
    df = df.merge(hot2, on='Team__2', how='left')
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Final Four pairing helpers
# ---------------------------------------------------------------------------

def derive_ff_pairings_from_data(data_root: Path, year: int) -> List[Tuple[int, int]]:
    """
    For a past year, read the actual Round 5 CSV and compare its teams to the
    actual Round 4 winners to figure out which indices (0-3) are paired.

    Returns a list of two (i, j) tuples, e.g. [(0, 1), (2, 3)].
    """
    df4 = load_bracket_round(data_root, year, 4)
    df5 = load_bracket_round(data_root, year, 5)

    actual_r4_winners = [
        row['Team__1'] if row['Win__1'] else row['Team__2']
        for _, row in df4.iterrows()
    ]

    pairings = []
    for _, game in df5.iterrows():
        t1, t2 = game['Team__1'], game['Team__2']
        try:
            i = actual_r4_winners.index(t1)
            j = actual_r4_winners.index(t2)
        except ValueError as e:
            raise RuntimeError(
                f'{year} Round 5 team not found in Round 4 winners: {e}\n'
                f'Round 4 winners: {actual_r4_winners}'
            )
        pairings.append((i, j))

    return pairings


def parse_ff_pairings_arg(arg: str) -> List[Tuple[int, int]]:
    """
    Parse a string like "0-1,2-3" or "0-2,1-3" into [(0,1),(2,3)].
    """
    pairings = []
    for part in arg.split(','):
        a, b = part.strip().split('-')
        pairings.append((int(a), int(b)))
    if len(pairings) != 2:
        raise ValueError(f'Expected exactly 2 pairings, got: {arg}')
    return pairings


# ---------------------------------------------------------------------------
# Temperature scaling calibration
# ---------------------------------------------------------------------------

class TemperatureScaledModel:
    """
    Wraps a fitted sklearn-style model and applies temperature scaling to
    predicted probabilities.  predict() is identical to the base model so
    the projected winner is always preserved; only predict_proba() changes.

    Temperature scaling: raw logit  log(p/(1-p))  is divided by T before
    converting back to a probability via sigmoid.
      T > 1  compresses probabilities toward 0.5 (confidence reduction).
      T < 1  stretches probabilities away from 0.5 (confidence amplification).
      T = 1  leaves probabilities unchanged.

    Because the sign of the logit is preserved for any T > 0, the predicted
    winner (team with p > 0.5) can never change.  Near-50%% predictions have
    logits close to zero, so dividing by T barely moves them regardless of T.
    """

    def __init__(self, base_model, temperature: float):
        self.base_model  = base_model
        self.temperature = float(temperature)

    # Explicit pickle support — must NOT go through __getattr__ or sklearn's
    # BaseEstimator.__getstate__ (which lives on the wrapped model) would be
    # returned, serialising the inner model's dict instead of ours.
    def __getstate__(self):
        return {'base_model': self.base_model, 'temperature': self.temperature}

    def __setstate__(self, state):
        if 'base_model' not in state:
            raise RuntimeError(
                'This model pickle was saved with a buggy version of '
                'TemperatureScaledModel and cannot be loaded. '
                'Please re-run training (predict_brackets.py) to regenerate it.'
            )
        self.base_model  = state['base_model']
        self.temperature = state['temperature']

    # Delegate unknown attribute access to the base model so sklearn utilities work.
    def __getattr__(self, name):
        # During unpickling __init__ hasn't run yet, so accessing self.base_model
        # via normal attribute lookup would re-enter __getattr__ and recurse forever.
        # Use object.__getattribute__ to bypass __getattr__ entirely.
        try:
            base = object.__getattribute__(self, 'base_model')
        except AttributeError:
            raise AttributeError(name)
        return getattr(base, name)

    def predict(self, X):
        return self.base_model.predict(X)

    def score(self, X, y):
        return self.base_model.score(X, y)

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)
        raw = np.clip(raw, 1e-9, 1.0 - 1e-9)
        logit_1 = np.log(raw[:, 1]) - np.log(raw[:, 0])  # log-odds for class 1
        scaled_logit = logit_1 / self.temperature
        p1 = 1.0 / (1.0 + np.exp(-scaled_logit))
        return np.column_stack([1.0 - p1, p1])


def _collect_oof_logits(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Run stratified k-fold CV and return OOF log-odds (logit for class 1)."""
    params = dict(model_params or {})
    if model_key == 'svc' and 'probability' not in params:
        params['probability'] = True
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_arr = y_train.values.astype(int)
    oof_logits = np.zeros(len(X_train))
    for train_idx, val_idx in kf.split(X_train, y_arr):
        clf = MODEL_REGISTRY[model_key](**params)
        clf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        proba = clf.predict_proba(X_train.iloc[val_idx])
        proba = np.clip(proba, 1e-9, 1.0 - 1e-9)
        oof_logits[val_idx] = np.log(proba[:, 1]) - np.log(proba[:, 0])
    return oof_logits


def fit_temperature_stretch(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
    p_target: float = 0.97,
    percentile: float = 98.0,
    n_folds: int = 5,
    seed: int = 0,
) -> float:
    """
    Find a temperature T that rescales the model's probability range so that
    the most-confident OOF prediction maps to approximately ``p_target``.

    Algorithm
    ---------
    1. Collect OOF logits via k-fold CV (unbiased: each row is scored by a
       model that never saw it during training).
    2. Compute the observed 'peak' logit magnitude using a high percentile
       (e.g. 98th) of |logit| to be robust against lone outliers.
    3. Set  T = L_observed / L_target  where L_target = logit(p_target).

    Behaviour
    ---------
    * If the model is conservative (e.g. max prediction 70%% but target 97%%),
      T < 1 and probabilities are stretched outward from 50%%.
    * If the model is overconfident (e.g. max prediction 99.9%%),
      T > 1 and extreme probabilities are compressed back toward 50%%.
    * Near-50%% predictions have logits ≈ 0 and are barely affected in either case.
    * T is always bounded to [0.05, 20] to avoid numerical extremes.
    """
    oof_logits = _collect_oof_logits(model_key, X_train, y_train, model_params, n_folds, seed)
    L_observed = float(np.percentile(np.abs(oof_logits), percentile))
    if L_observed < 1e-6:
        return 1.0  # degenerate model; no calibration
    L_target = float(np.log(p_target / (1.0 - p_target)))
    T = L_observed / L_target
    return float(np.clip(T, 0.05, 20.0))


def fit_temperature_nll(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
    n_folds: int = 5,
    seed: int = 0,
) -> float:
    """
    Find the temperature T in (0.1, 20) that minimises out-of-fold NLL.

    Unlike the stretch mode, this fits T to actual game outcomes, so the
    direction (compress or expand) is determined by data.  Useful when the
    model's raw probabilities are already well-calibrated and you mainly want
    a principled adjustment rather than a hard target.
    """
    oof_logits = _collect_oof_logits(model_key, X_train, y_train, model_params, n_folds, seed)
    y_arr = y_train.values.astype(int)

    def neg_log_likelihood(T: float) -> float:
        if T <= 0:
            return 1e9
        p1 = 1.0 / (1.0 + np.exp(-oof_logits / T))
        p1 = np.clip(p1, 1e-9, 1.0 - 1e-9)
        return -float(np.mean(y_arr * np.log(p1) + (1 - y_arr) * np.log(1 - p1)))

    result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 20.0), method='bounded')
    return float(result.x)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def build_and_train_model(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict = None,
    calibrate: bool = False,
    calibrate_temperature: float = None,
    calibrate_mode: str = 'stretch',
    calibrate_target: float = 0.97,
):
    """
    Train the requested model.  When ``calibrate=True``, wraps the fitted
    model in a ``TemperatureScaledModel`` that preserves predicted winners
    while rescaling win-probability confidence.

    Calibration modes
    -----------------
    stretch (default)
        Scale logits so the most-confident OOF prediction maps to
        ``calibrate_target`` (default 0.97).  Works in both directions:
        expands conservative models, compresses overconfident ones.
    nll
        Minimise out-of-fold NLL; direction is determined by the data.

    If ``calibrate_temperature`` is given it overrides the auto-fit for
    either mode and is used directly.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Options: {list(MODEL_REGISTRY)}")
    params = dict(model_params or {})
    # SVC requires probability=True to support predict_proba; inject it unless
    # the caller explicitly set probability=False.
    if model_key == 'svc' and 'probability' not in params:
        params['probability'] = True
    # XGBoost: suppress label-encoder deprecation warning.
    if model_key == 'xgboost':
        params.setdefault('eval_metric', 'logloss')
        params.setdefault('verbosity', 0)
    # LightGBM: suppress noisy stdout.
    if model_key == 'lightgbm':
        params.setdefault('verbose', -1)
    # Torch classifiers: use the factory so the arch default maps correctly.
    if _HAS_TORCH and model_key in TORCH_MODEL_KEYS:
        estimator = make_torch_classifier(model_key, **params)
    else:
        estimator = MODEL_REGISTRY[model_key](**params)
    estimator.fit(X_train, y_train)
    if calibrate:
        if calibrate_temperature is not None:
            T = float(calibrate_temperature)
        elif calibrate_mode == 'nll':
            T = fit_temperature_nll(model_key, X_train, y_train, params)
        else:  # 'stretch' (default)
            T = fit_temperature_stretch(model_key, X_train, y_train, params,
                                        p_target=calibrate_target)
        return TemperatureScaledModel(estimator, T)
    return estimator


# ---------------------------------------------------------------------------
# Bracket simulation
# ---------------------------------------------------------------------------

def simulate_bracket(
    model,
    data_root: Path,
    year: int,
    this_year: int = None,
    ff_pairings: List[Tuple[int, int]] = None,
    feature_list: list = None,
    cat_encoders: dict = None,
    norm_info: dict = None,
    delta_feats: bool = False,
    numeric_bases: list = None,
    model_feature_list: list = None,
    pca_transformer=None,
) -> Tuple[list, list, list, list, list, int]:
    """
    Simulate filling out a bracket from Round 1 using the model.

    Returns:
        pred_teams_by_round   – list of 6 lists of team names
        pred_seeds_by_round   – list of 6 lists of seeds
        pred_probs_by_round   – list of 6 lists of win-prob floats (None if model lacks predict_proba)
        correct_by_round      – list of 6 lists of bools (empty list for current year)
        num_correct_by_round  – list of 6 ints  (zeros for current year)
        score                 – total ESPN-style bracket score (0 for current year)
    """
    is_current = (this_year is not None and year == this_year)
    if ff_pairings is None:
        ff_pairings = [(0, 1), (2, 3)]
    if feature_list is None:
        feature_list = [
            f'KP__{b}__{i}' for b in DEFAULT_FEATURE_BASES for i in (1, 2)
        ]
    if model_feature_list is None:
        model_feature_list = feature_list
    if numeric_bases is None:
        numeric_bases = []
    if cat_encoders is None:
        cat_encoders = {}

    # Only load/attach a data source if any selected feature requires it.
    needs_bt    = any(f.startswith('BT__')    for f in feature_list)
    needs_bt2w  = any(f.startswith('BT2W__')  for f in feature_list)
    needs_bthot = any(f.startswith('BTHOT__') for f in feature_list)

    pred_teams_by_round: list = []
    pred_seeds_by_round: list = []
    pred_probs_by_round:  list = []
    correct_by_round:     list = []
    num_correct_by_round: list = []
    total_score = 0

    # The predicted winners from the previous round (used to build rnd 2+ matchups).
    prev_pred_teams: List[str] = []
    # Seed lookup built from Round 1 so later rounds can fill in seeds when KenPom
    # has blank seeds (e.g. current year loaded with --no-seeds).
    team_seed_map: dict = {}

    for rnd in range(1, 7):
        if rnd == 1 or not is_current:
            # For past years we always load the actual bracket CSV (which contains
            # the real Winning_Team column) to evaluate accuracy.
            df_round = load_bracket_round(data_root, year, rnd)
        # For rounds 2-6 of the current year, df_round is built from predictions.

        if rnd == 1:
            # Populate seed map from Round 1 for use in later rounds.
            for _, _row in df_round.iterrows():
                team_seed_map[_row['Team__1']] = _row['Seed__1']
                team_seed_map[_row['Team__2']] = _row['Seed__2']

        if rnd > 1:
            # Capture actual winning teams before rebuilding df_round (past years only).
            actual_winners = (
                df_round['Winning_Team'].reset_index(drop=True)
                if not is_current else None
            )

            # Determine matchup order for this round.
            if rnd == 5:
                ordered = [None] * 4
                for (i, j) in ff_pairings:
                    # Place team at position i as Team__1 and j as Team__2.
                    ordered[i] = prev_pred_teams[i]
                    ordered[j] = prev_pred_teams[j]
                matchup_teams = [
                    (prev_pred_teams[ff_pairings[0][0]], prev_pred_teams[ff_pairings[0][1]]),
                    (prev_pred_teams[ff_pairings[1][0]], prev_pred_teams[ff_pairings[1][1]]),
                ]
            else:
                matchup_teams = [
                    (prev_pred_teams[i], prev_pred_teams[i + 1])
                    for i in range(0, len(prev_pred_teams), 2)
                ]

            df_matchups = pd.DataFrame(matchup_teams, columns=['Team__1', 'Team__2'])
            df_kp = load_kenpom(data_root, year)
            df_round = attach_kenpom(df_matchups, df_kp)
            if needs_bt:
                df_bt = load_barttorvik(data_root, year)
                df_round = attach_barttorvik(df_round, df_bt)
            if needs_bt2w:
                df_bt2w = load_barttorvik_2week(data_root, year)
                df_round = attach_barttorvik_2week(df_round, df_bt2w)
            if needs_bthot:
                df_hot = load_barttorvik_hotness(data_root, year)
                df_round = attach_barttorvik_hotness(df_round, df_hot)

            # Seed columns are not present in dynamically built rounds (Seed is
            # dropped during attach); populate from the Round 1 seed map.
            df_round['Seed__1'] = df_round['Team__1'].map(team_seed_map)
            df_round['Seed__2'] = df_round['Team__2'].map(team_seed_map)

            if not is_current:
                df_round['Winning_Team'] = actual_winners

        # Predict.
        # Save raw seed values before encoding so display isn't affected.
        raw_seeds_1 = df_round['Seed__1'].copy() if 'Seed__1' in df_round.columns else None
        raw_seeds_2 = df_round['Seed__2'].copy() if 'Seed__2' in df_round.columns else None
        if cat_encoders:
            df_round = apply_label_encoders(df_round, cat_encoders)
        if norm_info is not None:
            df_round = apply_year_norm_single(df_round, year, norm_info)
        if delta_feats and numeric_bases:
            df_round = apply_delta_transform(df_round, numeric_bases)
        X = df_round[model_feature_list]
        if pca_transformer is not None:
            X = pca_transformer.transform(X)
        preds = model.predict(X).astype(bool)
        # Win probability for the predicted winner (None if model lacks predict_proba).
        try:
            proba = model.predict_proba(X)
            win_probs = [
                proba[k, 1] if preds[k] else proba[k, 0]
                for k in range(len(preds))
            ]
        except AttributeError:
            win_probs = [None] * len(preds)
        df_round = df_round.copy()
        df_round['Pred_Win__1'] = preds

        pred_teams = df_round['Team__1'].where(preds, df_round['Team__2'])
        # Use raw (pre-encoding) seeds for display.
        s1 = raw_seeds_1 if raw_seeds_1 is not None else df_round['Seed__1']
        s2 = raw_seeds_2 if raw_seeds_2 is not None else df_round['Seed__2']
        pred_seeds = s1.where(preds, s2)

        pred_teams_by_round.append(pred_teams.tolist())
        pred_seeds_by_round.append(pred_seeds.tolist())
        pred_probs_by_round.append(win_probs)
        prev_pred_teams = pred_teams.tolist()

        def _prob_str(p):
            return f'{p:.0%}' if p is not None else ''

        if not is_current:
            correct = (pred_teams == df_round['Winning_Team']).tolist()
            n_correct = sum(correct)
            round_score = n_correct * (2 ** (rnd - 1)) * 10
            correct_by_round.append(correct)
            num_correct_by_round.append(n_correct)
            total_score += round_score
            picks_str = '  '.join(
                f'[{pred_seeds.iloc[k]}]{pred_teams.iloc[k]} {_prob_str(win_probs[k])} {"✓" if correct[k] else "✗"}'
                for k in range(len(pred_teams))
            )
            print(f'  Round {rnd} ({n_correct} correct, {round_score} pts): {picks_str}')
        else:
            correct_by_round.append([])
            num_correct_by_round.append(0)
            picks_str = '  '.join(
                f'[{pred_seeds.iloc[k]}]{pred_teams.iloc[k]} {_prob_str(win_probs[k])}'
                for k in range(len(pred_teams))
            )
            print(f'  Round {rnd}: {picks_str}')

    return pred_teams_by_round, pred_seeds_by_round, pred_probs_by_round, correct_by_round, num_correct_by_round, total_score


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Predict NCAA brackets using a single trained model evaluated across all years.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--model', '-m',
        default='logistic_regression',
        choices=list(MODEL_REGISTRY),
        help='Model algorithm to use for predictions.',
    )
    parser.add_argument(
        '--data-root', '-d',
        default=str(Path(__file__).resolve().parents[1]),
        help='Path to repo root (contains Data/ directory).',
    )
    parser.add_argument(
        '--output-root', '-o',
        default=str(Path(__file__).resolve().parents[1]),
        help='Path under which Predictions/<model>/ outputs are written.',
    )
    parser.add_argument(
        '--this-year',
        type=int,
        default=None,
        help=(
            'Treat this year as the "current" year: its bracket is predicted but not scored. '
            'If omitted, all years are treated as historical and fully scored.'
        ),
    )
    parser.add_argument(
        '--final-four-pairings',
        default='0-1,2-3',
        help=(
            'How the 4 predicted Elite Eight winners (indexed 0-3 in CSV order) '
            'are paired for the Final Four.  Used ONLY for the current year; '
            'past years derive pairings from actual Round 5 data.  '
            'Format: "i-j,k-l", e.g. "0-2,1-3".  Default: "0-1,2-3".'
        ),
    )
    parser.add_argument(
        '--model-params',
        nargs='*',
        default=[],
        metavar='KEY=VALUE',
        help=(
            'Parameters to pass to the model constructor as key=value pairs. '
            'Values are auto-cast to int, float, bool, None, or str. '
            'Example: --model-params random_state=0 solver=lbfgs max_iter=1000'
        ),
    )
    parser.add_argument(
        '--features',
        nargs='+',
        default=DEFAULT_FEATURE_BASES,
        choices=ALL_FEATURE_BASES,
        metavar='FEATURE',
        help=(
            'Space-separated list of unprefixed base feature names. '
            'Common features (always KP__ prefix): '
            f'{COMMON_BASES}. '
            'KenPom-only (always KP__ prefix): '
            f'{KP_ONLY_BASES}. '
            'BartTorvik-only (always BT__ prefix): '
            f'{str(BT_ONLY_BASES).replace("%", "%%")}. '
            '2-week BartTorvik snapshot (always BT2W__ prefix, base names start with 2W_): '
            f'{str(BT2W_BASES).replace("%", "%%")}. '
            'Hotness BartTorvik diff (always BTHOT__ prefix, base names start with HOT_): '
            f'{str(BTHOT_BASES).replace("%", "%%")}. '
            'Categorical opt-ins: Conf, Seed. '
            f'Default: {DEFAULT_FEATURE_BASES}.'
        ),
    )
    parser.add_argument(
        '--norm-years',
        action='store_true',
        default=False,
        help=(
            'Normalise numeric features within each year independently '
            '(Z-score per year) before training and evaluation. '
            'Prevents cross-year scale drift from influencing the model. '
            'The output folder name will include a NY indicator (e.g. KPNY instead of KP).'
        ),
    )
    parser.add_argument(
        '--norm-all',
        action='store_true',
        default=False,
        help=(
            'Normalise numeric features across all years using a single global '
            'StandardScaler. Mutually exclusive with --norm-years. '
            'Output folder name will include a NA indicator (e.g. KPNA instead of KP).'
        ),
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        default=False,
        help=(
            'Apply temperature scaling to predicted win probabilities. '
            'Predicted winners are never changed — only the confidence adjusts. '
            'Near-50%% predictions are barely affected regardless of the mode. '
            'See --calibrate-mode and --calibrate-target for tuning. '
            'Output folder name will include a CAL indicator.'
        ),
    )
    parser.add_argument(
        '--calibrate-mode',
        default='stretch',
        choices=['stretch', 'nll'],
        help=(
            'How to fit the calibration temperature T when --calibrate is active. '
            '"stretch" (default): scale OOF logits so the most-confident prediction '
            'maps to --calibrate-target. Works in both directions (expands conservative '
            'models, compresses overconfident ones). '
            '"nll": minimise out-of-fold NLL; direction determined by data. '
            'Has no effect without --calibrate.'
        ),
    )
    parser.add_argument(
        '--calibrate-target',
        type=float,
        default=0.97,
        metavar='P',
        help=(
            'Target maximum win probability for --calibrate-mode stretch. '
            'The most confident OOF prediction is scaled to this value; all '
            'others are scaled proportionally in logit space. '
            'Must be in (0.5, 1.0). Default: 0.97. '
            'Has no effect without --calibrate or when --calibrate-mode nll is set.'
        ),
    )
    parser.add_argument(
        '--calibrate-temperature',
        type=float,
        default=None,
        metavar='T',
        help=(
            'Override the auto-fitted temperature with a fixed value (any T > 0). '
            'T > 1 compresses probabilities toward 50%%; T < 1 stretches them away. '
            'T = 1 leaves probabilities unchanged. '
            'Has no effect without --calibrate.'
        ),
    )
    parser.add_argument(
        '--delta-feats',
        action='store_true',
        default=False,
        help=(
            'Combine numeric __1 and __2 features into a single delta feature '
            '(team1 value minus team2 value) before training and prediction. '
            'Categorical features (Conf, Seed) are kept as separate __1/__2 columns. '
            'When normalisation is active, both __1 and __2 are scaled using a scaler '
            'fitted on the combined distribution of both columns, then the delta is '
            'computed in the normalised space. '
            'Output folder name will include a DF indicator.'
        ),
    )
    parser.add_argument(
        '--exclude-years',
        nargs='*',
        type=int,
        default=[],
        metavar='YEAR',
        help=(
            'Years to exclude from both training data and evaluation. '
            'These years are removed from the leave-one-year-out loop and from all '
            'training datasets.  Example: --exclude-years 2012 2013'
        ),
    )
    parser.add_argument(
        '--sim-data',
        default=None,
        metavar='IDENTIFIER',
        help=(
            'Identifier of a simulated data source to augment training with. '
            'Must match a directory Data/SimulatedData<IDENTIFIER>/ containing All.csv. '
            'Simulated data is only used for model training; testing always uses real data.'
        ),
    )
    parser.add_argument(
        '--pca-components',
        type=int,
        default=None,
        metavar='N',
        help=(
            'Reduce features to N principal components (PCA) before training. '
            'PCA is fit on each fold\'s training data after all other transforms '
            '(encoding, normalisation, delta). The same fitted transformer is used '
            'at inference time. Stored in the model pickle for use by simulate_bracket.py.'
        ),
    )
    parser.add_argument(
        '--run-name',
        required=True,
        help=(
            'Human-readable name for this model run.  Used as the folder name under '
            'Predictions/ and stored in model_info.json for display in the UI.  '
            'Example: "RF_AdjEM_NY".'
        ),
    )
    args = parser.parse_args()

    # Canonically sort feature bases so the model is identical regardless of
    # the order features are passed on the CLI / submitted from the UI.
    # (For Random Forest, feature *index* affects which columns are selected at
    # each split, so different input orderings with the same random_state would
    # otherwise produce different trees and different predictions.)
    args.features = sorted(set(args.features), key=str.lower)

    pca_n_components = args.pca_components  # int or None

    data_root = Path(args.data_root)
    model_params = parse_model_params(args.model_params)
    if model_params:
        print(f'Model params: {model_params}')
    # Resolve each base name to its source-prefixed column, then expand __1/__2.
    feature_list = [
        f'{resolve_feature_col(b)}__{i}'
        for b in args.features for i in (1, 2)
    ]
    # Identify categorical columns for label encoding.
    cat_col_set = {
        f'{resolve_feature_col(b)}__{i}'
        for b in args.features if b in CATEGORICAL_BASE_NAMES
        for i in (1, 2)
    }
    cat_cols = [c for c in feature_list if c in cat_col_set]
    # --delta-feats: compute per-base numeric list and the model's feature list.
    delta_feats = args.delta_feats
    if delta_feats:
        # Ordered numeric base col names (source-prefixed, no __1/__2 suffix).
        _seen_num: dict = {}
        numeric_bases: List[str] = []
        for b in args.features:
            if b not in CATEGORICAL_BASE_NAMES:
                bc = resolve_feature_col(b)
                if bc not in _seen_num:
                    _seen_num[bc] = True
                    numeric_bases.append(bc)
        # Model receives one __delta col per numeric base, plus __1/__2 for cats.
        _seen_mf: dict = {}
        model_feature_list: List[str] = []
        for b in args.features:
            bc = resolve_feature_col(b)
            if b in CATEGORICAL_BASE_NAMES:
                for i in (1, 2):
                    fc = f'{bc}__{i}'
                    if fc not in _seen_mf:
                        _seen_mf[fc] = True
                        model_feature_list.append(fc)
            else:
                fc = f'{bc}__delta'
                if fc not in _seen_mf:
                    _seen_mf[fc] = True
                    model_feature_list.append(fc)
    else:
        numeric_bases = []
        model_feature_list = feature_list
    # Write to a pending folder; renamed to the user-supplied run name at the end.
    output_root  = Path(args.output_root) / 'Predictions' / f'__{args.run_name}__pending'
    output_root.mkdir(parents=True, exist_ok=True)

    this_year = args.this_year
    current_year_ff_pairings = parse_ff_pairings_arg(args.final_four_pairings)

    # Build the list of years to process: all completed years, plus the current
    # year appended at the end if it was supplied and isn't already in ALL_YEARS.
    exclude_years = set(args.exclude_years or [])
    years_to_process = [y for y in ALL_YEARS if y not in exclude_years]
    if this_year is not None and this_year not in years_to_process:
        years_to_process.append(this_year)
    num_eval_years = len(years_to_process) - (1 if this_year is not None else 0)
    if exclude_years:
        print(f'Excluding years from training data: {sorted(exclude_years)}')

    # -----------------------------------------------------------------------
    # Load the full game dataset once (used to build per-year training sets).
    # -----------------------------------------------------------------------
    df_all_raw = load_combined_games(data_root)
    pre_drop = len(df_all_raw)
    df_all_raw = df_all_raw.dropna(subset=feature_list)
    dropped = pre_drop - len(df_all_raw)
    if dropped:
        print(f'Note: dropped {dropped}/{pre_drop} rows with NaN in selected features.')

    # Load simulated augmentation data if requested (training only; never used for testing).
    df_sim_raw = None
    sim_data_id = args.sim_data
    if sim_data_id:
        sim_path = data_root / 'Data' / f'SimulatedData{sim_data_id}' / 'All.csv'
        if not sim_path.exists():
            parser.error(f'--sim-data: dataset not found at {sim_path}')
        df_sim_raw = pd.read_csv(sim_path)
        pre_sim = len(df_sim_raw)
        df_sim_raw = df_sim_raw.dropna(subset=feature_list)
        dropped_sim = pre_sim - len(df_sim_raw)
        print(f'Simulated data:  SimulatedData{sim_data_id} '
              f'({len(df_sim_raw)} rows'
              + (f', {dropped_sim} dropped for NaN' if dropped_sim else '') + ')')

    # Fit label encoders on the full dataset so all values are known.
    cat_encoders = fit_label_encoders(df_all_raw, cat_cols) if cat_cols else {}

    # Normalisation (optional) — fit scalers on numeric columns only.
    norm_years = args.norm_years
    norm_all   = args.norm_all
    calibrate             = args.calibrate
    calibrate_mode        = args.calibrate_mode
    calibrate_target      = args.calibrate_target
    calibrate_temperature = args.calibrate_temperature
    if norm_years and norm_all:
        parser.error('--norm-years and --norm-all are mutually exclusive.')
    norm_info: dict = None
    if norm_years:
        df_for_norm = apply_label_encoders(df_all_raw, cat_encoders) if cat_encoders else df_all_raw
        if delta_feats:
            norm_info = fit_year_scalers_delta(df_for_norm, numeric_bases)
        else:
            num_cols = [c for c in feature_list if c not in cat_col_set]
            norm_info = fit_year_scalers(df_for_norm, num_cols)
        print(f'Normalisation:           per-year Z-score  ({len(norm_info["cols"])} numeric columns)')
    elif norm_all:
        df_for_norm = apply_label_encoders(df_all_raw, cat_encoders) if cat_encoders else df_all_raw
        if delta_feats:
            norm_info = fit_global_scaler_delta(df_for_norm, numeric_bases)
        else:
            num_cols = [c for c in feature_list if c not in cat_col_set]
            norm_info = fit_global_scaler(df_for_norm, num_cols)
        print(f'Normalisation:           global Z-score    ({len(norm_info["cols"])} numeric columns)')
    else:
        print(f'Normalisation:           OFF')
    if calibrate:
        if calibrate_temperature is not None:
            _cal_desc = f'ON (T={calibrate_temperature:.3f}, fixed)'
        elif calibrate_mode == 'stretch':
            _cal_desc = f'ON (mode=stretch, target={calibrate_target:.0%}, T=auto-fit)'
        else:
            _cal_desc = 'ON (mode=nll, T=auto-fit via OOF NLL)'
    else:
        _cal_desc = 'OFF'
    print(f'Probability calibration: {_cal_desc}')

    print(f'Model type: {args.model}')

    # -----------------------------------------------------------------------
    # Per-year loop — leave-one-year-out training to prevent data leakage.
    # -----------------------------------------------------------------------
    total_correct_by_round = [0] * 7   # index 0 unused; rounds 1-6 at [1]-[6]
    total_score = 0
    year_model_stats: List[dict] = []   # {year, train_acc, test_acc, score}

    for year in years_to_process:
        print(f'\n{"="*50}\n{year}\n{"="*50}')
        is_current = (this_year is not None and year == this_year)

        # --- Train a model for this year -----------------------------------
        if is_current:
            # Current year: train on all historical data (no test set), minus excluded years.
            df_train = df_all_raw[~df_all_raw['Year'].isin(exclude_years)].copy() if exclude_years else df_all_raw.copy()
        else:
            # Historical year: train on every other year to avoid leakage, minus excluded years.
            df_train = df_all_raw[~df_all_raw['Year'].isin({year} | exclude_years)].copy()

        df_test_year = df_all_raw[df_all_raw['Year'] == year].copy() if not is_current else None

        # Augment training set with simulated data (year-exclusion mirrored; no sim in test).
        if df_sim_raw is not None:
            if is_current:
                df_sim_slice = (
                    df_sim_raw[~df_sim_raw['Year'].isin(exclude_years)].copy()
                    if exclude_years else df_sim_raw.copy()
                )
            else:
                df_sim_slice = df_sim_raw[
                    ~df_sim_raw['Year'].isin({year} | exclude_years)
                ].copy()
            df_train = pd.concat([df_train, df_sim_slice], ignore_index=True)

        if cat_encoders:
            df_train = apply_label_encoders(df_train, cat_encoders)
            if df_test_year is not None:
                df_test_year = apply_label_encoders(df_test_year, cat_encoders)

        if norm_info is not None:
            df_train = apply_year_norm(df_train, norm_info)
            if df_test_year is not None:
                df_test_year = apply_year_norm(df_test_year, norm_info)

        if delta_feats and numeric_bases:
            df_train = apply_delta_transform(df_train, numeric_bases)
            if df_test_year is not None:
                df_test_year = apply_delta_transform(df_test_year, numeric_bases)
            # Mirror-augment training set: append a flipped copy of every row so
            # the class distribution is exactly 50/50 and coefficients get the
            # correct signs (positive delta → team1's stat is better → team1 wins).
            df_train = mirror_augment(df_train, model_feature_list)

        X_tr = df_train[model_feature_list]
        y_tr = df_train['Win__1']
        fold_pca = None
        if pca_n_components:
            _pc_cols = [f'PC{i}' for i in range(pca_n_components)]
            fold_pca = PCA(n_components=pca_n_components, random_state=42)
            X_tr = pd.DataFrame(fold_pca.fit_transform(X_tr), columns=_pc_cols)
            y_tr = y_tr.reset_index(drop=True)
        model = build_and_train_model(
            args.model, X_tr, y_tr, model_params,
            calibrate=calibrate, calibrate_temperature=calibrate_temperature,
            calibrate_mode=calibrate_mode, calibrate_target=calibrate_target,
        )
        if calibrate and hasattr(model, 'temperature'):
            print(f'  Calibration temperature T={model.temperature:.4f}')

        train_acc = model.score(X_tr, y_tr)
        if not is_current:
            X_te = (
                pd.DataFrame(fold_pca.transform(df_test_year[model_feature_list]),
                             columns=_pc_cols)
                if fold_pca is not None
                else df_test_year[model_feature_list]
            )
            y_te = df_test_year['Win__1']
            test_acc = model.score(X_te, y_te)
            print(f'  Model trained on {len(df_train)} rows (excl. {year})')
            print(f'  Train acc: {train_acc:.4f}  |  Test acc on {year}: {test_acc:.4f}')
        else:
            test_acc = None
            print(f'  Model trained on all {len(df_train)} historical rows')
            print(f'  Train acc: {train_acc:.4f}  (no test set for current year)')

        # Determine Final Four pairings.
        if is_current:
            ff_pairings = current_year_ff_pairings
        else:
            try:
                ff_pairings = derive_ff_pairings_from_data(data_root, year)
            except Exception as e:
                print(f'  WARNING: could not derive FF pairings ({e}), using default 0-1,2-3')
                ff_pairings = [(0, 1), (2, 3)]

        print(f'  FF pairings (R4 indices): {ff_pairings}')

        # Simulate bracket.
        pred_teams, pred_seeds, pred_probs, correct, n_correct, score = simulate_bracket(
            model=model,
            data_root=data_root,
            year=year,
            this_year=this_year,
            ff_pairings=ff_pairings,
            feature_list=feature_list,
            cat_encoders=cat_encoders,
            norm_info=norm_info,
            delta_feats=delta_feats,
            numeric_bases=numeric_bases,
            model_feature_list=model_feature_list,
            pca_transformer=fold_pca,
        )

        if not is_current:
            for rnd in range(1, 7):
                total_correct_by_round[rnd] += n_correct[rnd - 1]
            total_score += score
            print(f'  Year total: {sum(n_correct)} for 63, {score} pts')

        year_model_stats.append({
            'year': year,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'bracket_score': score if not is_current else None,
        })

        # Write prediction file (HTML bracket).
        html_str = format_bracket_html(
            data_root=data_root,
            year=year,
            pred_teams_by_round=pred_teams,
            pred_seeds_by_round=pred_seeds,
            pred_probs_by_round=pred_probs,
            correct_by_round=correct,
            num_correct_by_round=n_correct,
            total_score=score,
            is_current=is_current,
            model_key=args.model,
            feat_bases=args.features,
            ff_pairings=ff_pairings,
        )
        out_path = output_root / f'{year}.html'
        out_path.write_text(html_str, encoding='utf-8')

    # -----------------------------------------------------------------------
    # Reference model: behaviour depends on whether simulated data was used.
    #
    # No sim data  →  traditional 67/33 random train/test split on real data.
    # Sim data     →  train on full sim+real dataset; test on real data only.
    #                 This makes both numbers meaningful: train acc reflects
    #                 what the model actually learned on the augmented set, and
    #                 test acc is the accuracy on unperturbed real games (the
    #                 ground truth), avoiding the misleading 100% that an
    #                 unconstrained tree scores when evaluated on its own
    #                 (identical-feature) training copies.
    # -----------------------------------------------------------------------
    if df_sim_raw is not None:
        print(f'\n{"="*50}\nSIM DATA REFERENCE MODEL (train=sim+real, test=real only)\n{"="*50}')

        # Real data — full set, preprocessed the same way as during training.
        df_real_ref = df_all_raw[
            df_all_raw['Year'].isin(ALL_YEARS) & ~df_all_raw['Year'].isin(exclude_years)
        ].copy()
        if cat_encoders:
            df_real_ref = apply_label_encoders(df_real_ref, cat_encoders)
        if norm_info is not None:
            df_real_ref = apply_year_norm(df_real_ref, norm_info)
        if delta_feats and numeric_bases:
            df_real_ref = apply_delta_transform(df_real_ref, numeric_bases)

        # Sim data — apply the same preprocessing chain.
        df_sim_ref = df_sim_raw[
            ~df_sim_raw['Year'].isin(exclude_years)
        ].copy() if exclude_years else df_sim_raw.copy()
        if cat_encoders:
            df_sim_ref = apply_label_encoders(df_sim_ref, cat_encoders)
        if norm_info is not None:
            df_sim_ref = apply_year_norm(df_sim_ref, norm_info)
        if delta_feats and numeric_bases:
            df_sim_ref = apply_delta_transform(df_sim_ref, numeric_bases)

        # Training set = sim + real (all rows).
        df_train_ref = pd.concat([df_real_ref, df_sim_ref], ignore_index=True)
        if delta_feats and numeric_bases:
            df_train_ref = mirror_augment(df_train_ref, model_feature_list)

        X_tr_t = df_train_ref[model_feature_list]
        y_tr_t = df_train_ref['Win__1']
        # Test set = real data only.
        X_te_t = df_real_ref[model_feature_list]
        y_te_t = df_real_ref['Win__1']
        if pca_n_components:
            _pc_cols_t = [f'PC{i}' for i in range(pca_n_components)]
            _ref_pca = PCA(n_components=pca_n_components, random_state=42)
            X_tr_t = pd.DataFrame(_ref_pca.fit_transform(X_tr_t), columns=_pc_cols_t)
            y_tr_t = y_tr_t.reset_index(drop=True)
            X_te_t = pd.DataFrame(_ref_pca.transform(X_te_t), columns=_pc_cols_t)
            y_te_t = y_te_t.reset_index(drop=True)
        model_trad = build_and_train_model(
            args.model, X_tr_t, y_tr_t, model_params,
            calibrate=calibrate, calibrate_temperature=calibrate_temperature,
            calibrate_mode=calibrate_mode, calibrate_target=calibrate_target,
        )
        trad_train_acc = model_trad.score(X_tr_t, y_tr_t)
        trad_test_acc  = model_trad.score(X_te_t, y_te_t)
        trad_split_label = f'Sim+real train acc: {trad_train_acc:.4f}  |  Real-only test acc: {trad_test_acc:.4f}'
        print(f'  {trad_split_label}')
        print(f'  (Training rows: {len(X_tr_t)} sim+real — Test rows: {len(X_te_t)} real only)')
    else:
        print(f'\n{"="*50}\nTRADITIONAL 67/33 TRAIN-TEST SPLIT MODEL\n{"="*50}')
        df_trad = df_all_raw[df_all_raw['Year'].isin(ALL_YEARS) & ~df_all_raw['Year'].isin(exclude_years)].copy()
        if cat_encoders:
            df_trad = apply_label_encoders(df_trad, cat_encoders)
        if norm_info is not None:
            df_trad = apply_year_norm(df_trad, norm_info)
        if delta_feats and numeric_bases:
            df_trad = apply_delta_transform(df_trad, numeric_bases)
        X_trad, y_trad = df_trad[model_feature_list], df_trad['Win__1']
        X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(X_trad, y_trad, test_size=0.33, random_state=42)
        if delta_feats and numeric_bases:
            # Mirror-augment only the training split (avoid leaking mirrored test rows into train).
            df_tr_t = pd.concat([X_tr_t, y_tr_t], axis=1)
            df_tr_t = mirror_augment(df_tr_t, model_feature_list)
            X_tr_t, y_tr_t = df_tr_t[model_feature_list], df_tr_t['Win__1']
        if pca_n_components:
            _pc_cols_t = [f'PC{i}' for i in range(pca_n_components)]
            _ref_pca = PCA(n_components=pca_n_components, random_state=42)
            X_tr_t = pd.DataFrame(_ref_pca.fit_transform(X_tr_t), columns=_pc_cols_t)
            y_tr_t = y_tr_t.reset_index(drop=True)
            X_te_t = pd.DataFrame(_ref_pca.transform(X_te_t), columns=_pc_cols_t)
            y_te_t = y_te_t.reset_index(drop=True)
        model_trad = build_and_train_model(
            args.model, X_tr_t, y_tr_t, model_params,
            calibrate=calibrate, calibrate_temperature=calibrate_temperature,
            calibrate_mode=calibrate_mode, calibrate_target=calibrate_target,
        )
        trad_train_acc = model_trad.score(X_tr_t, y_tr_t)
        trad_test_acc  = model_trad.score(X_te_t, y_te_t)
        trad_split_label = f'Train acc: {trad_train_acc:.4f}  |  Test acc: {trad_test_acc:.4f}'
        print(f'  {trad_split_label}')

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    # LOYO averages — exclude current year which has no test set.
    loyo_stats = [s for s in year_model_stats if s['test_acc'] is not None]
    loyo_avg_train_acc = (
        sum(s['train_acc'] for s in loyo_stats) / len(loyo_stats)
    ) if loyo_stats else None
    loyo_avg_test_acc = (
        sum(s['test_acc'] for s in loyo_stats) / len(loyo_stats)
    ) if loyo_stats else None

    games_per_round = [32, 16, 8, 4, 2, 1]
    summary_lines = ['LEAVE-ONE-YEAR-OUT MODEL PERFORMANCE']
    summary_lines.append('')
    if loyo_avg_train_acc is not None:
        summary_lines.append(
            f'Avg LOYO train acc : {loyo_avg_train_acc:.4f}'
        )
    if loyo_avg_test_acc is not None:
        summary_lines.append(
            f'Avg LOYO test acc  : {loyo_avg_test_acc:.4f}  ({len(loyo_stats)} years)'
        )
    summary_lines.append('')
    summary_lines.append('Per-year model accuracy:')
    for stat in year_model_stats:
        if stat['test_acc'] is not None:
            summary_lines.append(
                f"  {stat['year']}: train={stat['train_acc']:.4f}  test={stat['test_acc']:.4f}"
                + (f"  bracket={stat['bracket_score']} pts" if stat['bracket_score'] is not None else '')
            )
        else:
            summary_lines.append(
                f"  {stat['year']}: train={stat['train_acc']:.4f}  (current year — no test set)"
            )

    summary_lines.append('')
    summary_lines.append('Bracket results:')
    for rnd in range(1, 7):
        total_games = games_per_round[rnd - 1] * num_eval_years
        n = total_correct_by_round[rnd]
        pct = n / total_games * 100 if total_games else 0
        pts = n * (2 ** (rnd - 1)) * 10
        summary_lines.append(
            f'  Round {rnd}: {n}/{total_games} ({pct:.1f}%), {pts} pts'
        )
    total_games_all = 63 * num_eval_years
    total_correct_all = sum(total_correct_by_round)
    if total_games_all:
        summary_lines.append(
            f'  All rounds: {total_correct_all}/{total_games_all} '
            f'({total_correct_all / total_games_all * 100:.1f}%)'
        )
        summary_lines.append(f'  Avg bracket score: {total_score / num_eval_years:.1f}')

    summary_lines.append('')
    if df_sim_raw is not None:
        summary_lines.append('SIM DATA REFERENCE MODEL (train=sim+real, test=real only)')
    else:
        summary_lines.append('TRADITIONAL 67/33 TRAIN-TEST SPLIT MODEL (for reference)')
    summary_lines.append(f'  {trad_split_label}')

    summary_str = '\n'.join(summary_lines)
    print(f'\n{summary_str}')
    (output_root / 'summary.txt').write_text(summary_str)

    # Rename the pending folder to a descriptive name: model_score_expert_features
    avg_score_val = total_score / num_eval_years if num_eval_years else 0
    expert_tag = 'KP'
    if norm_years:
        expert_tag += 'NY'
    elif norm_all:
        expert_tag += 'NA'
    if calibrate:
        expert_tag += 'CAL'
    if delta_feats:
        expert_tag += 'DF'
    if pca_n_components:
        expert_tag += f'PCA{pca_n_components}'
    seen_bases: set = set()
    feat_parts: List[str] = []
    for b in args.features:
        if b not in seen_bases:
            seen_bases.add(b)
            feat_parts.append(b)
    feat_str = '+'.join(feat_parts)
    params_tag = ('+'.join(f'{k}={v}' for k, v in model_params.items())) if model_params else ''
    # Folder name is now simply the user-supplied run name.
    final_dir_name = args.run_name
    final_output_root = output_root.parent / final_dir_name
    if final_output_root.exists():
        shutil.rmtree(final_output_root)
    output_root.rename(final_output_root)
    # Write model metadata for the UI to display.
    model_info = {
        'run_name':       args.run_name,
        'model_key':      args.model,
        'score':          int(avg_score_val),
        'expert_tag':     expert_tag,
        'features':       feat_str,
        'params':         params_tag,
        'norm_years':     norm_years,
        'norm_all':       norm_all,
        'calibrate':          calibrate,
        'calibrate_mode':     calibrate_mode if calibrate else None,
        'calibrate_target':   calibrate_target if (calibrate and calibrate_mode == 'stretch') else None,
        'calibrate_temperature': calibrate_temperature if calibrate else None,
        'delta_feats':        delta_feats,
        'exclude_years':      sorted(exclude_years),
        'sim_data':           sim_data_id,
        'pca_components':     pca_n_components,
        'model_params':   {str(k): str(v) for k, v in model_params.items()},
        'feature_bases':  list(args.features),
        'trad_train_acc': round(trad_train_acc, 4),
        'trad_test_acc':  round(trad_test_acc, 4),
        'loyo_avg_train_acc': round(loyo_avg_train_acc, 4) if loyo_avg_train_acc is not None else None,
        'loyo_avg_test_acc':  round(loyo_avg_test_acc,  4) if loyo_avg_test_acc  is not None else None,
    }
    (final_output_root / 'model_info.json').write_text(json.dumps(model_info, indent=2))
    # -----------------------------------------------------------------------
    # Save the full-data model as a pickle so it can be re-instantiated.
    # If --this-year was supplied the current-year model was already trained on
    # all historical data; otherwise train a fresh model on the full dataset now.
    # -----------------------------------------------------------------------
    if this_year is not None:
        # Reuse the last model trained (the current-year one, trained on all data).
        full_model = model
        full_pca   = fold_pca
    else:
        df_full = df_all_raw.copy()
        if cat_encoders:
            df_full = apply_label_encoders(df_full, cat_encoders)
        if norm_info is not None:
            df_full = apply_year_norm(df_full, norm_info)
        if delta_feats and numeric_bases:
            df_full = apply_delta_transform(df_full, numeric_bases)
        X_full = df_full[model_feature_list]
        y_full = df_full['Win__1']
        full_pca = None
        if pca_n_components:
            _pc_cols_full = [f'PC{i}' for i in range(pca_n_components)]
            full_pca = PCA(n_components=pca_n_components, random_state=42)
            X_full = pd.DataFrame(full_pca.fit_transform(X_full), columns=_pc_cols_full)
            y_full = y_full.reset_index(drop=True)
        full_model = build_and_train_model(
            args.model, X_full, y_full, model_params,
            calibrate=calibrate, calibrate_temperature=calibrate_temperature,
            calibrate_mode=calibrate_mode, calibrate_target=calibrate_target,
        )

    pickle_payload = {
        'model':              full_model,
        'model_key':          args.model,
        'model_params':       model_params,
        'feature_list':       feature_list,
        'model_feature_list': model_feature_list,
        'cat_encoders':       cat_encoders,
        'norm_info':          norm_info,
        'delta_feats':        delta_feats,
        'numeric_bases':      numeric_bases,
        'pca_transformer':    full_pca,
    }
    pickle_path = final_output_root / 'model.pkl'
    with open(pickle_path, 'wb') as fh:
        pickle.dump(pickle_payload, fh)
    print(f'Model pickle saved to: {pickle_path}')


if __name__ == '__main__':
    main()
