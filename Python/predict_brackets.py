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
import os
import pickle
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split
from bracket_html import format_bracket_html

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
    'logistic_regression':    LogisticRegression,
    'knn':                    KNeighborsClassifier,
    'svc':                    SVC,
    'decision_tree':          DecisionTreeClassifier,
    'random_forest':          RandomForestClassifier,
    'gradient_boosting':      GradientBoostingClassifier,
    'adaboost':               AdaBoostClassifier,
    'gpc':                    GaussianProcessClassifier,
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
    Fit a StandardScaler per year using the combined distribution of __1 and __2
    columns for each numeric base (delta mode).

    The scaler for each year is fitted on a stacked DataFrame where __1 and __2
    values are concatenated so that both populations inform the scaling.

    norm_info['cols']     — the base column names (no __1/__2 suffix)
    norm_info['is_delta'] — True, signals delta-aware apply helpers
    """
    avail = [b for b in numeric_bases
             if f'{b}__1' in df.columns and f'{b}__2' in df.columns]
    if not avail:
        return {'by_year': {}, 'fallback': None, 'cols': [], 'is_delta': True}
    by_year: dict = {}
    for year, grp in df.groupby('Year'):
        part1 = grp[[f'{b}__1' for b in avail]].rename(columns={f'{b}__1': b for b in avail})
        part2 = grp[[f'{b}__2' for b in avail]].rename(columns={f'{b}__2': b for b in avail})
        stacked = pd.concat([part1, part2], axis=0, ignore_index=True)
        sc = StandardScaler()
        sc.fit(stacked)
        by_year[year] = sc
    # Fallback: stack over all years.
    all1 = df[[f'{b}__1' for b in avail]].rename(columns={f'{b}__1': b for b in avail})
    all2 = df[[f'{b}__2' for b in avail]].rename(columns={f'{b}__2': b for b in avail})
    fallback = StandardScaler()
    fallback.fit(pd.concat([all1, all2], axis=0, ignore_index=True))
    return {'by_year': by_year, 'fallback': fallback, 'cols': avail, 'is_delta': True}


def fit_global_scaler_delta(df: pd.DataFrame, numeric_bases: List[str]) -> dict:
    """
    Fit a single global StandardScaler using the combined distribution of __1
    and __2 columns for each numeric base (delta / --norm-all counterpart).
    """
    avail = [b for b in numeric_bases
             if f'{b}__1' in df.columns and f'{b}__2' in df.columns]
    if not avail:
        return {'by_year': {}, 'fallback': None, 'cols': [], 'is_delta': True}
    all1 = df[[f'{b}__1' for b in avail]].rename(columns={f'{b}__1': b for b in avail})
    all2 = df[[f'{b}__2' for b in avail]].rename(columns={f'{b}__2': b for b in avail})
    sc = StandardScaler()
    sc.fit(pd.concat([all1, all2], axis=0, ignore_index=True))
    return {'by_year': {}, 'fallback': sc, 'cols': avail, 'is_delta': True}


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


def apply_year_norm(df: pd.DataFrame, norm_info: dict) -> pd.DataFrame:
    """
    Apply per-year Z-score normalisation to a DataFrame that contains a 'Year'
    column.  Each row is scaled by the scaler fitted for its year.

    In delta mode (norm_info['is_delta'] == True) the scaler expects base column
    names (no __1/__2 suffix); both __1 and __2 columns are transformed using the
    same scaler so their scales match before the delta is computed.
    """
    if norm_info is None:
        return df
    is_delta = norm_info.get('is_delta', False)
    cols = norm_info['cols']
    if not cols:
        return df
    df = df.copy()
    if is_delta:
        avail = [b for b in cols if f'{b}__1' in df.columns and f'{b}__2' in df.columns]
        if not avail:
            return df
        for year, grp in df.groupby('Year'):
            sc = norm_info['by_year'].get(year, norm_info['fallback'])
            if sc is None:
                continue
            sub1 = grp[[f'{b}__1' for b in avail]].rename(columns={f'{b}__1': b for b in avail})
            scaled1 = pd.DataFrame(sc.transform(sub1), columns=avail, index=grp.index)
            for b in avail:
                df.loc[grp.index, f'{b}__1'] = scaled1[b].values
            sub2 = grp[[f'{b}__2' for b in avail]].rename(columns={f'{b}__2': b for b in avail})
            scaled2 = pd.DataFrame(sc.transform(sub2), columns=avail, index=grp.index)
            for b in avail:
                df.loc[grp.index, f'{b}__2'] = scaled2[b].values
    else:
        avail = [c for c in cols if c in df.columns]
        if not avail:
            return df
        for year, grp in df.groupby('Year'):
            sc = norm_info['by_year'].get(year, norm_info['fallback'])
            if sc is None:
                continue
            df.loc[grp.index, avail] = sc.transform(grp[avail])
    return df


def apply_year_norm_single(df: pd.DataFrame, year: int, norm_info: dict) -> pd.DataFrame:
    """
    Apply the normalisation scaler for a single known *year* to a DataFrame
    that does NOT have a 'Year' column (e.g. per-round bracket data).

    In delta mode, both __1 and __2 are transformed using the same scaler.
    """
    if norm_info is None:
        return df
    is_delta = norm_info.get('is_delta', False)
    cols = norm_info['cols']
    if not cols:
        return df
    sc = norm_info['by_year'].get(year, norm_info['fallback'])
    if sc is None:
        return df
    df = df.copy()
    if is_delta:
        avail = [b for b in cols if f'{b}__1' in df.columns and f'{b}__2' in df.columns]
        if not avail:
            return df
        sub1 = df[[f'{b}__1' for b in avail]].rename(columns={f'{b}__1': b for b in avail})
        scaled1 = pd.DataFrame(sc.transform(sub1), columns=avail, index=df.index)
        for b in avail:
            df[f'{b}__1'] = scaled1[b].values
        sub2 = df[[f'{b}__2' for b in avail]].rename(columns={f'{b}__2': b for b in avail})
        scaled2 = pd.DataFrame(sc.transform(sub2), columns=avail, index=df.index)
        for b in avail:
            df[f'{b}__2'] = scaled2[b].values
    else:
        avail = [c for c in cols if c in df.columns]
        if not avail:
            return df
        df[avail] = sc.transform(df[avail])
    return df


def attach_kenpom(df_matchups: pd.DataFrame, df_kp: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns Team__1 and Team__2, merge in KenPom stats
    with the KP__ source prefix and __1 / __2 team suffix.
    Seed is dropped — seeds are tracked separately via team_seed_map.
    """
    kp = df_kp.drop(columns=['Seed'], errors='ignore')
    rename_map = {c: f'KP__{c}' for c in kp.columns if c != 'Team'}
    kp = kp.rename(columns=rename_map)
    kp1 = kp.add_suffix('__1')   # Team → Team__1, KP__AdjEM → KP__AdjEM__1
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
# Model training
# ---------------------------------------------------------------------------

def build_and_train_model(model_key: str, X_train: pd.DataFrame, y_train: pd.Series,
                          model_params: dict = None, calibrate: bool = False):
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Options: {list(MODEL_REGISTRY)}")
    params = dict(model_params or {})
    # SVC requires probability=True to support predict_proba; inject it unless
    # the caller explicitly set probability=False.
    if model_key == 'svc' and 'probability' not in params:
        params['probability'] = True
    estimator = MODEL_REGISTRY[model_key](**params)
    estimator.fit(X_train, y_train)
    if calibrate:
        # Wrap with Platt scaling (sigmoid) fitted on the same training data.
        # FrozenEstimator signals that the base estimator is already fitted.
        cal = CalibratedClassifierCV(FrozenEstimator(estimator), method='sigmoid')
        cal.fit(X_train, y_train)
        return cal
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
        preds = model.predict(X)
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
            'Apply Platt scaling (sigmoid calibration) to the trained model to produce '
            'better-calibrated win probabilities. Fitted on the same training data used '
            'to train the base model. Output folder name will include a CAL indicator.'
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
    args = parser.parse_args()

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
    # Write to a pending folder; renamed to model+score+features at the end.
    output_root  = Path(args.output_root) / 'Predictions' / f'__{args.model}__pending'
    output_root.mkdir(parents=True, exist_ok=True)

    this_year = args.this_year
    current_year_ff_pairings = parse_ff_pairings_arg(args.final_four_pairings)

    # Build the list of years to process: all completed years, plus the current
    # year appended at the end if it was supplied and isn't already in ALL_YEARS.
    years_to_process = list(ALL_YEARS)
    if this_year is not None and this_year not in years_to_process:
        years_to_process.append(this_year)
    num_eval_years = len(years_to_process) - (1 if this_year is not None else 0)

    # -----------------------------------------------------------------------
    # Load the full game dataset once (used to build per-year training sets).
    # -----------------------------------------------------------------------
    df_all_raw = load_combined_games(data_root)
    pre_drop = len(df_all_raw)
    df_all_raw = df_all_raw.dropna(subset=feature_list)
    dropped = pre_drop - len(df_all_raw)
    if dropped:
        print(f'Note: dropped {dropped}/{pre_drop} rows with NaN in selected features.')

    # Fit label encoders on the full dataset so all values are known.
    cat_encoders = fit_label_encoders(df_all_raw, cat_cols) if cat_cols else {}

    # Normalisation (optional) — fit scalers on numeric columns only.
    norm_years = args.norm_years
    norm_all   = args.norm_all
    calibrate  = args.calibrate
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
    print(f'Probability calibration: {"ON (Platt sigmoid)" if calibrate else "OFF"}')

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
            # Current year: train on ALL historical data (no test set).
            df_train = df_all_raw.copy()
        else:
            # Historical year: train on every other year to avoid leakage.
            df_train = df_all_raw[df_all_raw['Year'] != year].copy()

        df_test_year = df_all_raw[df_all_raw['Year'] == year].copy() if not is_current else None

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

        X_tr = df_train[model_feature_list]
        y_tr = df_train['Win__1']
        model = build_and_train_model(args.model, X_tr, y_tr, model_params, calibrate=calibrate)

        train_acc = model.score(X_tr, y_tr)
        if not is_current:
            X_te = df_test_year[model_feature_list]
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
    # Traditional random train/test split model (for reference comparison).
    # -----------------------------------------------------------------------
    print(f'\n{"="*50}\nTRADITIONAL 67/33 TRAIN-TEST SPLIT MODEL\n{"="*50}')
    df_trad = df_all_raw[df_all_raw['Year'].isin(ALL_YEARS)].copy()
    if cat_encoders:
        df_trad = apply_label_encoders(df_trad, cat_encoders)
    if norm_info is not None:
        df_trad = apply_year_norm(df_trad, norm_info)
    if delta_feats and numeric_bases:
        df_trad = apply_delta_transform(df_trad, numeric_bases)
    X_trad, y_trad = df_trad[model_feature_list], df_trad['Win__1']
    X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(X_trad, y_trad, test_size=0.33, random_state=42)
    model_trad = build_and_train_model(args.model, X_tr_t, y_tr_t, model_params, calibrate=calibrate)
    trad_train_acc = model_trad.score(X_tr_t, y_tr_t)
    trad_test_acc  = model_trad.score(X_te_t, y_te_t)
    print(f'  Train acc: {trad_train_acc:.4f}  |  Test acc: {trad_test_acc:.4f}')

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    games_per_round = [32, 16, 8, 4, 2, 1]
    summary_lines = ['LEAVE-ONE-YEAR-OUT MODEL PERFORMANCE']
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
    summary_lines.append('TRADITIONAL 67/33 TRAIN-TEST SPLIT MODEL (for reference)')
    summary_lines.append(f'  Train acc: {trad_train_acc:.4f}  |  Test acc: {trad_test_acc:.4f}')

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
    seen_bases: set = set()
    feat_parts: List[str] = []
    for b in args.features:
        if b not in seen_bases:
            seen_bases.add(b)
            feat_parts.append(b)
    feat_str = '+'.join(feat_parts)
    params_tag = ('+'.join(f'{k}={v}' for k, v in model_params.items())) if model_params else ''
    final_dir_name = (
        f'{args.model}_{int(avg_score_val)}_{expert_tag}_{feat_str}'
        + (f'_{params_tag}' if params_tag else '')
    )
    final_output_root = output_root.parent / final_dir_name
    if final_output_root.exists():
        shutil.rmtree(final_output_root)
    output_root.rename(final_output_root)
    # -----------------------------------------------------------------------
    # Save the full-data model as a pickle so it can be re-instantiated.
    # If --this-year was supplied the current-year model was already trained on
    # all historical data; otherwise train a fresh model on the full dataset now.
    # -----------------------------------------------------------------------
    if this_year is not None:
        # Reuse the last model trained (the current-year one, trained on all data).
        full_model = model
    else:
        df_full = df_all_raw.copy()
        if cat_encoders:
            df_full = apply_label_encoders(df_full, cat_encoders)
        if norm_info is not None:
            df_full = apply_year_norm(df_full, norm_info)
        if delta_feats and numeric_bases:
            df_full = apply_delta_transform(df_full, numeric_bases)
        full_model = build_and_train_model(args.model, df_full[model_feature_list], df_full['Win__1'], model_params, calibrate=calibrate)

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
    }
    pickle_path = final_output_root / 'model.pkl'
    with open(pickle_path, 'wb') as fh:
        pickle.dump(pickle_payload, fh)
    print(f'Model pickle saved to: {pickle_path}')


if __name__ == '__main__':
    main()
