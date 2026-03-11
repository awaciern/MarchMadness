#!/usr/bin/env python3
"""
feature_viz.py

Stand-alone feature exploration / analysis tool.
Called as a subprocess by app.py; outputs a single JSON blob to stdout.

Usage (CLI):
    python3 feature_viz.py --data-root /path/to/repo
    python3 feature_viz.py --data-root /path/to/repo --exclude-years 2012 2019
    python3 feature_viz.py --data-root /path/to/repo --rounds 1 2 --top-n 20

Arguments:
  --data-root  DIR             Repo root (must contain Data/).
  --exclude-years YEAR ...     Years to exclude from the analysis.
  --features  BASE ...         Subset of feature base names to analyse. Default: all numeric.
  --rounds    R ...            Tournament rounds to include (1–6). Default: all.
  --top-n     N                Number of top features to include in rf_importance output. Default: 40.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Feature registry — mirrors predict_brackets.py
# ---------------------------------------------------------------------------

COMMON_BASES: List[str] = ['WinPct', 'Wins', 'Losses']   # Conf excluded (non-ordinal)

KP_ONLY_BASES: List[str] = [
    'KP_AdjO', 'KP_Rk_AdjO', 'KP_AdjD', 'KP_Rk_AdjD', 'KP_AdjT', 'KP_Rk_AdjT',
    'AdjEM', 'Rk_AdjEM',
    'Luck', 'Rk_Luck',
    'SOS_AdjEM', 'Rk_SOS_AdjEM',
    'SOS_AdjO', 'Rk_SOS_AdjO',
    'SOS_AdjD', 'Rk_SOS_AdjD',
    'NCSOS_AdjEM', 'Rk_NCSOS_AdjEM',
]

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

SEED_BASES: List[str] = ['Seed']

ALL_NUMERIC_BASES: List[str] = (
    COMMON_BASES + KP_ONLY_BASES + BT_ONLY_BASES + BT2W_BASES + BTHOT_BASES + SEED_BASES
)

# Groups for UI filtering
FEATURE_GROUPS = {
    'common': COMMON_BASES,
    'kp':     KP_ONLY_BASES,
    'bt':     BT_ONLY_BASES,
    'bt2w':   BT2W_BASES,
    'bthot':  BTHOT_BASES,
    'seed':   SEED_BASES,
}


def resolve_feature_col(base: str) -> str:
    """Map an unprefixed base name to its source-prefixed column base (same as predict_brackets.py)."""
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


def short_label(base: str) -> str:
    """Human-readable short label for a feature base name."""
    # KP_Adj* and BT_Adj* would both strip to the same name (e.g. AdjO),
    # so keep a short source tag to keep labels unique.
    if base.startswith('KP_Adj'):
        return 'KP ' + base[3:]   # KP_AdjO -> KP AdjO
    if base.startswith('BT_Adj'):
        return 'BT ' + base[3:]   # BT_AdjO -> BT AdjO
    if base.startswith('KP_Rk_Adj'):
        return 'KP ' + base[3:]
    if base.startswith('BT_Rk_Adj'):
        return 'BT ' + base[3:]
    # Generic prefix stripping for non-ambiguous KP/BT features
    if base.startswith('KP_'):
        return base[3:]
    if base.startswith('BT_'):
        return base[3:]
    if base.startswith('2W_'):
        return '[2W] ' + base[3:]
    if base.startswith('HOT_'):
        return '[HOT] ' + base[4:]
    return base


def group_of(base: str) -> str:
    for g, members in FEATURE_GROUPS.items():
        if base in members:
            return g
    return 'other'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    data_root: Path,
    exclude_years: List[int],
    rounds: Optional[List[int]],
    bases: List[str],
) -> pd.DataFrame:
    """Load GameCombinedData/All.csv filtered by excluded years and rounds."""
    csv = data_root / 'Data' / 'GameCombinedData' / 'All.csv'
    if not csv.exists():
        raise FileNotFoundError(f'Combined game data not found: {csv}')
    df = pd.read_csv(csv, low_memory=False)

    if exclude_years:
        df = df[~df['Year'].isin(exclude_years)]
    if rounds:
        df = df[df['Round'].isin(rounds)]

    return df


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_win_correlations(df: pd.DataFrame, delta_cols: List[str], labels: dict) -> list:
    """
    Point-biserial correlation between each delta column and Win__1.
    Returns list of dicts sorted by abs(corr) descending.
    """
    win = df['Win__1'].astype(int)
    results = []
    for col in delta_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        common_idx = series.index.intersection(win.index)
        if len(common_idx) < 10:
            continue
        x = series.loc[common_idx].values
        y = win.loc[common_idx].values
        # Remove rows where x is NaN/inf
        mask = np.isfinite(x)
        if mask.sum() < 10:
            continue
        corr, pval = stats.pointbiserialr(x[mask], y[mask])
        if np.isnan(corr):
            continue
        # Win rate when delta > 0 (team1 has higher value)
        pos_mask = (x[mask] > 0)
        win_rate_higher = float(y[mask][pos_mask].mean()) if pos_mask.sum() > 0 else 0.5
        results.append({
            'feature': col,
            'base':    labels[col]['base'],
            'label':   labels[col]['label'],
            'group':   labels[col]['group'],
            'corr':    round(float(corr), 4),
            'pvalue':  float(pval),
            'abs_corr': round(abs(float(corr)), 4),
            'win_rate_higher': round(win_rate_higher, 4),
            'n':       int(mask.sum()),
        })
    results.sort(key=lambda x: x['abs_corr'], reverse=True)
    return results


def compute_win_rate_by_delta(df: pd.DataFrame, delta_cols: List[str], labels: dict) -> list:
    """
    For each feature delta, compute win rates in quantile bins of the delta.
    Returns data suitable for a line/bar chart.
    """
    win = df['Win__1'].astype(int)
    results = []
    n_bins = 10
    for col in delta_cols:
        if col not in df.columns:
            continue
        series = df[col]
        common_idx = series.dropna().index.intersection(win.index)
        if len(common_idx) < 30:
            continue
        x = series.loc[common_idx].values.astype(float)
        y = win.loc[common_idx].values
        mask = np.isfinite(x)
        if mask.sum() < 30:
            continue
        x_f, y_f = x[mask], y[mask]
        try:
            bins = np.quantile(x_f, np.linspace(0, 1, n_bins + 1))
            bins = np.unique(bins)
            if len(bins) < 3:
                continue
            bin_centers = []
            bin_winrates = []
            bin_counts = []
            for i in range(len(bins) - 1):
                in_bin = (x_f >= bins[i]) & (x_f < bins[i + 1])
                if in_bin.sum() < 3:
                    continue
                bin_centers.append(round(float((bins[i] + bins[i + 1]) / 2), 3))
                bin_winrates.append(round(float(y_f[in_bin].mean()), 4))
                bin_counts.append(int(in_bin.sum()))
            results.append({
                'feature':    col,
                'base':       labels[col]['base'],
                'label':      labels[col]['label'],
                'group':      labels[col]['group'],
                'bin_centers': bin_centers,
                'win_rates':   bin_winrates,
                'bin_counts':  bin_counts,
            })
        except Exception:
            continue
    return results


def compute_rf_importance(
    df: pd.DataFrame,
    delta_cols: List[str],
    labels: dict,
    top_n: int = 40,
) -> list:
    """
    Train a RandomForest on all available delta columns; return feature importances.
    """
    avail = [c for c in delta_cols if c in df.columns]
    if not avail:
        return []
    sub = df[avail + ['Win__1']].dropna()
    if len(sub) < 30:
        return []
    X = sub[avail].values
    y = sub['Win__1'].astype(int).values
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    results = []
    for col, imp in zip(avail, importances):
        results.append({
            'feature':    col,
            'base':       labels[col]['base'],
            'label':      labels[col]['label'],
            'group':      labels[col]['group'],
            'importance': round(float(imp), 6),
        })
    results.sort(key=lambda x: x['importance'], reverse=True)
    return results[:top_n]


def compute_corr_matrix(df: pd.DataFrame, delta_cols: List[str], labels: dict) -> dict:
    """
    Correlation matrix between selected delta columns.
    Returns dict with feature list, label list, and matrix (list of lists).
    """
    avail = [c for c in delta_cols if c in df.columns]
    if not avail:
        return {'features': [], 'labels': [], 'groups': [], 'matrix': []}
    sub = df[avail].dropna()
    if len(sub) < 10:
        return {'features': [], 'labels': [], 'groups': [], 'matrix': []}
    corr = sub.corr().values
    return {
        'features': avail,
        'labels':   [labels[c]['label'] for c in avail],
        'groups':   [labels[c]['group'] for c in avail],
        'matrix':   [[round(float(v), 3) for v in row] for row in corr],
    }


def compute_round_breakdown(df: pd.DataFrame, delta_cols: List[str], labels: dict) -> list:
    """
    For the top features (by overall correlation), compute their correlation per round.
    Returns list of {feature, label, rounds: [{round, corr, n}]}.
    """
    win = df['Win__1'].astype(int)
    avail = [c for c in delta_cols if c in df.columns]
    rounds = sorted(df['Round'].dropna().unique().astype(int).tolist())

    # First find top 10 by overall correlation
    overall = []
    for col in avail:
        series = df[col].dropna()
        common_idx = series.index.intersection(win.index)
        if len(common_idx) < 20:
            continue
        x = series.loc[common_idx].values.astype(float)
        y = win.loc[common_idx].values
        mask = np.isfinite(x)
        if mask.sum() < 20:
            continue
        corr, _ = stats.pointbiserialr(x[mask], y[mask])
        if not np.isnan(corr):
            overall.append((abs(corr), col))
    overall.sort(reverse=True)
    top_cols = [c for _, c in overall[:15]]

    results = []
    for col in top_cols:
        round_data = []
        for rnd in rounds:
            rdf = df[df['Round'] == rnd]
            series = rdf[col].dropna()
            common_idx = series.index.intersection(win.loc[rdf.index].index)
            if len(common_idx) < 5:
                round_data.append({'round': int(rnd), 'corr': None, 'n': 0})
                continue
            x = series.loc[common_idx].values.astype(float)
            y = win.loc[common_idx].values
            mask = np.isfinite(x)
            if mask.sum() < 5:
                round_data.append({'round': int(rnd), 'corr': None, 'n': 0})
                continue
            corr, _ = stats.pointbiserialr(x[mask], y[mask])
            round_data.append({'round': int(rnd), 'corr': round(float(corr), 4) if not np.isnan(corr) else None, 'n': int(mask.sum())})
        results.append({
            'feature':    col,
            'base':       labels[col]['base'],
            'label':      labels[col]['label'],
            'group':      labels[col]['group'],
            'rounds':     round_data,
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Feature visualization analysis.')
    parser.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--exclude-years', nargs='*', type=int, default=[])
    parser.add_argument('--features', nargs='*', default=None,
                        help='Base feature names to analyse. Default: all numeric.')
    parser.add_argument('--rounds', nargs='*', type=int, default=None,
                        help='Tournament rounds to include (1-6). Default: all.')
    parser.add_argument('--top-n', type=int, default=40,
                        help='Max features to return in RF importance output.')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    exclude_years: List[int] = args.exclude_years or []
    rounds: Optional[List[int]] = args.rounds or None
    top_n: int = args.top_n

    # Resolve which base features to analyse
    if args.features:
        bases = [b for b in args.features if b in ALL_NUMERIC_BASES]
    else:
        bases = list(ALL_NUMERIC_BASES)

    # Build col -> label/group map and delta column names
    delta_cols: List[str] = []
    labels: dict = {}
    for base in bases:
        col_base = resolve_feature_col(base)
        c1 = f'{col_base}__{1}'
        c2 = f'{col_base}__{2}'
        delta_name = f'{col_base}__delta'
        labels[delta_name] = {
            'base':  base,
            'label': short_label(base),
            'group': group_of(base),
            'col1':  c1,
            'col2':  c2,
        }
        delta_cols.append(delta_name)

    # Load data
    try:
        df_raw = load_data(data_root, exclude_years, rounds, bases)
    except FileNotFoundError as e:
        json.dump({'error': str(e)}, sys.stdout)
        return

    if df_raw.empty:
        json.dump({'error': 'No data after filtering.'}, sys.stdout)
        return

    # Compute delta columns (team1 - team2), only for existing columns
    df = df_raw.copy()
    for delta_name, info in labels.items():
        c1, c2 = info['col1'], info['col2']
        if c1 in df.columns and c2 in df.columns:
            df[delta_name] = pd.to_numeric(df[c1], errors='coerce') - pd.to_numeric(df[c2], errors='coerce')
        # else column stays absent — analyses will skip it

    years_used = sorted(int(y) for y in df['Year'].dropna().unique())
    n_games = int(len(df))

    # Run analyses
    win_corr   = compute_win_correlations(df, delta_cols, labels)
    rf_imp     = compute_rf_importance(df, delta_cols, labels, top_n=top_n)
    corr_mat   = compute_corr_matrix(df, delta_cols[:40], labels)  # limit matrix to first 40 for performance
    rnd_brkdwn = compute_round_breakdown(df, delta_cols, labels)

    output = {
        'n_games':          n_games,
        'years_used':       years_used,
        'rounds_used':      sorted(int(r) for r in df['Round'].dropna().unique()),
        'n_features':       len([c for c in delta_cols if c in df.columns]),
        'win_correlation':  win_corr,
        'rf_importance':    rf_imp,
        'corr_matrix':      corr_mat,
        'round_breakdown':  rnd_brkdwn,
    }

    json.dump(output, sys.stdout, indent=None)


if __name__ == '__main__':
    main()
