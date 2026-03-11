"""
app.py

Flask web UI for running March Madness bracket predictions.
Exposes a form where the user can configure a model, then runs
predict_brackets.py under the hood and streams live output back
to the browser via Server-Sent Events.

Usage:
    python3 Python/app.py
    # then open http://localhost:5000
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, request, send_file, abort

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT        = Path(__file__).resolve().parents[1]
PYTHON_EXE       = str(REPO_ROOT / 'env' / 'bin' / 'python3')
PREDICT_SCRIPT   = str(Path(__file__).resolve().parent / 'predict_brackets.py')
SIMULATE_SCRIPT  = str(Path(__file__).resolve().parent / 'simulate_bracket.py')
PREDICTIONS_DIR  = REPO_ROOT / 'Predictions'
SIMULATIONS_DIR  = REPO_ROOT / 'Simulations'
BRACKETS_DIR     = REPO_ROOT / 'Brackets'
THIS_YEAR        = 2026

# ---------------------------------------------------------------------------
# Feature / model metadata (mirrored from predict_brackets.py)
# ---------------------------------------------------------------------------

COMMON_BASES = [
    'WinPct', 'Wins', 'Losses',
    'Conf',
]

KP_ONLY_BASES = [
    'KP_AdjO', 'KP_Rk_AdjO', 'KP_AdjD', 'KP_Rk_AdjD', 'KP_AdjT', 'KP_Rk_AdjT',
    'AdjEM', 'Rk_AdjEM',
    'Luck', 'Rk_Luck',
    'SOS_AdjEM', 'Rk_SOS_AdjEM',
    'SOS_AdjO', 'Rk_SOS_AdjO',
    'SOS_AdjD', 'Rk_SOS_AdjD',
    'NCSOS_AdjEM', 'Rk_NCSOS_AdjEM',
]

BT_ONLY_BASES = [
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

BT2W_BASES = [
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

BTHOT_BASES = [
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

METADATA_BASES = ['Seed']

DEFAULT_FEATURES = ['WinPct', 'KP_AdjO', 'KP_AdjD', 'AdjEM']

# ---------------------------------------------------------------------------
# Simplified UI feature lists (6 categories shown in the web form)
# ---------------------------------------------------------------------------

UI_BASIC_BASES = ['WinPct', 'Conf', 'Seed']

UI_KP_BASES = [
    'AdjEM', 'Rk_AdjEM',
    'KP_AdjO', 'KP_Rk_AdjO', 'KP_AdjD', 'KP_Rk_AdjD', 'KP_AdjT', 'KP_Rk_AdjT',
    'Luck', 'Rk_Luck',
    'SOS_AdjEM', 'Rk_SOS_AdjEM',
    'NCSOS_AdjEM', 'Rk_NCSOS_AdjEM',
]

UI_BT_BASES = [
    'Barthag', 'Rk_Barthag',
    'WAB', 'Rk_WAB',
    'BT_AdjO', 'BT_Rk_AdjO', 'BT_AdjD', 'BT_Rk_AdjD', 'BT_AdjT', 'BT_Rk_AdjT',
]

UI_STATS_BASES = [
    'EFG%', 'Rk_EFG%', 'EFGD%', 'Rk_EFGD%',
    'TOR', 'Rk_TOR', 'TORD', 'Rk_TORD',
    'ORB', 'Rk_ORB', 'DRB', 'Rk_DRB',
    'FTR', 'Rk_FTR', 'FTRD', 'Rk_FTRD',
    '2P%', 'Rk_2P%', '2P%D', 'Rk_2P%D',
    '3P%', 'Rk_3P%', '3P%D', 'Rk_3P%D',
    '3PR', 'Rk_3PR', '3PRD', 'Rk_3PRD',
]

UI_BT2W_BASES = [
    '2W_Barthag', '2W_Rk_Barthag',
    '2W_WAB', '2W_Rk_WAB',
    '2W_AdjO', '2W_Rk_AdjO', '2W_AdjD', '2W_Rk_AdjD', '2W_AdjT', '2W_Rk_AdjT',
]

UI_BTHOT_BASES = [
    'HOT_Barthag', 'HOT_Rk_Barthag',
    'HOT_WAB', 'HOT_Rk_WAB',
    'HOT_AdjO', 'HOT_Rk_AdjO', 'HOT_AdjD', 'HOT_Rk_AdjD', 'HOT_AdjT', 'HOT_Rk_AdjT',
]

MODELS = [
    ('logistic_regression', 'Logistic Regression'),
    ('knn',                 'k-Nearest Neighbors'),
    ('svc',                 'Support Vector Machine (SVC)'),
    ('decision_tree',       'Decision Tree'),
    ('random_forest',       'Random Forest'),
    ('gradient_boosting',   'Gradient Boosting'),
    ('adaboost',            'AdaBoost'),
    ('gpc',                 'Gaussian Process'),
]

FEATURE_DESCRIPTIONS = {
    # ---- Common (always KenPom) -----------------------------------------
    'WinPct':         'Win percentage (wins / total games)',
    'Wins':           'Total wins in the season',
    'Losses':         'Total losses in the season',
    'Conf':           'Athletic conference affiliation',
    # ---- KenPom (KP__ prefix) -------------------------------------------
    'KP_AdjO':        '(KenPom) Adjusted Offensive Efficiency — points scored per 100 possessions vs. average defense',
    'KP_Rk_AdjO':     '(KenPom) National ranking for Adjusted Offensive Efficiency',
    'KP_AdjD':        '(KenPom) Adjusted Defensive Efficiency — points allowed per 100 possessions vs. average offense (lower is better)',
    'KP_Rk_AdjD':     '(KenPom) National ranking for Adjusted Defensive Efficiency (lower rank = better defense)',
    'KP_AdjT':        '(KenPom) Adjusted Tempo — estimated possessions per 40 minutes vs. average opponent',
    'KP_Rk_AdjT':     '(KenPom) National ranking for Adjusted Tempo',
    'AdjEM':          "KenPom's Adjusted Efficiency Margin = AdjO minus AdjD (primary overall team rating)",
    'Rk_AdjEM':       'National ranking for Adjusted Efficiency Margin',
    'Luck':           "KenPom's Luck rating — how much a team over/under-performed its expected win percentage",
    'Rk_Luck':        'National ranking for Luck rating',
    'SOS_AdjEM':      'Strength of Schedule — average AdjEM of all opponents faced',
    'Rk_SOS_AdjEM':   'National ranking for Strength of Schedule',
    'SOS_AdjO':       'Opponent average Adjusted Offensive Efficiency (how tough your schedule was on defense)',
    'Rk_SOS_AdjO':    'National ranking for Opponent Offensive Efficiency faced',
    'SOS_AdjD':       'Opponent average Adjusted Defensive Efficiency (how tough your schedule was on offense)',
    'Rk_SOS_AdjD':    'National ranking for Opponent Defensive Efficiency faced',
    'NCSOS_AdjEM':    'Non-Conference Strength of Schedule — average AdjEM of non-conference opponents',
    'Rk_NCSOS_AdjEM': 'National ranking for Non-Conference Strength of Schedule',
    # ---- BartTorvik (BT__ prefix) ----------------------------------------
    'BT_AdjO':        '(BartTorvik) Adjusted Offensive Efficiency — points scored per 100 possessions vs. average defense',
    'BT_Rk_AdjO':     '(BartTorvik) National ranking for Adjusted Offensive Efficiency',
    'BT_AdjD':        '(BartTorvik) Adjusted Defensive Efficiency — points allowed per 100 possessions vs. average offense (lower is better)',
    'BT_Rk_AdjD':     '(BartTorvik) National ranking for Adjusted Defensive Efficiency (lower rank = better defense)',
    'BT_AdjT':        '(BartTorvik) Adjusted Tempo — estimated possessions per 40 minutes vs. average opponent',
    'BT_Rk_AdjT':     '(BartTorvik) National ranking for Adjusted Tempo',
    'Barthag':        "BartTorvik's Power Rating — probability of beating an average D1 team on a neutral court",
    'Rk_Barthag':     'National ranking for Barthag Power Rating',
    'EFG%':           'Effective Field Goal % — accounts for 3-pointers being worth more: (FGM + 0.5x3PM) / FGA',
    'Rk_EFG%':        'National ranking for Effective Field Goal %',
    'EFGD%':          'Opponent Effective Field Goal % allowed (defensive eFG%)',
    'Rk_EFGD%':       'National ranking for Opponent Effective Field Goal % (lower = better perimeter defense)',
    'TOR':            'Turnover Rate — turnovers committed per 100 possessions (lower is better)',
    'Rk_TOR':         'National ranking for Turnover Rate (lower rank = fewer turnovers)',
    'TORD':           'Opponent Turnover Rate — turnovers forced per 100 possessions (higher is better)',
    'Rk_TORD':        'National ranking for Opponent Turnover Rate (higher rank = more turnovers forced)',
    'ORB':            'Offensive Rebound Rate — percentage of available offensive rebounds grabbed',
    'Rk_ORB':         'National ranking for Offensive Rebound Rate',
    'DRB':            'Defensive Rebound Rate — percentage of available defensive rebounds secured',
    'Rk_DRB':         'National ranking for Defensive Rebound Rate',
    'FTR':            'Free Throw Rate — free throw attempts per field goal attempt (how often a team draws fouls)',
    'Rk_FTR':         'National ranking for Free Throw Rate',
    'FTRD':           'Opponent Free Throw Rate — how often opponents get to the free throw line against you',
    'Rk_FTRD':        'National ranking for Opponent Free Throw Rate (lower = better at limiting opponent FTs)',
    '2P%':            'Two-point field goal percentage',
    'Rk_2P%':         'National ranking for Two-Point Field Goal %',
    '2P%D':           'Opponent two-point field goal % allowed (interior defense)',
    'Rk_2P%D':        'National ranking for Opponent Two-Point % (lower = better interior defense)',
    '3P%':            'Three-point field goal percentage',
    'Rk_3P%':         'National ranking for Three-Point Field Goal %',
    '3P%D':           'Opponent three-point field goal % allowed (perimeter defense)',
    'Rk_3P%D':        'National ranking for Opponent Three-Point % (lower = better perimeter defense)',
    '3PR':            'Three-Point Attempt Rate — share of all field goal attempts that are 3-pointers',
    'Rk_3PR':         'National ranking for Three-Point Attempt Rate',
    '3PRD':           'Opponent Three-Point Attempt Rate — share of opponent shot attempts that are 3-pointers',
    'Rk_3PRD':        'National ranking for Opponent Three-Point Attempt Rate',
    'WAB':            'Wins Above Bubble — wins relative to how a bubble-level team would perform against the same schedule',
    'Rk_WAB':         'National ranking for Wins Above Bubble',
    # ---- Bracket metadata ------------------------------------------------
    'Seed':           'Tournament seed (1 = top seed, 16 = lowest seed in each region)',
    # ---- 2-week BartTorvik snapshot (BT2W__ prefix) ----------------------
    '2W_WinPct':      '[2-week] Win percentage over the last two weeks',
    '2W_Wins':        '[2-week] Total wins over the last two weeks',
    '2W_Losses':      '[2-week] Total losses over the last two weeks',
    '2W_AdjO':        '[2-week] Adjusted Offensive Efficiency — 2-week snapshot',
    '2W_Rk_AdjO':     '[2-week] Ranking for Adjusted Offensive Efficiency',
    '2W_AdjD':        '[2-week] Adjusted Defensive Efficiency — 2-week snapshot',
    '2W_Rk_AdjD':     '[2-week] Ranking for Adjusted Defensive Efficiency',
    '2W_AdjT':        '[2-week] Adjusted Tempo — 2-week snapshot',
    '2W_Rk_AdjT':     '[2-week] Ranking for Adjusted Tempo',
    '2W_Barthag':     '[2-week] BartTorvik Power Rating — 2-week snapshot',
    '2W_Rk_Barthag':  '[2-week] Ranking for Barthag Power Rating',
    '2W_EFG%':        '[2-week] Effective Field Goal % — 2-week snapshot',
    '2W_Rk_EFG%':     '[2-week] Ranking for Effective Field Goal %',
    '2W_EFGD%':       '[2-week] Opponent Effective Field Goal % allowed — 2-week snapshot',
    '2W_Rk_EFGD%':    '[2-week] Ranking for Opponent Effective Field Goal %',
    '2W_TOR':         '[2-week] Turnover Rate — 2-week snapshot',
    '2W_Rk_TOR':      '[2-week] Ranking for Turnover Rate',
    '2W_TORD':        '[2-week] Opponent Turnover Rate — 2-week snapshot',
    '2W_Rk_TORD':     '[2-week] Ranking for Opponent Turnover Rate',
    '2W_ORB':         '[2-week] Offensive Rebound Rate — 2-week snapshot',
    '2W_Rk_ORB':      '[2-week] Ranking for Offensive Rebound Rate',
    '2W_DRB':         '[2-week] Defensive Rebound Rate — 2-week snapshot',
    '2W_Rk_DRB':      '[2-week] Ranking for Defensive Rebound Rate',
    '2W_FTR':         '[2-week] Free Throw Rate — 2-week snapshot',
    '2W_Rk_FTR':      '[2-week] Ranking for Free Throw Rate',
    '2W_FTRD':        '[2-week] Opponent Free Throw Rate — 2-week snapshot',
    '2W_Rk_FTRD':     '[2-week] Ranking for Opponent Free Throw Rate',
    '2W_2P%':         '[2-week] Two-point field goal percentage — 2-week snapshot',
    '2W_Rk_2P%':      '[2-week] Ranking for Two-Point %',
    '2W_2P%D':        '[2-week] Opponent two-point % allowed — 2-week snapshot',
    '2W_Rk_2P%D':     '[2-week] Ranking for Opponent Two-Point %',
    '2W_3P%':         '[2-week] Three-point field goal percentage — 2-week snapshot',
    '2W_Rk_3P%':      '[2-week] Ranking for Three-Point %',
    '2W_3P%D':        '[2-week] Opponent three-point % allowed — 2-week snapshot',
    '2W_Rk_3P%D':     '[2-week] Ranking for Opponent Three-Point %',
    '2W_3PR':         '[2-week] Three-Point Attempt Rate — 2-week snapshot',
    '2W_Rk_3PR':      '[2-week] Ranking for Three-Point Attempt Rate',
    '2W_3PRD':        '[2-week] Opponent Three-Point Attempt Rate — 2-week snapshot',
    '2W_Rk_3PRD':     '[2-week] Ranking for Opponent Three-Point Attempt Rate',
    '2W_WAB':         '[2-week] Wins Above Bubble — 2-week snapshot',
    '2W_Rk_WAB':      '[2-week] Ranking for Wins Above Bubble',
    # ---- Hotness BartTorvik diff (BTHOT__ prefix) ------------------------
    'HOT_WinPct':     '[Hotness] Win % change: 2-week minus season-long (+ = improving)',
    'HOT_Wins':       '[Hotness] Wins difference: 2-week minus season-long',
    'HOT_Losses':     '[Hotness] Losses difference: 2-week minus season-long',
    'HOT_AdjO':       '[Hotness] AdjO change: 2-week minus season-long (+ = offense improving)',
    'HOT_Rk_AdjO':    '[Hotness] AdjO rank change (negative = moving up the rankings)',
    'HOT_AdjD':       '[Hotness] AdjD change: 2-week minus season-long (negative = defense improving)',
    'HOT_Rk_AdjD':    '[Hotness] AdjD rank change',
    'HOT_AdjT':       '[Hotness] Adjusted Tempo change',
    'HOT_Rk_AdjT':    '[Hotness] Adjusted Tempo rank change',
    'HOT_Barthag':    '[Hotness] Barthag power rating change (+ = getting stronger)',
    'HOT_Rk_Barthag': '[Hotness] Barthag rank change',
    'HOT_EFG%':       '[Hotness] Effective Field Goal % change',
    'HOT_Rk_EFG%':    '[Hotness] Effective Field Goal % rank change',
    'HOT_EFGD%':      '[Hotness] Opponent Effective Field Goal % change',
    'HOT_Rk_EFGD%':   '[Hotness] Opponent Effective Field Goal % rank change',
    'HOT_TOR':        '[Hotness] Turnover Rate change',
    'HOT_Rk_TOR':     '[Hotness] Turnover Rate rank change',
    'HOT_TORD':       '[Hotness] Opponent Turnover Rate change',
    'HOT_Rk_TORD':    '[Hotness] Opponent Turnover Rate rank change',
    'HOT_ORB':        '[Hotness] Offensive Rebound Rate change',
    'HOT_Rk_ORB':     '[Hotness] Offensive Rebound Rate rank change',
    'HOT_DRB':        '[Hotness] Defensive Rebound Rate change',
    'HOT_Rk_DRB':     '[Hotness] Defensive Rebound Rate rank change',
    'HOT_FTR':        '[Hotness] Free Throw Rate change',
    'HOT_Rk_FTR':     '[Hotness] Free Throw Rate rank change',
    'HOT_FTRD':       '[Hotness] Opponent Free Throw Rate change',
    'HOT_Rk_FTRD':    '[Hotness] Opponent Free Throw Rate rank change',
    'HOT_2P%':        '[Hotness] Two-point % change',
    'HOT_Rk_2P%':     '[Hotness] Two-point % rank change',
    'HOT_2P%D':       '[Hotness] Opponent two-point % change',
    'HOT_Rk_2P%D':    '[Hotness] Opponent two-point % rank change',
    'HOT_3P%':        '[Hotness] Three-point % change',
    'HOT_Rk_3P%':     '[Hotness] Three-point % rank change',
    'HOT_3P%D':       '[Hotness] Opponent three-point % change',
    'HOT_Rk_3P%D':    '[Hotness] Opponent three-point % rank change',
    'HOT_3PR':        '[Hotness] Three-Point Attempt Rate change',
    'HOT_Rk_3PR':     '[Hotness] Three-Point Attempt Rate rank change',
    'HOT_3PRD':       '[Hotness] Opponent Three-Point Attempt Rate change',
    'HOT_Rk_3PRD':    '[Hotness] Opponent Three-Point Attempt Rate rank change',
    'HOT_WAB':        '[Hotness] Wins Above Bubble change (+ = trending toward stronger performance)',
    'HOT_Rk_WAB':     '[Hotness] Wins Above Bubble rank change',
}

ALL_YEARS = [y for y in range(2012, THIS_YEAR + 1) if y != 2020]

# ---------------------------------------------------------------------------
# Saved-model scanner
# ---------------------------------------------------------------------------

_SAVED_PATTERN = re.compile(r'^(.+?)_(\d+)_((?:KP|BT)\w*)_(.+)$')


def _parse_trad_acc(folder: Path):
    """Parse Traditional 67/33 train/test accuracy from a Predictions folder's summary.txt."""
    summary = folder / 'summary.txt'
    if not summary.exists():
        return None, None
    m = re.search(
        r'Train acc:\s*([\d.]+)\s*\|.*?Test acc:\s*([\d.]+)',
        summary.read_text(),
    )
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None

def scan_saved_models():
    """
    Scan PREDICTIONS_DIR for completed model folders (not __pending__ dirs).
    Returns a list of dicts sorted by score descending.
    Folders with model_info.json use that for metadata; legacy name-encoded
    folders fall back to _SAVED_PATTERN parsing.
    """
    results = []
    if not PREDICTIONS_DIR.is_dir():
        return results
    for d in PREDICTIONS_DIR.iterdir():
        if not d.is_dir() or d.name.startswith('__'):
            continue
        years = [y for y in ALL_YEARS if (d / f'{y}.html').exists()]
        # --- new-style: metadata in model_info.json ---
        info_file = d / 'model_info.json'
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text())
                expert_tag = info.get('expert_tag', '')
                train_acc, test_acc = info.get('trad_train_acc'), info.get('trad_test_acc')
                if train_acc is None:
                    train_acc, test_acc = _parse_trad_acc(d)
                results.append({
                    'dir_name':   d.name,
                    'score':      info.get('score', 0),
                    'model':      info.get('model_key', d.name),
                    'expert':     expert_tag,
                    'norm_years': info.get('norm_years', False),
                    'norm_all':   info.get('norm_all', False),
                    'calibrated': info.get('calibrate', False),
                    'delta_feats':info.get('delta_feats', False),
                    'features':   info.get('features', ''),
                    'params':     info.get('params', ''),
                    'train_acc':  train_acc,
                    'test_acc':   test_acc,
                    'has_pkl':    (d / 'model.pkl').exists(),
                    'years':      years,
                })
                continue
            except Exception:
                pass
        # --- legacy: decode metadata from folder name ---
        m = _SAVED_PATTERN.match(d.name)
        if not m:
            continue
        model_key, score_str, expert_tag, rest = m.groups()
        score = int(score_str)
        feat_str, params_str = rest, ''
        pm = re.search(r'_([^_]+=.+)$', rest)
        if pm:
            params_str = pm.group(1)
            feat_str = rest[:pm.start()]
        train_acc, test_acc = _parse_trad_acc(d)
        results.append({
            'dir_name':   d.name,
            'score':      score,
            'model':      model_key,
            'expert':     expert_tag,
            'norm_years': 'NY' in expert_tag,
            'norm_all':   'NA' in expert_tag,
            'calibrated': 'CAL' in expert_tag,
            'delta_feats':'DF' in expert_tag,
            'features':   feat_str,
            'params':     params_str,
            'train_acc':  train_acc,
            'test_acc':   test_acc,
            'has_pkl':    (d / 'model.pkl').exists(),
            'years':      years,
        })
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


def scan_simulations(year: int = THIS_YEAR) -> list:
    """
    Scan SIMULATIONS_DIR for model folders that have CSV results for `year`.
    Returns dicts with dir_name, model, score, iters, and whether a matching
    model.pkl exists in PREDICTIONS_DIR.
    Folders backed by model_info.json use that for characteristics; legacy
    name-encoded folders fall back to _SAVED_PATTERN.
    """
    results = []
    if not SIMULATIONS_DIR.is_dir():
        return results
    for d in SIMULATIONS_DIR.iterdir():
        if not d.is_dir():
            continue
        csv_files = sorted(
            d.glob(f'{year}_*iters.csv'),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not csv_files:
            continue
        latest = csv_files[0]
        iters_match = re.search(r'_(\d+)iters\.csv$', latest.name)
        iters = int(iters_match.group(1)) if iters_match else 0
        html_file = latest.with_suffix('.html').name
        has_html  = (d / html_file).exists()
        has_pkl = (PREDICTIONS_DIR / d.name / 'model.pkl').exists()
        # Try model_info.json from the matching Predictions folder first.
        info_file = PREDICTIONS_DIR / d.name / 'model_info.json'
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text())
                expert_tag = info.get('expert_tag', '')
                train_acc, test_acc = info.get('trad_train_acc'), info.get('trad_test_acc')
                if train_acc is None:
                    train_acc, test_acc = _parse_trad_acc(PREDICTIONS_DIR / d.name)
                results.append({
                    'dir_name':   d.name,
                    'model':      info.get('model_key', d.name),
                    'score':      info.get('score', 0),
                    'expert':     expert_tag,
                    'norm_years': info.get('norm_years', False),
                    'norm_all':   info.get('norm_all', False),
                    'calibrated': info.get('calibrate', False),
                    'delta_feats':info.get('delta_feats', False),
                    'features':   info.get('features', ''),
                    'params':     info.get('params', ''),
                    'train_acc':  train_acc,
                    'test_acc':   test_acc,
                    'iters':      iters,
                    'has_pkl':    has_pkl,
                    'csv_file':   latest.name,
                    'html_file':  html_file if has_html else '',
                })
                continue
            except Exception:
                pass
        # Legacy: parse from folder name.
        m = _SAVED_PATTERN.match(d.name)
        if not m:
            continue
        model_key, score_str, expert_tag, rest = m.groups()
        train_acc, test_acc = _parse_trad_acc(PREDICTIONS_DIR / d.name)
        results.append({
            'dir_name':   d.name,
            'model':      model_key,
            'score':      int(score_str),
            'expert':     expert_tag,
            'norm_years': 'NY' in expert_tag,
            'norm_all':   'NA' in expert_tag,
            'calibrated': 'CAL' in expert_tag,
            'delta_feats':'DF' in expert_tag,
            'features':   rest,
            'params':     '',
            'train_acc':  train_acc,
            'test_acc':   test_acc,
            'iters':      iters,
            'has_pkl':    has_pkl,
            'csv_file':   latest.name,
            'html_file':  html_file if has_html else '',
        })
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

class Job:
    def __init__(self):
        self.status     = 'running'   # 'running' | 'done' | 'error'
        self.lines      = []          # all output captured so far
        self.queue      = queue.Queue()
        self.output_dir = None        # set when process finishes


jobs: dict[str, Job] = {}


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

def run_job(job_id: str, cmd: list):
    job = jobs[job_id]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        for line in proc.stdout:
            line = line.rstrip('\n')
            job.lines.append(line)
            job.queue.put(line)
            # Detect pickle save line to capture output dir
            if 'Model pickle saved to:' in line:
                path_str = line.split('Model pickle saved to:', 1)[1].strip()
                job.output_dir = Path(path_str).parent
        proc.wait()
        job.status = 'done' if proc.returncode == 0 else 'error'
    except Exception as exc:
        job.lines.append(f'[ERROR] {exc}')
        job.queue.put(f'[ERROR] {exc}')
        job.status = 'error'
    finally:
        job.queue.put(None)  # sentinel


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string(INDEX_HTML,
        models=MODELS,
        ui_basic_bases=UI_BASIC_BASES,
        ui_kp_bases=UI_KP_BASES,
        ui_bt_bases=UI_BT_BASES,
        ui_stats_bases=UI_STATS_BASES,
        ui_bt2w_bases=UI_BT2W_BASES,
        ui_bthot_bases=UI_BTHOT_BASES,
        default_features=DEFAULT_FEATURES,
        feature_descs=FEATURE_DESCRIPTIONS,
    )


@app.route('/run', methods=['POST'])
def run_prediction():
    data = request.get_json()

    run_name = (data.get('run_name') or '').strip()
    if not run_name:
        return jsonify({'error': 'Model name is required.'}), 400
    # Ensure the name is a safe folder name (no path separators or leading dots).
    if any(c in run_name for c in ('/', '\\', '\0')) or run_name.startswith('.'):
        return jsonify({'error': 'Model name contains invalid characters.'}), 400

    model   = data.get('model', 'logistic_regression')
    params  = data.get('params', '').strip()   # raw "key=val key=val" string
    features = data.get('features', DEFAULT_FEATURES)

    if not features:
        return jsonify({'error': 'Select at least one feature.'}), 400

    cmd = [
        PYTHON_EXE, PREDICT_SCRIPT,
        '--model', model,
        '--features', *features,
        '--this-year', str(THIS_YEAR),
        '--final-four-pairings', _load_ff_pairings_str(THIS_YEAR),
        '--run-name', run_name,
    ]
    if params:
        cmd += ['--model-params'] + params.split()
    if data.get('norm_years'):
        cmd.append('--norm-years')
    if data.get('norm_all'):
        cmd.append('--norm-all')
    if data.get('calibrate'):
        cmd.append('--calibrate')
    if data.get('delta_feats'):
        cmd.append('--delta-feats')

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = Job()
    t = threading.Thread(target=run_job, args=(job_id, cmd), daemon=True)
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/stream/<job_id>')
def stream(job_id):
    if job_id not in jobs:
        abort(404)

    def generate():
        job = jobs[job_id]
        while True:
            try:
                line = job.queue.get(timeout=300)
            except queue.Empty:
                yield 'data: {"type":"timeout"}\n\n'
                break
            if line is None:
                # Send final status + output_dir
                payload = {
                    'type': 'done',
                    'status': job.status,
                    'output_dir': str(job.output_dir) if job.output_dir else None,
                }
                yield f'data: {json.dumps(payload)}\n\n'
                break
            yield f'data: {json.dumps({"type": "line", "text": line})}\n\n'

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/results/<job_id>')
def results(job_id):
    if job_id not in jobs:
        abort(404)
    job = jobs[job_id]
    if job.status == 'running':
        return jsonify({'status': 'running'}), 202

    # Read summary.txt if available
    summary = ''
    if job.output_dir and (job.output_dir / 'summary.txt').exists():
        summary = (job.output_dir / 'summary.txt').read_text()

    # Discover which year HTML files exist
    year_files = []
    if job.output_dir and job.output_dir.is_dir():
        for y in ALL_YEARS:
            f = job.output_dir / f'{y}.html'
            if f.exists():
                year_files.append(y)

    return jsonify({
        'status': job.status,
        'summary': summary,
        'years': year_files,
        'dir_name': job.output_dir.name if job.output_dir else None,
    })


@app.route('/bracket/<job_id>/<int:year>')
def bracket(job_id, year):
    if job_id not in jobs:
        abort(404)
    job = jobs[job_id]
    if not job.output_dir:
        abort(404)
    html_file = job.output_dir / f'{year}.html'
    if not html_file.exists():
        abort(404)
    return send_file(str(html_file), mimetype='text/html')


@app.route('/saved_models')
def saved_models_route():
    return jsonify(scan_saved_models())


@app.route('/saved_results/<path:dir_name>')
def saved_results(dir_name):
    d = PREDICTIONS_DIR / dir_name
    if not d.is_dir():
        abort(404)
    summary = (d / 'summary.txt').read_text() if (d / 'summary.txt').exists() else ''
    years = [y for y in ALL_YEARS if (d / f'{y}.html').exists()]
    return jsonify({'summary': summary, 'years': years, 'dir_name': dir_name})


@app.route('/saved_bracket/<path:dir_name>/<int:year>')
def saved_bracket(dir_name, year):
    html_file = PREDICTIONS_DIR / dir_name / f'{year}.html'
    if not html_file.exists():
        abort(404)
    return send_file(str(html_file), mimetype='text/html')


@app.route('/save_model_bracket', methods=['POST'])
def save_model_bracket():
    """Parse the existing year.html from a Predictions folder and save the
    model's picks to Brackets/<group>/<name>.json — no model re-run needed."""
    import re as _re
    import html as _html_lib
    import datetime as _dt

    data     = request.get_json(force=True)
    dir_name = (data.get('dir_name') or '').strip()
    year     = int(data.get('year') or THIS_YEAR)
    name     = (data.get('name') or '').strip()
    group    = (data.get('group') or 'Models').strip()

    if not dir_name:
        return jsonify({'error': 'dir_name is required.'}), 400
    if not name:
        return jsonify({'error': 'Bracket name is required.'}), 400
    if not group:
        return jsonify({'error': 'Group name is required.'}), 400

    html_file = PREDICTIONS_DIR / dir_name / f'{year}.html'
    if not html_file.exists():
        return jsonify({'error': f'{year}.html not found in Predictions/{dir_name}.'}), 400

    html_text = html_file.read_text(encoding='utf-8')

    # The bracket HTML has 11 round columns (mirrored left→right):
    #   0=R1-left  1=R2-left  2=S16-left  3=E8-left  4=FF-left
    #   5=Championship
    #   6=FF-right  7=E8-right  8=S16-right  9=R2-right  10=R1-right
    # Teams with class "adv" in each column = winners for that round/side.
    rnd_blocks = html_text.split('<div class="rnd"')[1:]  # drop text before first column
    if len(rnd_blocks) != 11:
        return jsonify({'error': f'Unexpected bracket structure ({len(rnd_blocks)} columns; expected 11).'}), 400

    def _adv_teams(block):
        """Extract winner names from a round column, in bracket order (by top px)."""
        hits = []
        for m in _re.finditer(
            r'top:(\d+)px[^>]*>.*?class="c adv[^"]*">.*?</span>(.*?)<span',
            block, _re.DOTALL,
        ):
            hits.append((int(m.group(1)), _html_lib.unescape(m.group(2).strip())))
        hits.sort(key=lambda x: x[0])
        return [t for _, t in hits]

    # Combine left + right halves in bracket order for each round
    semi_teams = _adv_teams(rnd_blocks[4]) + _adv_teams(rnd_blocks[6])
    champ_list  = _adv_teams(rnd_blocks[5])
    picks = {
        'r1':       _adv_teams(rnd_blocks[0])  + _adv_teams(rnd_blocks[10]),
        'r2':       _adv_teams(rnd_blocks[1])  + _adv_teams(rnd_blocks[9]),
        's16':      _adv_teams(rnd_blocks[2])  + _adv_teams(rnd_blocks[8]),
        'e8':       _adv_teams(rnd_blocks[3])  + _adv_teams(rnd_blocks[7]),
        'semi':     semi_teams,
        'champion': champ_list[0] if champ_list else None,
    }

    safe_name  = re.sub(r'[^\w\-\. ]', '', name).strip().replace(' ', '_')
    safe_group = re.sub(r'[^\w\-\. ]', '', group).strip().replace(' ', '_')
    if not safe_name or not safe_group:
        return jsonify({'error': 'Name or group contains only invalid characters.'}), 400

    out_dir = BRACKETS_DIR / safe_group
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{safe_name}.json'

    bracket_payload = {
        'name':        name,
        'group':       group,
        'year':        year,
        'created':     _dt.datetime.now().isoformat(timespec='seconds'),
        'ff_pairings': _load_ff_pairings_str(year),
        'model_dir':   dir_name,
        'picks':       picks,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(bracket_payload, f, indent=2)

    return jsonify({'saved': True, 'path': str(out_path.relative_to(REPO_ROOT))})


# ---------------------------------------------------------------------------
# Simulation routes
# ---------------------------------------------------------------------------

sim_jobs: dict = {}

group_jobs: dict = {}   # job_id → Job (with .results attribute added dynamically)


def run_sim_job(job_id: str, cmd: list):
    job = sim_jobs[job_id]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        for line in proc.stdout:
            line = line.rstrip('\n')
            job.lines.append(line)
            job.queue.put(line)
            if 'HTML saved to:' in line:
                job.output_dir = Path(line.split('HTML saved to:', 1)[1].strip())
        proc.wait()
        job.status = 'done' if proc.returncode == 0 else 'error'
    except Exception as exc:
        job.lines.append(f'[ERROR] {exc}')
        job.queue.put(f'[ERROR] {exc}')
        job.status = 'error'
    finally:
        job.queue.put(None)


@app.route('/simulate', methods=['POST'])
def simulate():
    data      = request.get_json()
    dir_name  = (data.get('dir_name') or '').strip()
    year      = int(data.get('year', THIS_YEAR))
    num_iters = int(data.get('num_iters', 1000))
    seed      = data.get('seed')

    pkl_path = PREDICTIONS_DIR / dir_name / 'model.pkl'
    if not pkl_path.exists():
        return jsonify({
            'error': f'model.pkl not found in {dir_name}.\nRe-run the prediction to generate it.'
        }), 400

    cmd = [
        PYTHON_EXE, SIMULATE_SCRIPT,
        '--model', str(pkl_path),
        '--year',  str(year),
        '--num-iters', str(num_iters),
        '--final-four-pairings', _load_ff_pairings_str(year),
    ]
    if seed is not None:
        cmd += ['--seed', str(seed)]

    job_id = str(uuid.uuid4())[:8]
    sim_jobs[job_id] = Job()
    t = threading.Thread(target=run_sim_job, args=(job_id, cmd), daemon=True)
    t.start()
    return jsonify({'job_id': job_id})


@app.route('/sim_stream/<job_id>')
def sim_stream(job_id):
    if job_id not in sim_jobs:
        abort(404)

    def generate():
        job = sim_jobs[job_id]
        while True:
            try:
                line = job.queue.get(timeout=300)
            except queue.Empty:
                yield 'data: {"type":"timeout"}\n\n'
                break
            if line is None:
                payload = {
                    'type':      'done',
                    'status':    job.status,
                    'html_path': str(job.output_dir) if job.output_dir else None,
                }
                yield f'data: {json.dumps(payload)}\n\n'
                break
            yield f'data: {json.dumps({"type": "line", "text": line})}\n\n'

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/sim_list/<path:dir_name>')
def sim_list_route(dir_name):
    sim_dir = SIMULATIONS_DIR / dir_name
    files = []
    if sim_dir.is_dir():
        files = [
            f.name for f in sorted(
                sim_dir.glob('*iters.html'),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        ]
    return jsonify({'files': files, 'dir_name': dir_name})


@app.route('/sim_html/<path:dir_name>/<path:filename>')
def sim_html_route(dir_name, filename):
    html_file = SIMULATIONS_DIR / dir_name / filename
    if not html_file.exists():
        abort(404)
    return send_file(str(html_file), mimetype='text/html')


# ---------------------------------------------------------------------------
# Bracket input routes
# ---------------------------------------------------------------------------

# Standard NCAA bracket seed matchups within each region, in top-to-bottom order.
SEED_MATCHUPS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

REGION_NAMES = ['South', 'East', 'Midwest', 'West']

# All possible Final Four pairings expressed as (str, [[a,b],[c,d]], label)
FF_PAIRING_OPTIONS = [
    ('0-3,1-2', [[0, 3], [1, 2]], 'South/West · East/Midwest'),
    ('0-1,2-3', [[0, 1], [2, 3]], 'South/East · Midwest/West'),
    ('0-2,1-3', [[0, 2], [1, 3]], 'South/Midwest · East/West'),
]


def _load_ff_pairings_str(year: int = THIS_YEAR) -> str:
    """Load saved FF pairings string from BracketData, defaulting to '0-3,1-2'."""
    p = REPO_ROOT / 'Data' / 'BracketData' / str(year) / f'FFPairings_{year}.json'
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return data.get('pairings', '0-3,1-2')
        except Exception:
            pass
    return '0-3,1-2'


def _ff_pairings_label(pairings_str: str) -> str:
    """Return a human-readable label for a pairings string like '0-3,1-2'."""
    for s, _, label in FF_PAIRING_OPTIONS:
        if s == pairings_str:
            return label
    return pairings_str


def _locked_results_path(year: int = THIS_YEAR) -> Path:
    return REPO_ROOT / 'Data' / 'BracketData' / str(year) / f'LockedResults_{year}.json'


def _load_locked_results(year: int = THIS_YEAR) -> dict:
    """
    Load locked game results for `year`.
    Returns {round_int: [winner_or_None, ...]} or {} if no file exists.
    """
    p = _locked_results_path(year)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        raw = data.get('round_winners', {})
        return {int(k): [(v if v else None) for v in v_list]
                for k, v_list in raw.items()}
    except Exception:
        return {}


def _read_kp_teams(year: int = THIS_YEAR):
    """Return sorted list of team names from KenPom CSV for the given year."""
    import csv as _csv
    kp_path = REPO_ROOT / 'Data' / 'KenPomData' / f'{year}.csv'
    if not kp_path.exists():
        return []
    teams = []
    with open(kp_path, newline='', encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            t = row.get('Team', '').strip()
            if t:
                teams.append(t)
    return sorted(teams)


def _read_round1_teams(year: int = THIS_YEAR):
    """
    Read existing Round1_<year>.csv and return team names by row order.
    Returns a list of 32 dicts {team1, team2}, padded with empty strings.
    Seeds come from SEED_MATCHUPS, not the CSV.
    """
    import csv as _csv
    csv_path = REPO_ROOT / 'Data' / 'BracketData' / str(year) / f'Round1_{year}.csv'
    games = []
    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                games.append({
                    'team1': row.get('Team1', '').strip(),
                    'team2': row.get('Team2', '').strip(),
                })
    while len(games) < 32:
        games.append({'team1': '', 'team2': ''})
    return games[:32]


def _build_display_games(team_rows):
    """
    Annotate 32 team rows with their fixed seed values from SEED_MATCHUPS.
    Returns list of 32 dicts: {team1, team2, seed1, seed2}.
    """
    result = []
    for i, row in enumerate(team_rows):
        s1, s2 = SEED_MATCHUPS[i % 8]
        result.append({
            'team1': row['team1'],
            'team2': row['team2'],
            'seed1': s1,
            'seed2': s2,
        })
    return result


@app.route('/bracket_input')
def bracket_input():
    teams = _read_kp_teams(THIS_YEAR)
    games = _build_display_games(_read_round1_teams(THIS_YEAR))
    return render_template_string(
        BRACKET_INPUT_HTML,
        year=THIS_YEAR,
        teams=teams,
        games=games,
        region_names=REGION_NAMES,
        ff_pairing_options=FF_PAIRING_OPTIONS,
        ff_pairings_str=_load_ff_pairings_str(THIS_YEAR),
        saved=False,
        error=None,
    )


@app.route('/save_bracket', methods=['POST'])
def save_bracket():
    import csv as _csv
    games_out = []
    for i in range(1, 33):
        s1, s2 = SEED_MATCHUPS[(i - 1) % 8]
        games_out.append({
            'Team1':       request.form.get(f'game_{i}_team1', '').strip(),
            'Team1_Seed':  s1,
            'Team1_Score': '',
            'Team2':       request.form.get(f'game_{i}_team2', '').strip(),
            'Team2_Seed':  s2,
            'Team2_Score': '',
            'WinningTeam': '',
            'Team1_Win':   '',
        })

    # Validate: all team names required
    missing = [i + 1 for i, g in enumerate(games_out) if not g['Team1'] or not g['Team2']]
    teams = _read_kp_teams(THIS_YEAR)
    if missing:
        games_redisplay = _build_display_games([
            {'team1': request.form.get(f'game_{i}_team1', '').strip(),
             'team2': request.form.get(f'game_{i}_team2', '').strip()}
            for i in range(1, 33)
        ])
        ff_pairings_post = request.form.get('ff_pairings', '0-3,1-2')
        return render_template_string(
            BRACKET_INPUT_HTML,
            year=THIS_YEAR,
            teams=teams,
            games=games_redisplay,
            region_names=REGION_NAMES,
            ff_pairing_options=FF_PAIRING_OPTIONS,
            ff_pairings_str=ff_pairings_post,
            saved=False,
            error=f'Missing team name(s) in game(s): {", ".join(str(m) for m in missing[:8])}{"…" if len(missing) > 8 else ""}',
        )

    out_dir = REPO_ROOT / 'Data' / 'BracketData' / str(THIS_YEAR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'Round1_{THIS_YEAR}.csv'
    fieldnames = ['Team1', 'Team1_Seed', 'Team1_Score', 'Team2', 'Team2_Seed',
                  'Team2_Score', 'WinningTeam', 'Team1_Win']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(games_out)

    # Save FF pairings to Data/BracketData/<year>/FFPairings_<year>.json
    ff_pairings_str = request.form.get('ff_pairings', '0-3,1-2')
    valid_ff = {s for s, _, _ in FF_PAIRING_OPTIONS}
    if ff_pairings_str not in valid_ff:
        ff_pairings_str = '0-3,1-2'
    ff_pairs_list = [[0, 3], [1, 2]]
    ff_label = ''
    for s, pairs, label in FF_PAIRING_OPTIONS:
        if s == ff_pairings_str:
            ff_pairs_list = pairs
            ff_label = label
            break
    ff_info = {'pairings': ff_pairings_str, 'pairs': ff_pairs_list, 'label': ff_label}
    import json as _json
    with open(out_dir / f'FFPairings_{THIS_YEAR}.json', 'w', encoding='utf-8') as f:
        _json.dump(ff_info, f, indent=2)

    games_redisplay = _build_display_games([
        {'team1': g['Team1'], 'team2': g['Team2']} for g in games_out
    ])
    return render_template_string(
        BRACKET_INPUT_HTML,
        year=THIS_YEAR,
        teams=teams,
        games=games_redisplay,
        region_names=REGION_NAMES,
        ff_pairing_options=FF_PAIRING_OPTIONS,
        ff_pairings_str=ff_pairings_str,
        saved=True,
        error=None,
    )


# ---------------------------------------------------------------------------
# Fill-out-my-bracket routes
# ---------------------------------------------------------------------------

def _load_round1_matchups(year: int = THIS_YEAR):
    """Load Round1_<year>.csv and return list of matchup dicts (32 items)."""
    import csv as _csv
    csv_path = REPO_ROOT / 'Data' / 'BracketData' / str(year) / f'Round1_{year}.csv'
    matchups = []
    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            for i, row in enumerate(reader):
                matchups.append({
                    'index':   i,
                    'region':  i // 8,
                    'team1':   row.get('Team1', '').strip(),
                    'seed1':   int(row.get('Team1_Seed', 0) or 0),
                    'team2':   row.get('Team2', '').strip(),
                    'seed2':   int(row.get('Team2_Seed', 0) or 0),
                })
    return matchups


@app.route('/fill_bracket')
def fill_bracket_route():
    matchups = _load_round1_matchups(THIS_YEAR)
    if not matchups:
        abort(404)
    current_ff_str = _load_ff_pairings_str(THIS_YEAR)
    current_ff_pairs = [[0, 3], [1, 2]]
    for s, pairs, _ in FF_PAIRING_OPTIONS:
        if s == current_ff_str:
            current_ff_pairs = pairs
            break
    return render_template_string(
        FILL_BRACKET_HTML,
        year=THIS_YEAR,
        matchups_json=json.dumps(matchups),
        region_names_json=json.dumps(REGION_NAMES),
        region_names=REGION_NAMES,
        ff_pairings_str=current_ff_str,
        ff_pairings_json=json.dumps(current_ff_pairs),
        ff_pairings_label=_ff_pairings_label(current_ff_str),
    )


@app.route('/save_my_bracket', methods=['POST'])
def save_my_bracket():
    import datetime
    data       = request.get_json(force=True)
    name       = (data.get('name') or '').strip()
    group      = (data.get('group') or '').strip()
    picks      = data.get('picks', {})
    # FF pairings are set at the bracket-setup level; read from server-side file
    ff_pairings = _load_ff_pairings_str(THIS_YEAR)

    if not name:
        return jsonify({'error': 'Bracket name is required.'}), 400
    if not group:
        return jsonify({'error': 'Group name is required.'}), 400

    safe_name  = re.sub(r'[^\w\-\. ]', '', name).strip().replace(' ', '_')
    safe_group = re.sub(r'[^\w\-\. ]', '', group).strip().replace(' ', '_')
    if not safe_name or not safe_group:
        return jsonify({'error': 'Name or group contains only invalid characters.'}), 400

    out_dir = BRACKETS_DIR / safe_group
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{safe_name}.json'

    payload = {
        'name':       name,
        'group':      group,
        'year':       THIS_YEAR,
        'created':    datetime.datetime.now().isoformat(timespec='seconds'),
        'ff_pairings': ff_pairings,
        'picks':      picks,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    return jsonify({'saved': True, 'path': str(out_path.relative_to(REPO_ROOT))})


@app.route('/api/bracket/<path:group>/<string:filename>')
def api_bracket(group, filename):
    fpath = BRACKETS_DIR / group / filename
    if not fpath.exists() or fpath.suffix != '.json':
        abort(404)
    return jsonify(json.loads(fpath.read_text()))


@app.route('/my_brackets')
def my_brackets_route():
    groups = {}
    if BRACKETS_DIR.is_dir():
        for gdir in sorted(BRACKETS_DIR.iterdir()):
            if not gdir.is_dir():
                continue
            brackets = []
            for f in sorted(gdir.glob('*.json')):
                if f.name.startswith('group_results_'):
                    continue
                try:
                    b = json.loads(f.read_text())
                    brackets.append({
                        'name':     b.get('name', f.stem),
                        'group':    b.get('group', gdir.name),
                        'year':     b.get('year', THIS_YEAR),
                        'created':  b.get('created', ''),
                        'file':     f.name,
                        'champion': (b.get('picks') or {}).get('champion') or '',
                    })
                except Exception:
                    pass
            if brackets:
                groups[gdir.name] = brackets
    return render_template_string(MY_BRACKETS_HTML, groups=groups, year=THIS_YEAR)


# ---------------------------------------------------------------------------
# Group bracket analysis routes
# ---------------------------------------------------------------------------

def _run_group_scoring(job_id: str, pkl_path_str: str, group_key: str,
                       year: int, num_iters: int, rng_seed):
    """
    Worker thread: run Monte Carlo simulations of the tournament and,
    for each iteration, score every user bracket in the group.

    FF pairings are loaded from Data/BracketData/<year>/FFPairings_<year>.json.
    Streams PROGRESS:<done>/<total> lines through job.queue.
    When finished sets job.results and job.status='done'.
    """
    import pickle as _pickle
    import traceback as _tb

    job = group_jobs[job_id]
    try:
        import numpy as _np
        import pathlib as _pl

        # Lazy import — keeps startup fast
        sys.path.insert(0, str(REPO_ROOT / 'Python'))
        from simulate_bracket import (                 # pylint: disable=import-error
            _simulate_one, to_html, get_actual_results, ROUND_LABELS as _ROUND_LABELS,
        )
        from predict_brackets import (                 # pylint: disable=import-error
            load_bracket_round, load_kenpom,
            load_barttorvik, load_barttorvik_2week, load_barttorvik_hotness,
            attach_kenpom, attach_barttorvik,
            attach_barttorvik_2week, attach_barttorvik_hotness,
            apply_label_encoders, apply_year_norm_single,
            apply_delta_transform, parse_ff_pairings_arg,
        )

        # ---- Load model pickle ----
        with open(pkl_path_str, 'rb') as fh:
            payload = _pickle.load(fh)

        model              = payload['model']
        model_key          = payload.get('model_key', _pl.Path(pkl_path_str).parent.name)
        feature_list       = payload['feature_list']
        cat_encoders       = payload.get('cat_encoders', {})
        norm_info          = payload.get('norm_info')
        delta_feats        = payload.get('delta_feats', False)
        numeric_bases      = payload.get('numeric_bases', [])
        model_feature_list = payload.get('model_feature_list', feature_list)

        ff_pairings_str = _load_ff_pairings_str(year)
        ff_pairings = parse_ff_pairings_arg(ff_pairings_str)

        data_root   = REPO_ROOT
        needs_bt    = any(f.startswith('BT__')    for f in feature_list)
        needs_bt2w  = any(f.startswith('BT2W__')  for f in feature_list)
        needs_bthot = any(f.startswith('BTHOT__') for f in feature_list)

        # ---- Load Round 1 fixture ----
        r1_df_raw = load_bracket_round(data_root, year, 1)
        team_seed_map: dict = {}
        for _, row in r1_df_raw.iterrows():
            team_seed_map[row['Team__1']] = row['Seed__1']
            team_seed_map[row['Team__2']] = row['Seed__2']

        # ---- Load stat files ----
        df_kp   = load_kenpom(data_root, year)
        df_bt   = load_barttorvik(data_root, year)         if needs_bt    else None
        df_bt2w = load_barttorvik_2week(data_root, year)   if needs_bt2w  else None
        df_hot  = load_barttorvik_hotness(data_root, year) if needs_bthot else None

        # ---- Pre-compute Round 1 features (fixed matchups) ----
        df_r1_kp = attach_kenpom(r1_df_raw[['Team__1', 'Team__2']].copy(), df_kp)
        if needs_bt   and df_bt   is not None:
            df_r1_kp = attach_barttorvik(df_r1_kp, df_bt)
        if needs_bt2w and df_bt2w is not None:
            df_r1_kp = attach_barttorvik_2week(df_r1_kp, df_bt2w)
        if needs_bthot and df_hot is not None:
            df_r1_kp = attach_barttorvik_hotness(df_r1_kp, df_hot)
        df_r1_kp['Seed__1'] = df_r1_kp['Team__1'].map(team_seed_map)
        df_r1_kp['Seed__2'] = df_r1_kp['Team__2'].map(team_seed_map)

        df_r1_enc = apply_label_encoders(df_r1_kp, cat_encoders) if cat_encoders else df_r1_kp.copy()
        if norm_info is not None:
            df_r1_enc = apply_year_norm_single(df_r1_enc, year, norm_info)
        if delta_feats and numeric_bases:
            df_r1_enc = apply_delta_transform(df_r1_enc, numeric_bases)
        r1_proba  = model.predict_proba(df_r1_enc[model_feature_list])
        r1_teams1 = df_r1_kp['Team__1'].tolist()
        r1_teams2 = df_r1_kp['Team__2'].tolist()

        # bracket insertion order (Team__1 then Team__2 per game, no duplicates)
        all_teams: list = []
        for t1, t2 in zip(r1_teams1, r1_teams2):
            if t1 not in all_teams:
                all_teams.append(t1)
            if t2 not in all_teams:
                all_teams.append(t2)

        # ---- Load all user brackets in the group ----
        group_dir = BRACKETS_DIR / group_key
        brackets: list = []
        for f in sorted(group_dir.glob('*.json')):
            if f.name.startswith('group_results_'):
                continue
            try:
                b = json.loads(f.read_text())
                brackets.append({
                    'name':  b.get('name', f.stem),
                    'picks': b.get('picks', {}),
                    'file':  f.name,
                })
            except Exception:
                pass

        if not brackets:
            job.status = 'error'
            job.queue.put('[ERROR] No brackets found in group.')
            job.queue.put(None)
            return

        names  = [b['name'] for b in brackets]
        scores = {n: [] for n in names}

        from collections import defaultdict as _defaultdict
        round_wins: dict = _defaultdict(lambda: _defaultdict(int))

        rng = _np.random.default_rng(rng_seed)
        # Load locked game results (ground truth) for the current year
        _locked_results = _load_locked_results(year)
        # (key, sim_round_number, points_per_correct_pick)
        _rnd_cfg = [
            ('r1',   1,  10),
            ('r2',   2,  20),
            ('s16',  3,  40),
            ('e8',   4,  80),
            ('semi', 5, 160),
        ]
        prog_interval = max(1, num_iters // 200)

        # ---- Monte Carlo loop ----
        for it in range(num_iters):
            draws_r1 = rng.random(len(r1_teams1))
            sim = _simulate_one(
                model, r1_teams1, r1_teams2, r1_proba,
                df_kp, df_bt, df_bt2w, df_hot,
                feature_list, cat_encoders,
                team_seed_map, ff_pairings,
                needs_bt, needs_bt2w, needs_bthot,
                draws_r1, rng,
                norm_info=norm_info,
                year=year,
                delta_feats=delta_feats,
                numeric_bases=numeric_bases,
                model_feature_list=model_feature_list,
                locked_results=_locked_results,
            )

            for b in brackets:
                picks = b['picks']
                sc = 0
                for (key, sim_rnd, pts) in _rnd_cfg:
                    for u, s in zip(picks.get(key) or [], sim.get(sim_rnd) or []):
                        if u and u == s:
                            sc += pts
                champ_pick = picks.get('champion')
                if champ_pick and sim.get(6) and champ_pick == sim[6][0]:
                    sc += 320
                scores[b['name']].append(sc)

            # accumulate team advancement counts for simulation table
            for rnd, winners in sim.items():
                for team in winners:
                    round_wins[team][rnd] += 1

            if (it + 1) % prog_interval == 0 or it + 1 == num_iters:
                job.queue.put(f'PROGRESS:{it + 1}/{num_iters}')

        # ---- Aggregate: win counts (ties split equally) ----
        win_counts = {n: 0.0 for n in names}
        for it in range(num_iters):
            iter_scores = [(n, scores[n][it]) for n in names]
            best = max(sc for _, sc in iter_scores)
            tied = [n for n, sc in iter_scores if sc == best]
            share = 1.0 / len(tied)
            for n in tied:
                win_counts[n] += share

        results = []
        for b in brackets:
            n = b['name']
            sc_list = scores[n]
            results.append({
                'name':      n,
                'file':      b['file'],
                'avg_score': round(sum(sc_list) / num_iters, 1),
                'win_prob':  round(win_counts[n] / num_iters, 4),
                'max_score': int(max(sc_list)),
                'min_score': int(min(sc_list)),
            })

        results.sort(key=lambda x: (-x['win_prob'], -x['avg_score']))
        job.results = results
        job.status  = 'done'

        # ---- Build and save simulation table (team advancement probs) ----
        import pandas as _pd
        sim_rows = []
        for team in all_teams:
            row = {'Team': team, 'Seed': team_seed_map.get(team, '?')}
            for rnd, lbl in enumerate(_ROUND_LABELS, start=1):
                row[lbl] = round_wins[team][rnd] / num_iters
            sim_rows.append(row)
        df_sim = _pd.DataFrame(sim_rows)
        df_sim = df_sim.sort_values(
            list(reversed(_ROUND_LABELS)), ascending=False
        ).reset_index(drop=True)

        _actual = {}
        try:
            _actual = get_actual_results(REPO_ROOT, year)
        except Exception:
            pass

        _dir_name = _pl.Path(pkl_path_str).parent.name
        _sim_out_dir = SIMULATIONS_DIR / _dir_name
        _sim_out_dir.mkdir(parents=True, exist_ok=True)
        _stem = f'{year}_{num_iters}iters'
        _html_path = _sim_out_dir / f'{_stem}.html'
        _csv_path  = _sim_out_dir / f'{_stem}.csv'
        _html_path.write_text(
            to_html(df_sim, year, num_iters, model_key, _actual or None),
            encoding='utf-8',
        )
        df_sim.to_csv(_csv_path, index=False)
        job.sim_url = f'/sim_html/{_dir_name}/{_stem}.html'

        # ---- Save group analysis results to disk ----
        import datetime as _dt
        _save_dir = BRACKETS_DIR / group_key
        _save_dir.mkdir(parents=True, exist_ok=True)
        _ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        _save_name = f'group_results_{_dir_name}_{year}_{num_iters}iters_{_ts}.json'
        _save_payload = {
            'group':      group_key,
            'dir_name':   _dir_name,
            'year':       year,
            'num_iters':  num_iters,
            'ff_pairings': ff_pairings_str,
            'computed':   _ts,
            'results':    results,
        }
        with open(_save_dir / _save_name, 'w', encoding='utf-8') as _fh:
            json.dump(_save_payload, _fh, indent=2)

    except Exception as exc:
        job.queue.put(f'[ERROR] {exc}')
        job.queue.put(f'[TRACEBACK] {_tb.format_exc()}')
        job.status = 'error'
    finally:
        job.queue.put(None)   # always send sentinel


@app.route('/group_analysis')
def group_analysis_route():
    group = request.args.get('group', '')
    saved = scan_saved_models()
    simulations = scan_simulations(THIS_YEAR)
    # Build a quick-lookup map of existing sim results keyed by dir_name.
    sim_map = {s['dir_name']: s for s in simulations}
    groups_list = []
    if BRACKETS_DIR.is_dir():
        groups_list = [d.name for d in sorted(BRACKETS_DIR.iterdir()) if d.is_dir()]
    current_ff_str = _load_ff_pairings_str(THIS_YEAR)
    current_ff_label = _ff_pairings_label(current_ff_str)
    # Load round-1 matchups and locked results for live tournament progress UI
    r1_matchups = _build_display_games(_read_round1_teams(THIS_YEAR))
    locked_results_raw = _load_locked_results(THIS_YEAR)
    # Convert to string-keyed for JSON serialisation
    locked_results_json = {str(k): v for k, v in locked_results_raw.items()}
    return render_template_string(
        GROUP_ANALYSIS_HTML,
        group=group,
        groups_list=groups_list,
        saved_models=saved,
        sim_map=sim_map,
        current_ff_str=current_ff_str,
        current_ff_label=current_ff_label,
        year=THIS_YEAR,
        r1_matchups=r1_matchups,
        locked_results=locked_results_json,
    )


@app.route('/run_group_analysis', methods=['POST'])
def run_group_analysis():
    data        = request.get_json(force=True)
    group_key   = (data.get('group')       or '').strip()
    dir_name    = (data.get('dir_name')    or '').strip()
    num_iters   = int(data.get('num_iters', 1000))
    rng_seed    = data.get('seed')
    year        = int(data.get('year', THIS_YEAR))

    if not group_key:
        return jsonify({'error': 'Group name required.'}), 400
    if not dir_name:
        return jsonify({'error': 'Model required.'}), 400

    pkl_path = PREDICTIONS_DIR / dir_name / 'model.pkl'
    if not pkl_path.exists():
        return jsonify({'error': f'model.pkl not found in {dir_name}.'}), 400

    job_id = str(uuid.uuid4())[:8]
    j = Job()
    j.results = None
    group_jobs[job_id] = j

    t = threading.Thread(
        target=_run_group_scoring,
        args=(
            job_id, str(pkl_path), group_key, year, num_iters,
            int(rng_seed) if rng_seed is not None and str(rng_seed).strip() != '' else None,
        ),
        daemon=True,
    )
    t.start()
    return jsonify({'job_id': job_id})


@app.route('/group_analysis_stream/<job_id>')
def group_analysis_stream(job_id):
    if job_id not in group_jobs:
        abort(404)

    def generate():
        job = group_jobs[job_id]
        while True:
            try:
                line = job.queue.get(timeout=600)
            except queue.Empty:
                yield 'data: {"type":"timeout"}\n\n'
                break
            if line is None:
                payload = {
                    'type':    'done',
                    'status':  job.status,
                    'results': job.results,
                    'sim_url': getattr(job, 'sim_url', None),
                }
                yield f'data: {json.dumps(payload)}\n\n'
                break
            yield f'data: {json.dumps({"type": "line", "text": line})}\n\n'

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/locked_results')
def api_get_locked_results():
    """Return locked game results for the current year as {round_str: [winner_or_null, ...]}."""
    lr = _load_locked_results(THIS_YEAR)
    return jsonify({str(k): v for k, v in lr.items()})


@app.route('/api/locked_results', methods=['POST'])
def api_save_locked_results():
    """Save locked game results for the current year."""
    data = request.get_json(force=True)
    round_winners = data.get('round_winners', {})
    p = _locked_results_path(THIS_YEAR)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({'round_winners': round_winners}, indent=2, ensure_ascii=False),
                 encoding='utf-8')
    return jsonify({'ok': True})


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

BRACKET_INPUT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ year }} Bracket Setup</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Segoe UI", Arial, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  min-height: 100vh;
  padding: 28px 24px 60px;
}
a.back-link {
  display: inline-flex; align-items: center; gap: 6px;
  color: #64748b; font-size: 12px; text-decoration: none;
  margin-bottom: 18px; transition: color .15s;
}
a.back-link:hover { color: #93c5fd; }
h1 { font-size: 20px; color: #fbbf24; margin-bottom: 4px; }
.subtitle { color: #64748b; font-size: 13px; margin-bottom: 24px; }

.regions-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
  max-width: 1060px;
}
@media (max-width: 760px) { .regions-grid { grid-template-columns: 1fr; } }

.region-card {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 18px 20px;
}
.region-title {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .8px; color: #94a3b8; margin-bottom: 14px;
}

.matchup-row {
  display: grid;
  grid-template-columns: 28px 1fr 18px 28px 1fr;
  align-items: center;
  gap: 5px;
  padding: 6px 0;
  border-bottom: 1px solid #0f172a;
}
.matchup-row:last-child { border-bottom: none; }

.seed-badge {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #fbbf24;
  font-size: 11px;
  font-weight: 700;
  text-align: center;
  padding: 4px 3px;
  user-select: none;
}
.team-input {
  width: 100%;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 5px;
  color: #e2e8f0;
  padding: 5px 8px;
  font-size: 12px;
  outline: none;
  transition: border-color .15s;
}
.team-input:focus { border-color: #3b82f6; }
.team-input.empty { border-color: #7f1d1d; }
.vs-sep {
  font-size: 9px; color: #475569;
  font-weight: 700; text-align: center;
}

.action-bar {
  max-width: 1060px; margin-top: 22px;
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
}
button.save-btn {
  background: #1d4ed8; color: #fff; border: none; border-radius: 7px;
  padding: 10px 28px; font-size: 14px; font-weight: 600; cursor: pointer;
  transition: background .15s;
}
button.save-btn:hover { background: #2563eb; }
.msg-saved { color: #4ade80; font-size: 13px; }
.msg-error { color: #f87171; font-size: 13px; }

.ff-pairing-card {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 18px 20px;
  max-width: 1060px;
  margin-top: 20px;
}
.ff-pairing-title {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .8px; color: #94a3b8; margin-bottom: 6px;
}
.ff-pairing-sub {
  color: #475569; font-size: 12px; margin-bottom: 12px;
}
.ff-radio-group { display: flex; flex-wrap: wrap; gap: 16px; }
.ff-radio {
  display: flex; align-items: center; gap: 6px;
  color: #e2e8f0; font-size: 13px; cursor: pointer;
}
.ff-radio input[type=radio] { accent-color: #3b82f6; cursor: pointer; }
.ff-radio:has(input:checked) { color: #fbbf24; }
</style>
</head>
<body>

<a href="/" class="back-link">&#8592; Back to Predictor</a>
<h1>&#127942; {{ year }} Bracket Setup</h1>
<p class="subtitle">
  Select the team for each seed slot. Autocomplete is sourced from {{ year }} KenPom data.
  Saving writes <code>Round1_{{ year }}.csv</code> and <code>FFPairings_{{ year }}.json</code> to <code>Data/BracketData/{{ year }}/</code>.
</p>

<datalist id="kp-teams">
  {% for t in teams %}<option value="{{ t }}">{% endfor %}
</datalist>

<form method="post" action="/save_bracket" autocomplete="off" id="bracket-form">
<div class="regions-grid">
  {% for r in range(4) %}
  <div class="region-card">
    <div class="region-title">{{ region_names[r] }}</div>
    {% for g in range(8) %}
    {% set i = r*8 + g + 1 %}
    {% set gd = games[i-1] %}
    <div class="matchup-row">
      <span class="seed-badge">{{ gd.seed1 }}</span>
      <input class="team-input{% if not gd.team1 %} empty{% endif %}"
             list="kp-teams" name="game_{{ i }}_team1"
             placeholder="Team (seed {{ gd.seed1 }})"
             value="{{ gd.team1 }}"
             oninput="this.classList.toggle('empty', !this.value.trim())">
      <span class="vs-sep">vs</span>
      <span class="seed-badge">{{ gd.seed2 }}</span>
      <input class="team-input{% if not gd.team2 %} empty{% endif %}"
             list="kp-teams" name="game_{{ i }}_team2"
             placeholder="Team (seed {{ gd.seed2 }})"
             value="{{ gd.team2 }}"
             oninput="this.classList.toggle('empty', !this.value.trim())">
    </div>
    {% endfor %}
  </div>
  {% endfor %}
</div>

<div class="ff-pairing-card">
  <div class="ff-pairing-title">&#127952; Final Four Pairings</div>
  <p class="ff-pairing-sub">Which regions face each other in the Final Four?</p>
  <div class="ff-radio-group">
    {% for val, pairs, lbl in ff_pairing_options %}
    <label class="ff-radio">
      <input type="radio" name="ff_pairings" value="{{ val }}"
        {% if val == ff_pairings_str %}checked{% endif %}>
      {{ lbl }}
    </label>
    {% endfor %}
  </div>
</div>

<div class="action-bar">
  <button type="submit" class="save-btn">&#128190; Save Bracket &amp; Pairings</button>
  {% if saved %}
  <span class="msg-saved">&#10003; Saved Round1_{{ year }}.csv and FFPairings_{{ year }}.json</span>
  {% endif %}
  {% if error %}
  <span class="msg-error">&#9888; {{ error }}</span>
  {% endif %}
</div>
</form>

</body>
</html>
"""

# ---------------------------------------------------------------------------
# Fill-My-Bracket HTML template
# ---------------------------------------------------------------------------

FILL_BRACKET_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ year }} Bracket Picker</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Segoe UI", Arial, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  min-height: 100vh;
  padding: 22px 20px 60px;
}
a.back-link {
  display: inline-flex; align-items: center; gap: 6px;
  color: #64748b; font-size: 12px; text-decoration: none;
  margin-bottom: 16px; transition: color .15s;
}
a.back-link:hover { color: #93c5fd; }
.page-title { font-size: 20px; color: #fbbf24; margin-bottom: 3px; }
.page-sub   { color: #64748b; font-size: 12px; margin-bottom: 20px; }

/* ----- meta form ----- */
.meta-bar {
  display: flex; align-items: flex-end; gap: 12px; flex-wrap: wrap;
  background: #1e293b; border: 1px solid #334155; border-radius: 10px;
  padding: 14px 18px; margin-bottom: 18px; max-width: 900px;
}
.meta-bar .field { display: flex; flex-direction: column; gap: 4px; flex: 1; min-width: 160px; }
.meta-bar label  { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; color: #94a3b8; }
.meta-bar input  {
  background: #0f172a; border: 1px solid #334155; border-radius: 6px;
  color: #e2e8f0; padding: 7px 10px; font-size: 13px; outline: none;
}
.meta-bar input:focus { border-color: #3b82f6; }
#save-btn {
  background: #ca8a04; color: #fefce8; border: none; border-radius: 7px;
  padding: 9px 22px; font-size: 13px; font-weight: 600; cursor: pointer;
  transition: background .15s; white-space: nowrap;
}
#save-btn:hover:not(:disabled) { background: #eab308; }
#save-btn:disabled { opacity: .5; cursor: not-allowed; }
#save-msg { font-size: 12px; white-space: nowrap; }

/* ----- progress ----- */
.progress-bar {
  max-width: 900px; margin-bottom: 18px;
  display: flex; align-items: center; gap: 10px;
}
.prog-track { flex: 1; background: #1e293b; border-radius: 4px; height: 8px; overflow: hidden; border: 1px solid #334155; }
#prog-fill   { height: 100%; background: linear-gradient(90deg, #ca8a04, #fbbf24); border-radius: 4px; width: 0%; transition: width .2s ease; }
#prog-text   { font-size: 11px; color: #94a3b8; font-family: monospace; white-space: nowrap; }

/* ----- bracket layout ----- */
.bracket-outer { max-width: 100%; overflow-x: auto; }
.region-section { margin-bottom: 22px; }
.region-heading {
  font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px;
  color: #94a3b8; margin-bottom: 6px;
}
.rounds-row {
  display: flex; gap: 0; align-items: stretch;
  border: 1px solid #334155; border-radius: 8px; overflow: hidden;
}

/* ----- round columns ----- */
:root { --u: 64px; }
.round-col { display: flex; flex-direction: column; min-width: 160px; }
.round-col + .round-col { border-left: 1px solid #1a2744; }
.round-hdr {
  font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .7px;
  color: #475569; background: #1a2744; padding: 5px 10px; text-align: center;
  border-bottom: 1px solid #1a2744; white-space: nowrap;
}

/* ----- game cards ----- */
.game-card {
  display: flex; flex-direction: column; justify-content: center;
  gap: 2px; padding: 4px 6px; background: #1e293b; position: relative;
}
.game-card + .game-card { border-top: 1px solid #0f172a; }

.r1-card  { height: var(--u); }
.r2-card  { height: calc(var(--u) * 2); }
.s16-card { height: calc(var(--u) * 4); }
.e8-card  { height: calc(var(--u) * 8); flex: 1; }

/* ----- team buttons ----- */
.team-btn {
  display: flex; align-items: center; gap: 6px;
  width: 100%; background: #0f172a; border: 1px solid #334155; border-radius: 4px;
  color: #cbd5e1; padding: 5px 8px; font-size: 11px; text-align: left;
  cursor: pointer; transition: border-color .12s, color .12s, background .12s;
  overflow: hidden; white-space: nowrap;
}
.team-btn:hover:not(:disabled):not(.tbd) { border-color: #3b82f6; color: #93c5fd; background: #1e3a5f; }
.team-btn.winner { background: #322f1a; border-color: #ca8a04; color: #fbbf24; font-weight: 600; }
.team-btn.tbd    { color: #334155; border-color: #1e293b; cursor: default; }
.team-btn:disabled { cursor: default; }
.team-btn .s {
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 18px; height: 18px; border-radius: 3px;
  background: #0f172a; border: 1px solid #334155;
  color: #64748b; font-size: 10px; font-weight: 700; flex-shrink: 0;
}
.team-btn.winner .s { background: #1a1500; border-color: #78540a; color: #fbbf24; }

/* ----- final four ----- */
.ff-section {
  background: #1e293b; border: 1px solid #334155; border-radius: 10px;
  padding: 18px 20px; max-width: 900px; margin-bottom: 18px;
}
.ff-title {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px;
  color: #94a3b8; margin-bottom: 14px;
}
.ff-games {
  display: grid; grid-template-columns: 1fr 200px 1fr; gap: 16px; align-items: center;
}
.ff-matchup-label {
  font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .6px;
  color: #475569; margin-bottom: 6px;
}
.ff-card, .champ-card {
  display: flex; flex-direction: column; gap: 2px;
  background: #0f172a; border-radius: 6px; padding: 6px;
}
.champ-col { text-align: center; }
.champ-col .ff-matchup-label { color: #ca8a04; justify-content: center; display: flex; }
.champ-display {
  margin-top: 10px; padding: 8px 12px; background: #1a1500;
  border: 1px solid #ca8a04; border-radius: 6px; text-align: center;
  font-size: 13px; font-weight: 700; color: #fbbf24;
}

/* ----- links bar ----- */
.links-bar { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.nav-link {
  display: inline-flex; align-items: center; gap: 6px;
  background: #1e293b; border: 1px solid #334155; border-radius: 7px;
  padding: 6px 14px; font-size: 12px; color: #93c5fd; text-decoration: none;
  transition: border-color .15s;
}
.nav-link:hover { border-color: #3b82f6; }

/* ----- ff pairing radios ----- */
.ff-pairing-display { font-size: 13px; color: #fbbf24; font-weight: 600; margin-top: 3px; }
.ff-radio-group { display: flex; flex-direction: column; gap: 4px; margin-top: 1px; }
.ff-radio {
  display: flex; align-items: center; gap: 5px;
  font-size: 11px; color: #94a3b8; cursor: pointer; white-space: nowrap;
}
.ff-radio input[type=radio] { accent-color: #ca8a04; cursor: pointer; }
.ff-radio:has(input:checked) { color: #fbbf24; }
</style>
</head>
<body>

<a href="/" class="back-link">&#8592; Back to Predictor</a>
<div class="links-bar">
  <a href="/my_brackets" class="nav-link">&#128196; Browse Saved Brackets</a>
</div>
<h1 class="page-title">&#10003; {{ year }} Bracket Picker</h1>
<p class="page-sub">Click a team name to advance them. When complete, give your bracket a name and group, then save.</p>

<!-- Meta bar -->
<div class="meta-bar" id="meta-form">
  <div class="field">
    <label for="bracket-name">Bracket Name</label>
    <input type="text" id="bracket-name" placeholder="e.g. My Bracket">
  </div>
  <div class="field">
    <label for="bracket-group">Group</label>
    <input type="text" id="bracket-group" placeholder="e.g. Friends">
  </div>
  <div class="field" style="min-width:140px">
    <label>Final Four Pairings</label>
    <div class="ff-pairing-display">{{ ff_pairings_label }}</div>
  </div>
  <button id="save-btn" onclick="saveBracket()">&#128190; Save Bracket</button>
  <span id="save-msg"></span>
</div>

<!-- Progress -->
<div class="progress-bar">
  <span id="prog-text">0 / 63 picks</span>
  <div class="prog-track"><div id="prog-fill"></div></div>
</div>

<!-- Bracket -->
<div class="bracket-outer">

  {% for r in range(4) %}
  <div class="region-section">
    <div class="region-heading">{{ region_names[r] }}</div>
    <div class="rounds-row">
      <!-- Round 1 -->
      <div class="round-col">
        <div class="round-hdr">Round 1</div>
        {% for g in range(8) %}
        {% set gi = r*8 + g %}
        <div class="game-card r1-card">
          <button class="team-btn" id="btn-r1-{{ gi }}-0" onclick="makePick('r1',{{ gi }},0)"></button>
          <button class="team-btn" id="btn-r1-{{ gi }}-1" onclick="makePick('r1',{{ gi }},1)"></button>
        </div>
        {% endfor %}
      </div>
      <!-- Round 2 -->
      <div class="round-col">
        <div class="round-hdr">Round 2</div>
        {% for g in range(4) %}
        {% set gi = r*4 + g %}
        <div class="game-card r2-card">
          <button class="team-btn tbd" id="btn-r2-{{ gi }}-0" onclick="makePick('r2',{{ gi }},0)" disabled>TBD</button>
          <button class="team-btn tbd" id="btn-r2-{{ gi }}-1" onclick="makePick('r2',{{ gi }},1)" disabled>TBD</button>
        </div>
        {% endfor %}
      </div>
      <!-- Sweet 16 -->
      <div class="round-col">
        <div class="round-hdr">Sweet 16</div>
        {% for g in range(2) %}
        {% set gi = r*2 + g %}
        <div class="game-card s16-card">
          <button class="team-btn tbd" id="btn-s16-{{ gi }}-0" onclick="makePick('s16',{{ gi }},0)" disabled>TBD</button>
          <button class="team-btn tbd" id="btn-s16-{{ gi }}-1" onclick="makePick('s16',{{ gi }},1)" disabled>TBD</button>
        </div>
        {% endfor %}
      </div>
      <!-- Elite 8 -->
      <div class="round-col">
        <div class="round-hdr">Elite 8</div>
        <div class="game-card e8-card">
          <button class="team-btn tbd" id="btn-e8-{{ r }}-0" onclick="makePick('e8',{{ r }},0)" disabled>TBD</button>
          <button class="team-btn tbd" id="btn-e8-{{ r }}-1" onclick="makePick('e8',{{ r }},1)" disabled>TBD</button>
        </div>
      </div>
    </div>
  </div>
  {% endfor %}

  <!-- Final Four + Championship -->
  <div class="ff-section">
    <div class="ff-title">&#127952; Final Four &amp; Championship</div>
    <div class="ff-games">
      <!-- Semi 0 -->
      <div>
        <div class="ff-matchup-label" id="ff-label-0">South vs West</div>
        <div class="ff-card">
          <button class="team-btn tbd" id="btn-semi-0-0" onclick="makePick('semi',0,0)" disabled>TBD</button>
          <button class="team-btn tbd" id="btn-semi-0-1" onclick="makePick('semi',0,1)" disabled>TBD</button>
        </div>
      </div>
      <!-- Championship -->
      <div class="champ-col">
        <div class="ff-matchup-label">&#127942; Championship</div>
        <div class="champ-card">
          <button class="team-btn tbd" id="btn-champ-0-0" onclick="makePick('champ',0,0)" disabled>TBD</button>
          <button class="team-btn tbd" id="btn-champ-0-1" onclick="makePick('champ',0,1)" disabled>TBD</button>
        </div>
        <div id="champion-display" class="champ-display" style="display:none"></div>
      </div>
      <!-- Semi 1 -->
      <div>
        <div class="ff-matchup-label" id="ff-label-1">East vs Midwest</div>
        <div class="ff-card">
          <button class="team-btn tbd" id="btn-semi-1-0" onclick="makePick('semi',1,0)" disabled>TBD</button>
          <button class="team-btn tbd" id="btn-semi-1-1" onclick="makePick('semi',1,1)" disabled>TBD</button>
        </div>
      </div>
    </div>
  </div>

</div><!-- bracket-outer -->

<script>
const R1          = {{ matchups_json | safe }};
const REGIONS     = {{ region_names_json | safe }};
const FF_PAIRING_MAP = {
  '0-3,1-2': [[0, 3], [1, 2]],
  '0-1,2-3': [[0, 1], [2, 3]],
  '0-2,1-3': [[0, 2], [1, 3]],
};
let FF_PAIRINGS = {{ ff_pairings_json | safe }};

// ---- state ----
const state = {
  r1:   new Array(32).fill(null),
  r2:   new Array(16).fill(null),
  s16:  new Array(8).fill(null),
  e8:   new Array(4).fill(null),
  semi: new Array(2).fill(null),
  champ: null,
};

// ---- seed lookup ----
const seedOf = {};
R1.forEach(m => { seedOf[m.team1] = m.seed1; seedOf[m.team2] = m.seed2; });

// ---- FF pairing change ----
function updateFFPairings(pairingStr) {
  FF_PAIRINGS = FF_PAIRING_MAP[pairingStr] || [[0, 3], [1, 2]];
  state.semi[0] = null;
  state.semi[1] = null;
  state.champ = null;
  renderAll();
  updateFFLabels();
}

function updateFFLabels() {
  const regionOf = i => REGIONS[i] || String(i);
  const l0 = document.getElementById('ff-label-0');
  const l1 = document.getElementById('ff-label-1');
  if (l0) l0.textContent = regionOf(FF_PAIRINGS[0][0]) + ' vs ' + regionOf(FF_PAIRINGS[0][1]);
  if (l1) l1.textContent = regionOf(FF_PAIRINGS[1][0]) + ' vs ' + regionOf(FF_PAIRINGS[1][1]);
}

// ---- routing helpers ----
function feedsInto(level, gameIdx) {
  if (level === 'r1')  return { level: 'r2',   gameIdx: Math.floor(gameIdx / 2) };
  if (level === 'r2')  return { level: 's16',  gameIdx: Math.floor(gameIdx / 2) };
  if (level === 's16') return { level: 'e8',   gameIdx: Math.floor(gameIdx / 2) };
  if (level === 'e8') {
    for (let s = 0; s < FF_PAIRINGS.length; s++) {
      if (FF_PAIRINGS[s].includes(gameIdx)) return { level: 'semi', gameIdx: s };
    }
    return null;
  }
  if (level === 'semi') return { level: 'champ', gameIdx: 0 };
  return null;
}

function clearDownstream(level, gameIdx) {
  const next = feedsInto(level, gameIdx);
  if (!next) return;
  if (next.level === 'champ') { state.champ = null; }
  else { state[next.level][next.gameIdx] = null; }
  clearDownstream(next.level, next.gameIdx);
}

// ---- pick ----
function makePick(level, gameIdx, slot) {
  let teamName;
  if (level === 'r1') {
    const m = R1[gameIdx];
    teamName = slot === 0 ? m.team1 : m.team2;
  } else {
    const btn = document.getElementById('btn-' + level + '-' + gameIdx + '-' + slot);
    if (!btn || btn.disabled || !btn.dataset.team) return;
    teamName = btn.dataset.team;
  }
  const prev = level === 'champ' ? state.champ : state[level][gameIdx];
  if (prev === teamName) return;
  if (level === 'champ') state.champ = teamName;
  else state[level][gameIdx] = teamName;
  clearDownstream(level, gameIdx);
  renderAll();
}

// ---- which team occupies a slot in a game ----
function getTeamForSlot(level, gameIdx, slot) {
  if (level === 'r1') {
    const m = R1[gameIdx];
    return slot === 0 ? { team: m.team1, seed: m.seed1 }
                      : { team: m.team2, seed: m.seed2 };
  }
  let prevLevel, prevGameIdx;
  switch (level) {
    case 'r2':   prevLevel = 'r1';   prevGameIdx = gameIdx * 2 + slot; break;
    case 's16':  prevLevel = 'r2';   prevGameIdx = gameIdx * 2 + slot; break;
    case 'e8':   prevLevel = 's16';  prevGameIdx = gameIdx * 2 + slot; break;
    case 'semi': prevLevel = 'e8';   prevGameIdx = FF_PAIRINGS[gameIdx][slot]; break;
    case 'champ':prevLevel = 'semi'; prevGameIdx = slot; break;
    default: return { team: null, seed: null };
  }
  const winner = state[prevLevel][prevGameIdx];
  return { team: winner || null, seed: winner ? (seedOf[winner] || '') : null };
}

// ---- render ----
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderBtn(btn, team, seed, winner) {
  if (team) {
    btn.disabled = false;
    btn.classList.remove('tbd');
    btn.dataset.team = team;
    btn.innerHTML = '<span class="s">' + escHtml(seed || '') + '</span> ' + escHtml(team);
    btn.classList.toggle('winner', winner === team);
  } else {
    btn.disabled = true;
    btn.classList.add('tbd');
    btn.classList.remove('winner');
    btn.dataset.team = '';
    btn.innerHTML = 'TBD';
  }
}

function renderGame(level, gameIdx) {
  const winner = level === 'champ' ? state.champ : state[level][gameIdx];
  for (let slot = 0; slot < 2; slot++) {
    const btn = document.getElementById('btn-' + level + '-' + gameIdx + '-' + slot);
    if (!btn) continue;
    const { team, seed } = getTeamForSlot(level, gameIdx, slot);
    renderBtn(btn, team, seed, winner);
  }
}

function renderAll() {
  // R1 — just toggle winner class
  for (let i = 0; i < 32; i++) {
    const winner = state.r1[i];
    for (let s = 0; s < 2; s++) {
      const btn = document.getElementById('btn-r1-' + i + '-' + s);
      if (!btn) continue;
      const m = R1[i];
      const team = s === 0 ? m.team1 : m.team2;
      btn.classList.toggle('winner', winner === team);
    }
  }
  // All other levels
  const levelsAndCounts = [['r2',16],['s16',8],['e8',4],['semi',2],['champ',1]];
  for (const [lv, cnt] of levelsAndCounts) {
    for (let g = 0; g < cnt; g++) renderGame(lv, g);
  }
  // Champion display
  const cd = document.getElementById('champion-display');
  if (state.champ) {
    cd.textContent = '\u{1F3C6} ' + state.champ;
    cd.style.display = '';
  } else {
    cd.style.display = 'none';
  }
  updateProgress();
}

function updateProgress() {
  let done = 0;
  state.r1.forEach(v => v && done++);
  state.r2.forEach(v => v && done++);
  state.s16.forEach(v => v && done++);
  state.e8.forEach(v => v && done++);
  state.semi.forEach(v => v && done++);
  if (state.champ) done++;
  const pct = Math.round(done / 63 * 100);
  document.getElementById('prog-text').textContent = done + ' / 63 picks';
  document.getElementById('prog-fill').style.width = pct + '%';
}

// ---- save ----
function saveBracket() {
  const name  = document.getElementById('bracket-name').value.trim();
  const group = document.getElementById('bracket-group').value.trim();
  if (!name)  { alert('Enter a bracket name.'); return; }
  if (!group) { alert('Enter a group name.'); return; }
  if (!state.champ && !confirm('Your bracket is not complete. Save anyway?')) return;

  const btn = document.getElementById('save-btn');
  const msg = document.getElementById('save-msg');
  btn.disabled = true;
  msg.style.color = '#94a3b8';
  msg.textContent = 'Saving\u2026';

  fetch('/save_my_bracket', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name, group,
      picks: {
        r1:       [...state.r1],
        r2:       [...state.r2],
        s16:      [...state.s16],
        e8:       [...state.e8],
        semi:     [...state.semi],
        champion: state.champ,
      },
    }),
  })
  .then(r => r.json())
  .then(data => {
    btn.disabled = false;
    if (data.error) {
      msg.style.color = '#f87171';
      msg.textContent = '\u26a0 ' + data.error;
    } else {
      msg.style.color = '#4ade80';
      msg.textContent = '\u2713 Saved to ' + data.path;
    }
  })
  .catch(() => {
    btn.disabled = false;
    msg.style.color = '#f87171';
    msg.textContent = '\u26a0 Failed to save.';
  });
}

// ---- load saved picks (called from view bracket page) ----
function loadPicks(picks, ffPairings) {
  if (!picks) return;
  ['r1','r2','s16','e8'].forEach(lv => {
    (picks[lv] || []).forEach((v, i) => { if (i < state[lv].length) state[lv][i] = v; });
  });
  (picks.semi || []).forEach((v, i) => { if (i < 2) state.semi[i] = v; });
  state.champ = picks.champion || null;
  updateFFLabels();
}

// ---- initialise R1 buttons with team names from JSON ----
(function init() {
  R1.forEach((m, i) => {
    const b0 = document.getElementById('btn-r1-' + i + '-0');
    const b1 = document.getElementById('btn-r1-' + i + '-1');
    if (b0) { b0.innerHTML = '<span class="s">' + m.seed1 + '</span> ' + escHtml(m.team1); b0.dataset.team = m.team1; }
    if (b1) { b1.innerHTML = '<span class="s">' + m.seed2 + '</span> ' + escHtml(m.team2); b1.dataset.team = m.team2; }
  });

  // Pre-load if ?load=<group>/<file> param present
  const params = new URLSearchParams(window.location.search);
  const loadPath = params.get('load');
  if (loadPath) {
    const bname = params.get('bname') || '';
    const bgrp  = params.get('bgrp')  || '';
    if (bname) document.getElementById('bracket-name').value = bname;
    if (bgrp)  document.getElementById('bracket-group').value = bgrp;

    fetch('/api/bracket/' + encodeURIComponent(loadPath))
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (!data) return;
        document.getElementById('bracket-name').value  = data.name  || '';
        document.getElementById('bracket-group').value = data.group || '';
        loadPicks(data.picks, data.ff_pairings);
        renderAll();
      });
  }

  updateProgress();
  updateFFLabels();
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# My-Brackets HTML template
# ---------------------------------------------------------------------------

MY_BRACKETS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>My Brackets</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Segoe UI", Arial, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  min-height: 100vh;
  padding: 26px 24px 60px;
}
a.back-link {
  display: inline-flex; align-items: center; gap: 6px;
  color: #64748b; font-size: 12px; text-decoration: none;
  margin-bottom: 18px; transition: color .15s;
}
a.back-link:hover { color: #93c5fd; }
h1 { font-size: 20px; color: #fbbf24; margin-bottom: 4px; }
.subtitle { color: #64748b; font-size: 13px; margin-bottom: 22px; }
.new-link {
  display: inline-flex; align-items: center; gap: 6px;
  background: #ca8a04; color: #1a1000; border-radius: 7px;
  padding: 8px 18px; font-size: 13px; font-weight: 700;
  text-decoration: none; margin-bottom: 24px; transition: background .15s;
}
.new-link:hover { background: #eab308; }

.group-section { margin-bottom: 28px; }
.group-name {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px;
  color: #94a3b8; margin-bottom: 8px;
  border-bottom: 1px solid #334155; padding-bottom: 4px;
}
.bracket-grid { display: flex; flex-wrap: wrap; gap: 12px; }
.bracket-card {
  background: #1e293b; border: 1px solid #334155; border-radius: 8px;
  padding: 14px 16px; min-width: 200px; max-width: 260px;
  text-decoration: none; color: inherit; transition: border-color .15s;
  display: block;
}
.bracket-card:hover { border-color: #3b82f6; }
.b-name { font-size: 14px; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
.b-champ {
  font-size: 12px; color: #fbbf24; margin-bottom: 6px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.b-meta { font-size: 10px; color: #475569; }

.empty { color: #475569; font-size: 13px; }
</style>
</head>
<body>

<a href="/" class="back-link">&#8592; Back to Predictor</a>
<h1>&#128196; My Brackets</h1>
<p class="subtitle">All saved brackets, grouped by group name. Click a bracket to view or edit it.</p>
<a href="/fill_bracket" class="new-link">&#43; New Bracket</a>

{% if not groups %}
<p class="empty">No saved brackets yet. <a href="/fill_bracket" style="color:#93c5fd">Fill out your first bracket!</a></p>
{% endif %}

{% for group_key, brackets in groups.items() %}
<div class="group-section">
  <div class="group-name" style="display:flex;justify-content:space-between;align-items:center">
    <span>{{ brackets[0].group }}</span>
    <a href="/group_analysis?group={{ group_key }}"
       style="font-size:10px;font-weight:500;text-transform:none;letter-spacing:0;
              color:#93c5fd;text-decoration:none;background:#0f172a;border:1px solid #334155;
              border-radius:5px;padding:3px 10px;transition:border-color .15s"
       onmouseover="this.style.borderColor='#3b82f6'"
       onmouseout="this.style.borderColor='#334155'">&#128202; Analyze Group</a>
  </div>
  <div class="bracket-grid">
    {% for b in brackets %}
    <a class="bracket-card"
       href="/fill_bracket?load={{ group_key }}/{{ b.file }}&bname={{ b.name | urlencode }}&bgrp={{ b.group | urlencode }}">
      <div class="b-name">{{ b.name }}</div>
      {% if b.champion %}
      <div class="b-champ">&#127942; {{ b.champion }}</div>
      {% else %}
      <div class="b-champ" style="color:#475569">&#8212; Incomplete</div>
      {% endif %}
      <div class="b-meta">{{ b.year }} &middot; {{ b.created[:10] if b.created else '' }}</div>
    </a>
    {% endfor %}
  </div>
</div>
{% endfor %}

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Group Analysis HTML template
# ---------------------------------------------------------------------------

GROUP_ANALYSIS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Group Analysis</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Segoe UI", Arial, sans-serif;
  background: #0f172a; color: #e2e8f0;
  min-height: 100vh; padding: 26px 24px 60px;
}
a.back-link {
  display: inline-flex; align-items: center; gap: 6px;
  color: #64748b; font-size: 12px; text-decoration: none;
  margin-bottom: 18px; transition: color .15s;
}
a.back-link:hover { color: #93c5fd; }
h1 { font-size: 20px; color: #fbbf24; margin-bottom: 4px; }
.subtitle { color: #64748b; font-size: 12px; margin-bottom: 22px; }

/* Config card */
.config-card {
  background: #1e293b; border: 1px solid #334155; border-radius: 10px;
  padding: 18px 20px; max-width: 820px; margin-bottom: 20px;
}
.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px 16px; align-items: end;
}
.field { display: flex; flex-direction: column; gap: 4px; }
.field label {
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .6px; color: #94a3b8;
}
.field select, .field input {
  background: #0f172a; border: 1px solid #334155; border-radius: 6px;
  color: #e2e8f0; padding: 7px 9px; font-size: 12px; outline: none;
}
.field select:focus, .field input:focus { border-color: #3b82f6; }
#run-btn {
  background: #ca8a04; color: #1a1000; border: none; border-radius: 7px;
  padding: 8px 22px; font-size: 13px; font-weight: 700; cursor: pointer;
  transition: background .15s; white-space: nowrap; align-self: end;
}
#run-btn:hover:not(:disabled) { background: #eab308; }
#run-btn:disabled { opacity: .5; cursor: not-allowed; }

/* Progress */
.prog-wrap { max-width: 820px; margin-bottom: 16px; }
.prog-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.prog-label { font-size: 10px; color: #64748b; }
#prog-text  { font-size: 11px; color: #93c5fd; font-family: monospace; }
.prog-track { background: #1e293b; border-radius: 4px; overflow: hidden; height: 8px; border: 1px solid #334155; }
#prog-bar   { height: 100%; background: linear-gradient(90deg, #ca8a04, #fbbf24); border-radius: 4px; width: 0%; transition: width .2s ease; }

/* Status badge */
.status-badge {
  display: inline-block; border-radius: 12px;
  padding: 2px 10px; font-size: 11px; font-weight: 600;
}
.status-running { background: #1e3a5f; color: #93c5fd; }
.status-done    { background: #14432a; color: #86efac; }
.status-error   { background: #450a0a; color: #fca5a5; }

/* Error box */
#error-box {
  max-width: 820px; background: #450a0a; border-radius: 8px;
  padding: 10px 14px; font-size: 12px; color: #fca5a5;
  margin-bottom: 14px; display: none; white-space: pre-wrap; font-family: monospace;
}

/* FF pairings info bar */
.ff-info-bar {
  max-width: 820px; margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.ff-info-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; color: #64748b; }
.ff-info-val {
  font-size: 12px; font-weight: 600; color: #fbbf24;
  background: #1a1500; border: 1px solid #78540a; border-radius: 5px; padding: 2px 10px;
}
.ff-info-link { font-size: 11px; color: #64748b; text-decoration: none; }
.ff-info-link:hover { color: #93c5fd; }

/* Simulations section */
.sims-card {
  background: #1e293b; border: 1px solid #334155; border-radius: 10px;
  padding: 16px 20px; max-width: 100%; margin-bottom: 18px; overflow-x: auto;
}
.sims-title {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .7px;
  color: #94a3b8; margin-bottom: 10px;
}
table.sims-table { width: 100%; border-collapse: collapse; font-size: 12px; }
table.sims-table th {
  text-align: left; padding: 5px 8px; color: #64748b; font-size: 10px;
  text-transform: uppercase; letter-spacing: .6px; border-bottom: 1px solid #334155; white-space: nowrap;
}
table.sims-table th.sortable { cursor: pointer; user-select: none; }
table.sims-table th.sortable:hover { color: #93c5fd; }
table.sims-table th.sort-asc::after  { content: ' \25b2'; font-size: 9px; }
table.sims-table th.sort-desc::after { content: ' \25bc'; font-size: 9px; }
table.sims-table td {
  padding: 7px 8px; border-bottom: 1px solid #1a2744; color: #cbd5e1; vertical-align: middle;
}
table.sims-table tr:last-child td { border-bottom: none; }
table.sims-table tbody tr:hover td { background: #172554; }
.quick-run-btn {
  background: #1e3a5f; color: #93c5fd; border: 1px solid #2563eb; border-radius: 5px;
  padding: 3px 10px; font-size: 11px; font-weight: 600; cursor: pointer;
  transition: background .12s; white-space: nowrap;
}
.quick-run-btn:hover { background: #1d4ed8; color: #fff; }
.quick-run-btn:disabled { opacity: .5; cursor: not-allowed; }
.no-pkl-note { font-size: 10px; color: #94a3b8; }

/* Live Tournament Results – Bracket */
.lr-card {
  background: #1e293b; border: 1px solid #334155; border-radius: 10px;
  padding: 16px 20px; margin-bottom: 18px;
}
.lr-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .7px; color: #94a3b8; margin-bottom: 4px; }
.lr-subtitle { font-size: 11px; color: #64748b; margin-bottom: 14px; }
.lr-actions { display: flex; align-items: center; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
.lr-save-btn {
  background: #166534; border: 1px solid #16a34a; color: #86efac;
  border-radius: 5px; padding: 4px 14px; font-size: 11px; font-weight: 600;
  cursor: pointer; transition: background .1s;
}
.lr-save-btn:hover:not(:disabled) { background: #15803d; }
.lr-save-btn:disabled { opacity: .5; cursor: not-allowed; }
.lr-clear-btn {
  background: transparent; border: 1px solid #334155; color: #64748b;
  border-radius: 5px; padding: 4px 12px; font-size: 11px; cursor: pointer;
}
.lr-clear-btn:hover { border-color: #ef4444; color: #f87171; }
#locked-save-msg { font-size: 11px; }
#locked-count { font-size: 10px; color: #64748b; margin-left: 4px; }
.lr-toggle { cursor: pointer; user-select: none; display: flex; align-items: center; gap: 6px; }
.lr-toggle:hover { color: #cbd5e1; }
.bk-toggle-icon { font-size: 10px; color: #64748b; margin-left: auto; }

/* Bracket grid */
.bk-wrap { display: flex; flex-direction: row; align-items: flex-start; overflow-x: auto; padding-bottom: 4px; gap: 0; }
.bk-half { display: flex; flex-direction: row; }
.bk-half-r { flex-direction: row-reverse; }
.bk-center { display: flex; flex-direction: row; flex-shrink: 0; }
.bk-col {
  position: relative; width: 118px; flex-shrink: 0;
}
.bk-col-center { position: relative; width: 122px; flex-shrink: 0; }
.bk-col-hdr {
  text-align: center; font-size: 9px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .5px; color: #475569; height: 16px; line-height: 16px; margin-bottom: 2px;
}
.bk-region-hdr {
  position: absolute; font-size: 9px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .6px; color: #64748b; left: 4px; white-space: nowrap;
}
.bk-game {
  position: absolute; left: 0; right: 0;
  display: flex; flex-direction: column;
}
/* Right-side connector arms (left half) */
.bk-left .bk-game::after {
  content: \'\'; position: absolute; right: -1px; top: 50%;
  width: 8px; height: 0; border-top: 1px solid #334155;
  pointer-events: none;
}
/* Left-side connector arms (right half) */
.bk-right .bk-game::after {
  content: \'\'; position: absolute; left: -1px; top: 50%;
  width: 8px; height: 0; border-top: 1px solid #334155;
  pointer-events: none;
}
/* Vertical bracket connector between two games in same column (right side vertical bar) */
.bk-left .bk-game::before {
  content: \'\'; position: absolute; top: 12px; bottom: 12px;
  right: -1px; width: 0; border-right: 1px solid #334155;
  pointer-events: none;
}
.bk-right .bk-game::before {
  content: \'\'; position: absolute; top: 12px; bottom: 12px;
  left: -1px; width: 0; border-left: 1px solid #334155;
  pointer-events: none;
}
/* No connector arm/bar on E8 games (they connect to center differently) */
.bk-e8 .bk-game::before, .bk-e8 .bk-game::after { display: none; }
.bk-col-center .bk-game::before, .bk-col-center .bk-game::after { display: none; }
.bk-team {
  display: block; height: 24px; line-height: 24px;
  padding: 0 6px; font-size: 10px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  background: #0f172a; border: 1px solid #2d3f55; border-radius: 3px;
  color: #94a3b8; cursor: pointer; text-align: left;
  transition: background .12s, color .12s, border-color .12s;
}
.bk-team:hover:not(.bk-tbd) { background: #1e3a5f; color: #93c5fd; border-color: #2563eb; }
.bk-team.bk-winner { background: #14432a; color: #86efac; border-color: #16a34a; font-weight: 700; }
.bk-team.bk-loser  { background: #0a1120; color: #33415a; border-color: #1a2744; }
.bk-team.bk-tbd    { background: #0a1120; color: #1e3a5f; border-color: #0f172a; cursor: default; font-style: italic; }
.bk-seed { font-size: 9px; color: #475569; margin-right: 3px; }
.bk-champ-label {
  text-align: center; font-size: 9px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .5px; color: #fbbf24; margin-top: 4px;
}

/* Results */
.results-card {
  background: #1e293b; border: 1px solid #334155; border-radius: 10px;
  padding: 18px 20px; max-width: 820px;
}
.results-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 14px;
}
.results-title { font-size: 13px; font-weight: 600; color: #e2e8f0; }
.results-meta  { font-size: 11px; color: #64748b; }

table.res-table { width: 100%; border-collapse: collapse; font-size: 12px; }
table.res-table th {
  text-align: left; padding: 6px 10px; color: #64748b; font-size: 10px;
  text-transform: uppercase; letter-spacing: .6px; border-bottom: 1px solid #334155;
  white-space: nowrap;
}
table.res-table td {
  padding: 8px 10px; border-bottom: 1px solid #1a2744;
  color: #cbd5e1; vertical-align: middle;
}
table.res-table tr:last-child td { border-bottom: none; }
table.res-table tbody tr:hover td { background: #172554; }

.rank-cell { color: #475569; font-size: 11px; text-align: right; width: 28px; }
.name-cell { font-weight: 600; color: #e2e8f0; }
.name-link { color: #93c5fd; text-decoration: none; }
.name-link:hover { text-decoration: underline; }
.score-cell { font-size: 14px; font-weight: 700; color: #fbbf24; text-align: right; }
.prob-cell  { text-align: right; min-width: 110px; }
.prob-bar-wrap { display: flex; align-items: center; gap: 8px; justify-content: flex-end; }
.prob-bar-bg {
  width: 80px; height: 8px; background: #0f172a;
  border-radius: 4px; overflow: hidden; border: 1px solid #334155;
  flex-shrink: 0;
}
.prob-bar-fill { height: 100%; border-radius: 4px; transition: width .3s ease; }
.prob-pct { font-size: 12px; font-weight: 700; font-family: monospace; min-width: 46px; text-align: right; }
.minmax-cell { font-size: 11px; color: #64748b; text-align: right; white-space: nowrap; }
</style>
</head>
<body>

<a href="/my_brackets" class="back-link">&#8592; My Brackets</a>
<h1>&#128202; Group Analysis</h1>
<p class="subtitle">
  Select a group and a saved model, then run Monte Carlo simulations to see each bracket's
  expected score and probability of winning the group.
</p>

<!-- Current FF Pairings -->
<div class="ff-info-bar">
  <span class="ff-info-label">Final Four Pairings:</span>
  <span class="ff-info-val">{{ current_ff_label }}</span>
</div>

<!-- Live Tournament Results Bracket -->
<div class="lr-card" id="lr-card" style="display:none">
  <div class="lr-title lr-toggle" onclick="toggleBracket()">
    &#128274; Live Tournament Results
    <span id="bk-toggle-icon" class="bk-toggle-icon">&#9660;</span>
    <span id="locked-count" style="font-size:10px;color:#64748b;margin-left:4px"></span>
  </div>
  <div id="bk-body" style="display:none">
    <div class="lr-subtitle">Click a team to mark them as the winner &mdash; they advance to the next round. Click again to un-lock. Simulations will treat locked results as certain.</div>
    <div id="bk-container"></div>
    <div class="lr-actions">
      <button class="lr-save-btn" id="save-locked-btn" onclick="saveLockedResults()">&#128190; Save Results</button>
      <button class="lr-clear-btn" onclick="clearLockedResults()">&#10006; Clear All</button>
      <span id="locked-save-msg"></span>
    </div>
  </div>
</div>

{% if saved_models %}
<!-- Existing Models -->
<div class="sims-card">
  <div class="sims-title">&#127936; Models &mdash; Select to Run Simulation</div>
  <table class="sims-table" id="ga-models-table">
    <thead><tr>
      <th class="sortable" data-sort="name" onclick="gaSortBy('name')">Name</th>
      <th class="sortable" data-sort="model" onclick="gaSortBy('model')">Model</th>
      <th class="sortable" data-sort="flags" onclick="gaSortBy('flags')">Flags</th>
      <th class="sortable" data-sort="features" onclick="gaSortBy('features')">Features</th>
      <th class="sortable sort-desc" style="text-align:right" data-sort="score" onclick="gaSortBy('score')">Score</th>
      <th class="sortable" style="text-align:right" data-sort="train_acc" onclick="gaSortBy('train_acc')">Train&nbsp;Acc</th>
      <th class="sortable" style="text-align:right" data-sort="test_acc" onclick="gaSortBy('test_acc')">Test&nbsp;Acc</th>
      <th></th>
      <th></th>
    </tr></thead>
    <tbody id="ga-models-tbody"></tbody>
  </table>
</div>
{% endif %}

<!-- Config -->
<div class="config-card">
  <div class="config-grid">
    <div class="field">
      <label for="grp-sel">Group</label>
      <select id="grp-sel">
        {% for g in groups_list %}
        <option value="{{ g }}" {% if g == group %}selected{% endif %}>{{ g }}</option>
        {% endfor %}
      </select>
    </div>
    <div class="field" style="grid-column:span 2">
      <label for="model-sel">Model</label>
      <select id="model-sel">
        <option value="">-- select model --</option>
        {% for m in saved_models %}
        <option value="{{ m.dir_name }}">{{ m.dir_name }}</option>
        {% endfor %}
      </select>
    </div>
    <div class="field">
      <label for="iters-inp">Iterations</label>
      <input type="number" id="iters-inp" value="1000" min="100" max="100000">
    </div>
    <div class="field">
      <label for="seed-inp">Seed (opt.)</label>
      <input type="number" id="seed-inp" placeholder="random" style="width:90px">
    </div>
    <button id="run-btn" onclick="runAnalysis()">&#9654; Run</button>
    <span id="status-badge" class="status-badge" style="display:none;align-self:center"></span>
  </div>
</div>

<!-- Progress -->
<div class="prog-wrap" id="prog-wrap" style="display:none">
  <div class="prog-header">
    <span class="prog-label">Simulating&hellip;</span>
    <span id="prog-text">0 / 0</span>
  </div>
  <div class="prog-track"><div id="prog-bar"></div></div>
</div>

<!-- Error -->
<div id="error-box"></div>

<!-- Results -->
<div class="results-card" id="results-section" style="display:none">
  <div class="results-header">
    <span class="results-title">Results</span>
    <span class="results-meta" id="results-meta"></span>
  </div>
  <table class="res-table">
    <thead>
      <tr>
        <th style="width:28px">#</th>
        <th>Bracket</th>
        <th style="text-align:right">Avg Score</th>
        <th style="text-align:right">Win Probability</th>
        <th style="text-align:right">Max&nbsp;/&nbsp;Min</th>
      </tr>
    </thead>
    <tbody id="results-tbody"></tbody>
  </table>
</div>

<div style="margin-top:10px;text-align:center">
  <a id="sim-view-link" href="#" target="_blank"
     style="display:none;padding:6px 14px;background:#334155;color:#e2e8f0;
            border-radius:6px;text-decoration:none;font-size:0.85rem">
    &#128202; View Simulation Table
  </a>
</div>

<script>
const THIS_YEAR   = {{ year }};
const GROUP_INIT  = {{ group | tojson }};
const GA_MODELS   = {{ saved_models | tojson }};
const GA_SIM_MAP  = {{ sim_map | tojson }};
const BRACKET_R1          = {{ r1_matchups | tojson }};
const LOCKED_RESULTS_INIT = {{ locked_results | tojson }};

let gaSort = { key: 'score', dir: -1 };

function gaSortVal(m, key) {
  if (key === 'name')      return (m.dir_name  || '').toLowerCase();
  if (key === 'model')     return (m.model     || '').toLowerCase();
  if (key === 'flags')     return (m.expert    || '').toLowerCase();
  if (key === 'features')  return (m.features  || '').toLowerCase();
  if (key === 'score')     return m.score     != null ? m.score     : -Infinity;
  if (key === 'train_acc') return m.train_acc != null ? m.train_acc : -Infinity;
  if (key === 'test_acc')  return m.test_acc  != null ? m.test_acc  : -Infinity;
  return '';
}

function gaSortBy(key) {
  if (gaSort.key === key) { gaSort.dir *= -1; }
  else { gaSort.key = key; gaSort.dir = (key === 'score' || key === 'train_acc' || key === 'test_acc') ? -1 : 1; }
  gaRenderModels();
}

function gaRenderModels() {
  const sorted = GA_MODELS.slice().sort(function(a, b) {
    const av = gaSortVal(a, gaSort.key), bv = gaSortVal(b, gaSort.key);
    if (av < bv) return gaSort.dir; if (av > bv) return -gaSort.dir; return 0;
  });
  document.querySelectorAll('#ga-models-table th[data-sort]').forEach(function(th) {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.sort === gaSort.key) th.classList.add(gaSort.dir === 1 ? 'sort-asc' : 'sort-desc');
  });
  const tbody = document.getElementById('ga-models-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  const fmtAcc = function(v) { return v == null ? '\u2014' : (v * 100).toFixed(1) + '%'; };
  sorted.forEach(function(m) {
    const sim   = GA_SIM_MAP[m.dir_name];
    const flags = [m.norm_years ? 'NY' : '', m.norm_all ? 'NA' : '', m.calibrated ? 'CAL' : '', m.delta_feats ? 'DF' : ''].filter(Boolean).join('\u00a0') || '\u2014';
    const feat  = m.features.length > 35 ? m.features.slice(0, 35) + '\u2026' : m.features;
    const viewBtn = (sim && sim.html_file)
      ? '<a href="/sim_html/' + encodeURIComponent(m.dir_name) + '/' + encodeURIComponent(sim.html_file) + '" target="_blank" class="quick-run-btn" style="text-decoration:none;display:inline-block">&#128200; View</a>'
      : '';
    const useBtn = m.has_pkl
      ? '<button class="quick-run-btn" onclick=\'quickRun(' + JSON.stringify(m.dir_name) + ')\'>&#9654; Use This</button>'
      : '<span class="no-pkl-note">No model.pkl</span>';
    const tr = document.createElement('tr');
    tr.style.cursor = 'pointer';
    tr.innerHTML =
      '<td style="color:#e2e8f0;font-weight:600">' + m.dir_name + '</td>' +
      '<td style="color:#93c5fd;font-size:11px">' + m.model + '</td>' +
      '<td style="color:#fbbf24;font-size:10px;white-space:nowrap">' + flags + '</td>' +
      '<td style="color:#94a3b8;font-size:11px">' + feat + '</td>' +
      '<td style="color:#fbbf24;font-weight:700;text-align:right">' + m.score + 'pts</td>' +
      '<td style="color:#94a3b8;font-size:11px;font-family:monospace;text-align:right">' + fmtAcc(m.train_acc) + '</td>' +
      '<td style="color:#94a3b8;font-size:11px;font-family:monospace;text-align:right">' + fmtAcc(m.test_acc)  + '</td>' +
      '<td>' + viewBtn + '</td>' +
      '<td>' + useBtn  + '</td>';
    tr.addEventListener('click', function(e) {
      if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A') return;
      const sel = document.getElementById('model-sel');
      for (let i = 0; i < sel.options.length; i++) {
        if (sel.options[i].value === m.dir_name) { sel.selectedIndex = i; break; }
      }
    });
    tbody.appendChild(tr);
  });
}

gaRenderModels();

// ---- Locked Tournament Results ----
(function() {
  // Deep-copy LOCKED_RESULTS_INIT into mutable lockedResults
  window.lockedResults = {};
  for (const k of Object.keys(LOCKED_RESULTS_INIT)) {
    window.lockedResults[k] = (LOCKED_RESULTS_INIT[k] || []).map(function(v) { return v || null; });
  }
})();

function parseFfPairings(s) {
  return s.split(',').map(function(p) { return p.split('-').map(Number); });
}
const FF_PAIRINGS = parseFfPairings('{{ current_ff_str }}');

const ROUND_LABELS_LR = {1:'Round of 64', 2:'Round of 32', 3:'Sweet 16', 4:'Elite Eight', 5:'Final Four', 6:'Championship'};

function buildRoundMatchups(rnd, prevWinners) {
  if (rnd === 5) {
    return [
      [prevWinners[FF_PAIRINGS[0][0]] || null, prevWinners[FF_PAIRINGS[0][1]] || null],
      [prevWinners[FF_PAIRINGS[1][0]] || null, prevWinners[FF_PAIRINGS[1][1]] || null],
    ];
  }
  var matchups = [];
  for (var i = 0; i < prevWinners.length; i += 2) {
    matchups.push([prevWinners[i] || null, prevWinners[i + 1] || null]);
  }
  return matchups;
}

// ---- Bracket rendering constants ----
var BK_SLOT   = 52;   // px: height of one game slot (2×22px btn + 8px gap)
var BK_GAP    = 8;    // px gap between the two team buttons inside a slot
var BK_BTN    = (BK_SLOT - BK_GAP) / 2;  // 22px button height
// Column widths
var BK_COL_W       = 118;  // round column
var BK_COL_CENTER_W = 122; // FF/Champ column
// Per-region height = 8 games × BK_SLOT
var BK_REGION_H  = 8 * BK_SLOT;    // 416
var BK_RGN_LABEL = 14;             // px reserved above each region for its name label
// Total column height = 2 regions stacked (each region has a label row above it)
var BK_HALF_H    = 2 * (BK_REGION_H + BK_RGN_LABEL); // 860

// Region definitions: [regionName, r1StartIdx]
// r1StartIdx = start index in BRACKET_R1/lockedResults[1] arrays
var BK_REGIONS = [
  { name: 'South',   idx: 0  },
  { name: 'East',    idx: 8  },
  { name: 'West',    idx: 16 },
  { name: 'Midwest', idx: 24 },
];

// Global game index for region r, round rnd (1-4), game i within region
function bkGlobalIdx(r, rnd, i) {
  var gamesPerRegion = 8 >> (rnd - 1);  // 8,4,2,1
  return gamesPerRegion * r + i;
}

// Absolute top position (px) for game i in round rnd within a region that starts at regionY
function bkGameTop(rnd, i, regionY) {
  var stride = BK_SLOT * (1 << (rnd - 1));
  return regionY + stride * i + (stride - BK_SLOT) / 2;
}

// Derive all winners up to round maxRnd from lockedResults
function bkAllWinners() {
  var w = {};
  for (var r = 1; r <= 6; r++) {
    w[r] = (lockedResults[String(r)] || []).map(function(v) { return v || null; });
  }
  return w;
}

// Build one game div and attach click handler
function bkMakeGame(rnd, globalIdx, t1, t2, seed1, seed2, winner) {
  var div = document.createElement('div');
  div.className = 'bk-game';

  function makeBtn(team, seed, isWinner, hasWinner) {
    var btn = document.createElement('button');
    if (!team) {
      btn.className = 'bk-team bk-tbd';
      btn.textContent = 'TBD';
      btn.disabled = true;
    } else {
      btn.className = 'bk-team' + (isWinner ? ' bk-winner' : (hasWinner ? ' bk-loser' : ''));
      var seedSpan = document.createElement('span');
      seedSpan.className = 'bk-seed';
      seedSpan.textContent = seed ? '[' + seed + ']' : '';
      btn.appendChild(seedSpan);
      btn.appendChild(document.createTextNode(team));
      btn.title = team;
      (function(tm) {
        btn.addEventListener('click', function() { lockGame(rnd, globalIdx, tm); });
      }(team));
    }
    btn.style.height = BK_BTN + 'px';
    btn.style.lineHeight = BK_BTN + 'px';
    return btn;
  }

  div.appendChild(makeBtn(t1, seed1, winner === t1, !!winner));
  var gap = document.createElement('div');
  gap.style.height = BK_GAP + 'px';
  div.appendChild(gap);
  div.appendChild(makeBtn(t2, seed2, winner === t2, !!winner));
  return div;
}

function renderBracket() {
  var container = document.getElementById('bk-container');
  if (!container) return;
  container.innerHTML = '';
  container.className = 'bk-wrap';

  var W = bkAllWinners();

  // Seeds lookup from BRACKET_R1
  var seedMap = {};
  BRACKET_R1.forEach(function(g) {
    if (g.team1) seedMap[g.team1] = g.seed1;
    if (g.team2) seedMap[g.team2] = g.seed2;
  });

  // Build a column div for a single region-round slice
  // side: 'bk-left' or 'bk-right'
  // regionIdx: 0-3
  // rnd: 1-4
  // regionInHalf: 0 (top) or 1 (bottom) within the half
  function makeRegionRoundCol(side, regionIdx, rnd, regionInHalf) {
    var col = document.createElement('div');
    col.className = 'bk-col ' + side + (rnd === 4 ? ' bk-e8' : '');
    col.style.height = (BK_HALF_H + 18) + 'px';  // +18 for header

    // Each region is preceded by BK_RGN_LABEL px for its name label
    var regionY = regionInHalf * (BK_REGION_H + BK_RGN_LABEL) + BK_RGN_LABEL;
    var gamesInRound = 8 >> (rnd - 1);  // 8,4,2,1

    for (var i = 0; i < gamesInRound; i++) {
      var gi = bkGlobalIdx(regionIdx, rnd, i);
      var t1 = null, t2 = null, seed1 = null, seed2 = null;

      if (rnd === 1) {
        t1 = BRACKET_R1[gi].team1; t2 = BRACKET_R1[gi].team2;
        seed1 = BRACKET_R1[gi].seed1; seed2 = BRACKET_R1[gi].seed2;
      } else {
        // team1 = winner of prev-round game 2*i, team2 = winner of prev game 2*i+1
        var prevGi1 = bkGlobalIdx(regionIdx, rnd - 1, 2 * i);
        var prevGi2 = bkGlobalIdx(regionIdx, rnd - 1, 2 * i + 1);
        t1 = (W[rnd - 1] || [])[prevGi1] || null;
        t2 = (W[rnd - 1] || [])[prevGi2] || null;
        if (t1) seed1 = seedMap[t1] || null;
        if (t2) seed2 = seedMap[t2] || null;
      }

      var winner = (W[rnd] || [])[gi] || null;
      var gameDiv = bkMakeGame(rnd, gi, t1, t2, seed1, seed2, winner);
      var top = bkGameTop(rnd, i, regionY);
      gameDiv.style.top  = (top + 18) + 'px';  // +18 for header
      gameDiv.style.left = '4px';
      gameDiv.style.right = '4px';
      col.appendChild(gameDiv);
    }
    return col;
  }

  // Region-name labels (placed on R64 column, at region start)
  function addRegionLabel(col, name, regionInHalf) {
    var lbl = document.createElement('div');
    lbl.className = 'bk-region-hdr';
    lbl.textContent = name;
    lbl.style.top = (18 + regionInHalf * (BK_REGION_H + BK_RGN_LABEL)) + 'px';
    col.appendChild(lbl);
  }

  // Round-name header row (placed across top of col)
  var ROUND_HDRS = {1: 'R64', 2: 'R32', 3: 'Sweet 16', 4: 'Elite 8', 5: 'Final Four', 6: 'Champ'};

  function makeColHdr(col, label) {
    var hdr = document.createElement('div');
    hdr.className = 'bk-col-hdr';
    hdr.textContent = label;
    hdr.style.position = 'absolute';
    hdr.style.top = '0';
    hdr.style.left = '0';
    hdr.style.right = '0';
    col.appendChild(hdr);
  }

  // --- LEFT HALF (South + East, rounds 1-4 going left→right) ---
  var leftHalf = document.createElement('div');
  leftHalf.className = 'bk-half';

  for (var rnd = 1; rnd <= 4; rnd++) {
    var col = document.createElement('div');
    col.className = 'bk-col bk-left' + (rnd === 4 ? ' bk-e8' : '');
    col.style.height = (BK_HALF_H + 18) + 'px';
    makeColHdr(col, ROUND_HDRS[rnd]);

    // South (region 0, top half)
    var southR = makeRegionRoundCol('bk-left', 0, rnd, 0);
    Array.prototype.forEach.call(southR.querySelectorAll('.bk-game, .bk-region-hdr'), function(el) {
      col.appendChild(el);
    });
    // East (region 1, bottom half)
    var eastR = makeRegionRoundCol('bk-left', 1, rnd, 1);
    Array.prototype.forEach.call(eastR.querySelectorAll('.bk-game, .bk-region-hdr'), function(el) {
      col.appendChild(el);
    });

    if (rnd === 1) {
      addRegionLabel(col, 'South', 0);
      addRegionLabel(col, 'East', 1);
    }
    leftHalf.appendChild(col);
  }

  // --- CENTER (FF1 | Championship | FF2) ---
  var centerSec = document.createElement('div');
  centerSec.className = 'bk-center';

  // E8 winners for FF pairings: [South_E8, East_E8, West_E8, Midwest_E8]
  var e8Winners = [0, 1, 2, 3].map(function(r) {
    var gi = bkGlobalIdx(r, 4, 0);
    return (W[4] || [])[gi] || null;
  });

  // FF game: center between the two region bands, accounting for label rows
  var ffTop = BK_RGN_LABEL + BK_REGION_H + Math.floor(BK_RGN_LABEL / 2) - Math.floor(BK_SLOT / 2);

  function makeCenterCol(label, rnd, gi, t1, t2, seed1, seed2) {
    var col = document.createElement('div');
    col.className = 'bk-col-center';
    col.style.height = (BK_HALF_H + 18) + 'px';
    makeColHdr(col, label);
    var winner = (W[rnd] || [])[gi] || null;
    var gameDiv = bkMakeGame(rnd, gi, t1, t2, seed1, seed2, winner);
    gameDiv.style.top   = (ffTop + 18) + 'px';
    gameDiv.style.left  = '4px';
    gameDiv.style.right = '4px';
    col.appendChild(gameDiv);
    if (rnd === 6 && winner) {
      var lbl = document.createElement('div');
      lbl.className = 'bk-champ-label';
      lbl.textContent = '\u2605 Champion';
      lbl.style.position = 'absolute';
      lbl.style.top = (ffTop + BK_SLOT + 22) + 'px';
      lbl.style.left = '0'; lbl.style.right = '0';
      col.appendChild(lbl);
    }
    return col;
  }

  // FF1: FF_PAIRINGS[0] = [e8Idx0, e8Idx1]
  var ff1T1 = e8Winners[FF_PAIRINGS[0][0]], ff1T2 = e8Winners[FF_PAIRINGS[0][1]];
  // FF2: FF_PAIRINGS[1] = [e8Idx2, e8Idx3]
  var ff2T1 = e8Winners[FF_PAIRINGS[1][0]], ff2T2 = e8Winners[FF_PAIRINGS[1][1]];
  // Champ: winner of FF1 vs winner of FF2
  var champT1 = (W[5] || [])[0] || null;
  var champT2 = (W[5] || [])[1] || null;

  centerSec.appendChild(makeCenterCol('Final Four', 5, 0, ff1T1, ff1T2,
    ff1T1 ? (seedMap[ff1T1] || null) : null, ff1T2 ? (seedMap[ff1T2] || null) : null));
  centerSec.appendChild(makeCenterCol('Championship', 6, 0, champT1, champT2,
    champT1 ? (seedMap[champT1] || null) : null, champT2 ? (seedMap[champT2] || null) : null));
  centerSec.appendChild(makeCenterCol('Final Four', 5, 1, ff2T1, ff2T2,
    ff2T1 ? (seedMap[ff2T1] || null) : null, ff2T2 ? (seedMap[ff2T2] || null) : null));

  // --- RIGHT HALF (West + Midwest, rounds 4-1 going left→right, i.e. E8 nearest center) ---
  var rightHalf = document.createElement('div');
  rightHalf.className = 'bk-half bk-half-r';

  for (var rnd2 = 1; rnd2 <= 4; rnd2++) {
    var col2 = document.createElement('div');
    col2.className = 'bk-col bk-right' + (rnd2 === 4 ? ' bk-e8' : '');
    col2.style.height = (BK_HALF_H + 18) + 'px';
    makeColHdr(col2, ROUND_HDRS[rnd2]);

    // West (region 2, top half)
    var westR = makeRegionRoundCol('bk-right', 2, rnd2, 0);
    Array.prototype.forEach.call(westR.querySelectorAll('.bk-game, .bk-region-hdr'), function(el) {
      col2.appendChild(el);
    });
    // Midwest (region 3, bottom half)
    var mwR = makeRegionRoundCol('bk-right', 3, rnd2, 1);
    Array.prototype.forEach.call(mwR.querySelectorAll('.bk-game, .bk-region-hdr'), function(el) {
      col2.appendChild(el);
    });

    if (rnd2 === 1) {
      addRegionLabel(col2, 'West', 0);
      addRegionLabel(col2, 'Midwest', 1);
    }
    rightHalf.appendChild(col2);
  }

  container.appendChild(leftHalf);
  container.appendChild(centerSec);
  container.appendChild(rightHalf);

  // Update locked count badge
  var total = 0;
  for (var r2 = 1; r2 <= 6; r2++) {
    var arr = lockedResults[String(r2)] || [];
    total += arr.filter(function(v) { return v !== null && v !== ''; }).length;
  }
  var countEl = document.getElementById('locked-count');
  if (countEl) countEl.textContent = total > 0 ? '(' + total + ' game' + (total !== 1 ? 's' : '') + ' locked)' : '';
}

function lockGame(rnd, gi, team) {
  var rnds = String(rnd);
  var lengths = {1:32, 2:16, 3:8, 4:4, 5:2, 6:1};
  if (!lockedResults[rnds]) {
    lockedResults[rnds] = new Array(lengths[rnd] || 1).fill(null);
  }
  var prev = lockedResults[rnds][gi];
  if (prev === team) {
    lockedResults[rnds][gi] = null;
    clearDownstream(rnd);
  } else {
    if (prev && prev !== team) { clearDownstream(rnd); }
    lockedResults[rnds][gi] = team;
  }
  renderBracket();
}

function clearDownstream(rnd) {
  for (var r = rnd + 1; r <= 6; r++) {
    delete lockedResults[String(r)];
  }
}

function saveLockedResults() {
  var btn   = document.getElementById('save-locked-btn');
  var msgEl = document.getElementById('locked-save-msg');
  if (btn) btn.disabled = true;
  if (msgEl) { msgEl.textContent = 'Saving\u2026'; msgEl.style.color = '#94a3b8'; }
  var toSave = {};
  for (var rnd of Object.keys(lockedResults)) {
    var arr = lockedResults[rnd];
    if (arr && arr.some(function(w) { return w !== null && w !== ''; })) {
      toSave[rnd] = arr;
    }
  }
  fetch('/api/locked_results', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({round_winners: toSave}),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (btn) btn.disabled = false;
    if (data.ok) {
      if (msgEl) { msgEl.textContent = '\u2713 Saved'; msgEl.style.color = '#86efac'; }
      setTimeout(function() { if (msgEl) msgEl.textContent = ''; }, 2000);
    } else {
      if (msgEl) { msgEl.textContent = 'Error saving'; msgEl.style.color = '#f87171'; }
    }
  })
  .catch(function() {
    if (btn) btn.disabled = false;
    if (msgEl) { msgEl.textContent = 'Error saving'; msgEl.style.color = '#f87171'; }
  });
}

function clearLockedResults() {
  if (!confirm('Clear all locked game results?')) return;
  lockedResults = {};
  saveLockedResults();
  renderBracket();
}

function toggleBracket() {
  var body = document.getElementById('bk-body');
  var icon = document.getElementById('bk-toggle-icon');
  if (!body) return;
  var expanding = body.style.display === 'none';
  body.style.display = expanding ? '' : 'none';
  if (icon) icon.textContent = expanding ? '\u25b2' : '\u25bc';
}

function initLockedResults() {
  if (!BRACKET_R1 || !BRACKET_R1.length || !BRACKET_R1[0].team1) return;
  var card = document.getElementById('lr-card');
  if (card) card.style.display = '';
  renderBracket();
}

initLockedResults();

function probColor(p) {
  if (p >= 0.40) return '#86efac';
  if (p >= 0.20) return '#fde68a';
  if (p >= 0.05) return '#fb923c';
  return '#94a3b8';
}

function fmtPct(p) {
  if (p === 0) return '0.0%';
  if (p >= 0.9995) return '100%';
  if (p < 0.001)   return '<0.1%';
  return (p * 100).toFixed(1) + '%';
}

function renderResults(results, meta) {
  const tbody = document.getElementById('results-tbody');
  const section = document.getElementById('results-section');
  const metaEl  = document.getElementById('results-meta');
  tbody.innerHTML = '';
  metaEl.textContent = meta || '';

  // Find max avg for bar scaling
  const maxAvg = Math.max(...results.map(r => r.avg_score), 1);
  const groupKey = document.getElementById('grp-sel').value;

  results.forEach((r, i) => {
    const pct = fmtPct(r.win_prob);
    const barW = Math.round(r.win_prob * 100);
    const color = probColor(r.win_prob);
    const loadUrl = `/fill_bracket?load=${encodeURIComponent(groupKey + '/' + r.file)}&bname=${encodeURIComponent(r.name)}&bgrp=${encodeURIComponent(groupKey)}`;
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td class="rank-cell">${i + 1}</td>` +
      `<td class="name-cell"><a class="name-link" href="${loadUrl}" target="_blank">${r.name}</a></td>` +
      `<td class="score-cell">${r.avg_score.toFixed(1)}</td>` +
      `<td class="prob-cell">
         <div class="prob-bar-wrap">
           <span class="prob-pct" style="color:${color}">${pct}</span>
           <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${barW}%;background:${color}"></div></div>
         </div>
       </td>` +
      `<td class="minmax-cell">${r.max_score} / ${r.min_score}</td>`;
    tbody.appendChild(tr);
  });

  section.style.display = '';
}

function setStatus(text, cls) {
  const b = document.getElementById('status-badge');
  b.style.display = '';
  b.className = 'status-badge ' + cls;
  b.textContent = text;
}

function runAnalysis(dirNameOverride, itersOverride) {
  const group    = document.getElementById('grp-sel').value;
  const dirName  = dirNameOverride || document.getElementById('model-sel').value;
  const iters    = itersOverride   || parseInt(document.getElementById('iters-inp').value) || 2000;
  const seedRaw  = document.getElementById('seed-inp').value.trim();
  const seed     = seedRaw !== '' ? parseInt(seedRaw) : null;

  if (!group)   { alert('Select a group.'); return; }
  if (!dirName) { alert('Select a model.'); return; }

  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  document.getElementById('error-box').style.display = 'none';
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('sim-view-link').style.display = 'none';
  document.getElementById('prog-bar').style.width = '0%';
  document.getElementById('prog-text').textContent = '0 / ' + iters.toLocaleString();
  document.getElementById('prog-wrap').style.display = '';
  setStatus('Running\u2026', 'status-running');

  fetch('/run_group_analysis', {
    method:  'POST',
    headers: {'Content-Type': 'application/json'},
    body:    JSON.stringify({ group, dir_name: dirName, num_iters: iters,
                              seed, year: THIS_YEAR }),
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) {
      btn.disabled = false;
      document.getElementById('prog-wrap').style.display = 'none';
      const eb = document.getElementById('error-box');
      eb.textContent = data.error;
      eb.style.display = '';
      setStatus('Error', 'status-error');
      return;
    }
    startStream(data.job_id, iters, dirName);
  })
  .catch(err => {
    btn.disabled = false;
    document.getElementById('prog-wrap').style.display = 'none';
    setStatus('Error', 'status-error');
  });
}

function quickRun(dirName, iters) {
  const sel = document.getElementById('model-sel');
  for (let i = 0; i < sel.options.length; i++) {
    if (sel.options[i].value === dirName) { sel.selectedIndex = i; break; }
  }
  if (iters != null) document.getElementById('iters-inp').value = iters;
  runAnalysis(dirName, iters || parseInt(document.getElementById('iters-inp').value) || 1000);
}

function startStream(jobId, iters, dirName) {
  const es = new EventSource('/group_analysis_stream/' + jobId);
  es.onmessage = function(e) {
    const msg = JSON.parse(e.data);

    if (msg.type === 'line') {
      if (msg.text.startsWith('PROGRESS:')) {
        const parts = msg.text.slice(9).split('/');
        const done  = parseInt(parts[0]);
        const total = parseInt(parts[1]);
        const pct   = total > 0 ? done / total * 100 : 0;
        document.getElementById('prog-bar').style.width = pct + '%';
        document.getElementById('prog-text').textContent =
          done.toLocaleString() + ' / ' + total.toLocaleString();
        return;
      }
      if (msg.text.startsWith('[ERROR]') || msg.text.startsWith('[TRACEBACK]')) {
        const eb = document.getElementById('error-box');
        eb.textContent += msg.text + '\n';
        eb.style.display = '';
      }
      return;
    }

    if (msg.type === 'done') {
      es.close();
      document.getElementById('prog-wrap').style.display = 'none';
      document.getElementById('run-btn').disabled = false;

      if (msg.status === 'done' && msg.results) {
        setStatus('Done', 'status-done');
        const meta = `${iters.toLocaleString()} iterations \u00b7 model: ${dirName}`;
        renderResults(msg.results, meta);
        if (msg.sim_url) {
          const lnk = document.getElementById('sim-view-link');
          lnk.href = msg.sim_url;
          lnk.style.display = '';
        }
      } else {
        setStatus('Error', 'status-error');
      }
      return;
    }

    if (msg.type === 'timeout') {
      es.close();
      document.getElementById('prog-wrap').style.display = 'none';
      document.getElementById('run-btn').disabled = false;
      setStatus('Timeout', 'status-error');
    }
  };
  es.onerror = function() {
    es.close();
    document.getElementById('prog-wrap').style.display = 'none';
    document.getElementById('run-btn').disabled = false;
    setStatus('Disconnected', 'status-error');
  };
}

// Pre-select group from URL param
(function() {
  if (GROUP_INIT) {
    const sel = document.getElementById('grp-sel');
    for (let i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === GROUP_INIT) { sel.selectedIndex = i; break; }
    }
  }
})();
</script>
</body>
</html>
"""


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>March Madness Bracket Predictor</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Segoe UI", Arial, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  min-height: 100vh;
  padding: 32px 24px;
}
h1 { font-size: 22px; color: #fbbf24; margin-bottom: 4px; }
.subtitle { color: #64748b; font-size: 13px; margin-bottom: 28px; }

/* ---- layout ---- */
.layout { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
.form-panel {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 24px;
  width: 480px;
  flex-shrink: 0;
}
.output-panel {
  flex: 1;
  min-width: 340px;
}

/* ---- form elements ---- */
label.field-label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .7px;
  color: #94a3b8;
  margin-bottom: 6px;
  margin-top: 18px;
}
label.field-label:first-of-type { margin-top: 0; }

select, input[type="text"] {
  width: 100%;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  color: #e2e8f0;
  padding: 8px 10px;
  font-size: 13px;
  outline: none;
}
select:focus, input[type="text"]:focus { border-color: #3b82f6; }

.radio-group { display: flex; gap: 12px; }
.radio-group label {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  font-size: 13px;
  color: #e2e8f0;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 7px 14px;
  transition: border-color .15s;
}
.radio-group input[type="radio"] { accent-color: #3b82f6; cursor: pointer; }
.radio-group label:has(input:checked) { border-color: #3b82f6; color: #93c5fd; }

/* ---- feature grid ---- */
.feat-section { margin-bottom: 10px; }
.feat-section-title {
  font-size: 10px;
  font-weight: 600;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: .6px;
  margin-bottom: 5px;
}
.feat-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.feat-chip {
  display: flex;
  align-items: center;
  gap: 4px;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 20px;
  padding: 3px 10px;
  font-size: 11px;
  cursor: pointer;
  color: #94a3b8;
  transition: border-color .12s, color .12s, background .12s;
  user-select: none;
}
.feat-chip input[type="checkbox"] { display: none; }
.feat-chip.selected {
  border-color: #3b82f6;
  color: #93c5fd;
  background: #1e3a5f;
}
.feat-chip.kp-only.selected   { border-color: #8b5cf6; color: #c4b5fd; background: #2e1065; }
.feat-chip.bt-only.selected   { border-color: #ec4899; color: #f9a8d4; background: #500724; }
.feat-chip.bt2w-only.selected { border-color: #f97316; color: #fed7aa; background: #431407; }
.feat-chip.bthot-only.selected{ border-color: #14b8a6; color: #99f6e4; background: #042f2e; }
.feat-chip.meta.selected      { border-color: #22c55e; color: #86efac; background: #14432a; }

.feat-legend {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 8px;
  font-size: 10px;
  color: #64748b;
}
.feat-legend span { display: flex; align-items: center; gap: 4px; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-common { background: #3b82f6; }
.dot-kp     { background: #8b5cf6; }
.dot-bt     { background: #ec4899; }
.dot-bt2w   { background: #f97316; }
.dot-bthot  { background: #14b8a6; }
.dot-meta   { background: #22c55e; }

.hint { font-size: 10px; color: #475569; margin-top: 4px; }

/* ---- run button ---- */
#run-btn {
  margin-top: 22px;
  width: 100%;
  background: #ca8a04;
  color: #fefce8;
  border: none;
  border-radius: 7px;
  padding: 11px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background .15s;
}
#run-btn:hover:not(:disabled) { background: #eab308; }
#run-btn:disabled { opacity: .5; cursor: not-allowed; }

/* ---- output panel ---- */
.panel-card {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 16px;
}
.panel-title {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .6px;
  color: #64748b;
  margin-bottom: 10px;
}
#log-box {
  background: #020617;
  border-radius: 6px;
  padding: 10px 12px;
  font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
  font-size: 11px;
  color: #94a3b8;
  height: 360px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
.log-line-eq     { color: #64748b; }
.log-line-round  { color: #7dd3fc; }
.log-line-year   { color: #fbbf24; font-weight: 600; }
.log-line-model  { color: #a5f3fc; }
.log-line-result { color: #86efac; }
.log-line-err    { color: #fca5a5; }

.status-badge {
  display: inline-block;
  border-radius: 12px;
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 600;
}
.status-running { background: #1e3a5f; color: #93c5fd; }
.status-done    { background: #14432a; color: #86efac; }
.status-error   { background: #450a0a; color: #fca5a5; }
.status-idle    { background: #1e293b; color: #64748b; }

#summary-box {
  background: #020617;
  border-radius: 6px;
  padding: 12px;
  font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
  font-size: 11px;
  color: #94a3b8;
  white-space: pre;
  overflow-x: auto;
  max-height: 320px;
  overflow-y: auto;
}

/* ---- year links ---- */
.year-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.year-btn {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 6px 14px;
  font-size: 12px;
  color: #94a3b8;
  text-decoration: none;
  transition: border-color .15s, color .15s;
}
.year-btn:hover { border-color: #fbbf24; color: #fbbf24; }
.year-btn.current { border-color: #f59e0b; color: #fde68a; font-weight: 600; }

/* ---- idle state ---- */
#results-section { display: none; }

/* ---- saved models table ---- */
.saved-section { margin-bottom: 24px; }
.saved-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.saved-table th {
  text-align: left; padding: 6px 10px;
  color: #64748b; font-size: 10px;
  text-transform: uppercase; letter-spacing: .6px;
  border-bottom: 1px solid #334155;
}
.saved-table td {
  padding: 7px 10px;
  border-bottom: 1px solid #1a2744;
  color: #cbd5e1; vertical-align: middle;
}
.saved-table tr.model-row { cursor: pointer; transition: background .1s; }
.saved-table tr.model-row:hover  { background: #1e293b; }
.saved-table tr.model-row.active { background: #172554; }
.score-cell { font-weight: 700; font-size: 14px; color: #fbbf24; white-space: nowrap; }
.tag {
  display: inline-block; border-radius: 4px;
  padding: 1px 6px; font-size: 10px; font-weight: 600; margin-right: 3px;
}
.tag-kp { background: #1e3a5f; color: #93c5fd; }
.tag-ny  { background: #2d1b4e; color: #c4b5fd; margin-left: 3px; }
.tag-na  { background: #0c2a1a; color: #6ee7b7; margin-left: 3px; }
.tag-cal { background: #3b1a0a; color: #fdba74; margin-left: 3px; }
.tag-bt { background: #500724; color: #f9a8d4; }
.feat-tags { color: #64748b; font-size: 10px; }
.saved-table th.sortable { cursor: pointer; user-select: none; white-space: nowrap; }
.saved-table th.sortable:hover { color: #93c5fd; }
.saved-table th.sort-asc::after  { content: ' \25b2'; font-size: 9px; }
.saved-table th.sort-desc::after { content: ' \25bc'; font-size: 9px; }

/* ---- simulation card ---- */
.sim-form { display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 12px; }
.sim-form > div { display: flex; flex-direction: column; }
.sim-form label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; color: #94a3b8; margin-bottom: 4px; }
.sim-form select, .sim-form input[type="number"] {
  background: #0f172a; border: 1px solid #334155; border-radius: 6px;
  color: #e2e8f0; padding: 7px 9px; font-size: 12px; outline: none;
}
.sim-form select:focus, .sim-form input[type="number"]:focus { border-color: #3b82f6; }
#sim-year  { width: 90px; }
#sim-iters { width: 90px; }
#sim-seed  { width: 80px; }
#sim-btn {
  background: #0f4c81; color: #bfdbfe; border: 1px solid #1d4ed8;
  border-radius: 6px; padding: 7px 18px; font-size: 12px; font-weight: 600;
  cursor: pointer; transition: background .15s; white-space: nowrap;
}
#sim-btn:hover:not(:disabled) { background: #1d4ed8; color: #eff6ff; }
#sim-btn:disabled { opacity: .5; cursor: not-allowed; }
#sim-log {
  background: #020617; border-radius: 6px; padding: 8px 10px;
  font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
  font-size: 10px; color: #94a3b8; max-height: 160px;
  overflow-y: auto; white-space: pre-wrap; word-break: break-all;
  display: none; margin-bottom: 10px;
}
.sim-file-link {
  display: inline-block; background: #0f172a; border: 1px solid #1d4ed8;
  border-radius: 6px; padding: 5px 14px; font-size: 11px; color: #93c5fd;
  text-decoration: none; transition: border-color .15s, color .15s;
}
.sim-file-link:hover { border-color: #60a5fa; color: #bfdbfe; }
#no-pkl-warning {
  color: #fca5a5; font-size: 11px; background: #450a0a;
  border-radius: 6px; padding: 8px 12px; margin-bottom: 10px; display: none;
  white-space: pre-wrap;
}
.sim-prev-title {
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .6px; color: #64748b; margin-bottom: 6px;
}
/* ---- simulation progress bar ---- */
#sim-progress-wrap { margin-bottom: 12px; }
.sim-prog-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; }
.sim-prog-label { font-size: 10px; color: #64748b; }
#sim-progress-text { font-size: 11px; color: #93c5fd; font-family: monospace; }
.sim-prog-track { background:#1e293b; border-radius:4px; overflow:hidden; height:8px; border:1px solid #334155; }
#sim-progress-bar { height:100%; background:linear-gradient(90deg,#1d4ed8,#3b82f6); width:0%; transition:width .15s ease; border-radius:4px; }
/* ---- feature chip tooltip custom cursor ---- */
label.feat-chip[title] { cursor: help; }
</style>
</head>
<body>

<h1>&#127936; March Madness Bracket Predictor</h1>
<p class="subtitle">Configure a model below, then run the evaluation across all historical years (2012–2025) plus a {{ THIS_YEAR }} prediction.</p>
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px;">
<a href="/bracket_input" style="display:inline-flex;align-items:center;gap:6px;background:#1e293b;border:1px solid #334155;border-radius:7px;padding:6px 14px;font-size:12px;color:#93c5fd;text-decoration:none;transition:border-color .15s;" onmouseover="this.style.borderColor='#3b82f6'" onmouseout="this.style.borderColor='#334155'">&#127942; Set Up {{ THIS_YEAR }} Bracket</a>
<a href="/fill_bracket" style="display:inline-flex;align-items:center;gap:6px;background:#1a1500;border:1px solid #78540a;border-radius:7px;padding:6px 14px;font-size:12px;color:#fbbf24;text-decoration:none;transition:border-color .15s;" onmouseover="this.style.borderColor='#ca8a04'" onmouseout="this.style.borderColor='#78540a'">&#10003; Fill Out My Bracket</a>
<a href="/my_brackets" style="display:inline-flex;align-items:center;gap:6px;background:#1e293b;border:1px solid #334155;border-radius:7px;padding:6px 14px;font-size:12px;color:#93c5fd;text-decoration:none;transition:border-color .15s;" onmouseover="this.style.borderColor='#3b82f6'" onmouseout="this.style.borderColor='#334155'">&#128196; My Brackets</a>
</div>

<!-- ===== SAVED MODELS ===== -->
<div class="panel-card saved-section">
  <div class="panel-title" style="display:flex;justify-content:space-between;align-items:center">
    <span>Existing Models</span>
    <span id="saved-count" style="color:#475569;font-size:11px"></span>
  </div>
  <div id="saved-empty" style="color:#475569;font-size:12px;display:none">No saved models found in Predictions/.</div>
  <table class="saved-table" id="saved-table" style="display:none">
    <thead><tr>
      <th style="width:30px">#</th>
      <th class="sortable" data-sort="name" onclick="savedSortBy('name')">Name</th>
      <th class="sortable" data-sort="model" onclick="savedSortBy('model')">Model</th>
      <th class="sortable" style="width:60px" data-sort="flags" onclick="savedSortBy('flags')">Flags</th>
      <th class="sortable" data-sort="features" onclick="savedSortBy('features')">Features</th>
      <th class="sortable" data-sort="params" onclick="savedSortBy('params')">Params</th>
      <th class="sortable sort-desc" style="width:70px;text-align:right" data-sort="score" onclick="savedSortBy('score')">Score</th>
      <th class="sortable" style="width:72px;text-align:right" data-sort="train_acc" onclick="savedSortBy('train_acc')">Train&nbsp;Acc</th>
      <th class="sortable" style="width:72px;text-align:right" data-sort="test_acc" onclick="savedSortBy('test_acc')">Test&nbsp;Acc</th>
    </tr></thead>
    <tbody id="saved-tbody"></tbody>
  </table>
</div>

<div class="layout">
  <!-- ===== FORM ===== -->
  <div class="form-panel">

    <label class="field-label">Model Type</label>
    <select id="model-select">
      {% for key, label in models %}
      <option value="{{ key }}" {% if key == 'logistic_regression' %}selected{% endif %}>{{ label }}</option>
      {% endfor %}
    </select>

    <label class="field-label">Model Parameters <span style="font-weight:400;text-transform:none">(optional)</span></label>
    <input type="text" id="params-input" placeholder="e.g. n_estimators=200 random_state=42 max_iter=1000">
    <p class="hint">Space-separated key=value pairs passed directly to the sklearn constructor.</p>

    <label class="field-label">Features</label>
    <div class="feat-legend">
      <span><span class="dot dot-common"></span> Basic</span>
      <span><span class="dot dot-kp"></span> KenPom</span>
      <span><span class="dot dot-bt"></span> BartTorvik / Stats</span>
      <span><span class="dot dot-bt2w"></span> 2-Week BartTorvik</span>
      <span><span class="dot dot-bthot"></span> Hotness (2W&minus;Season)</span>
    </div>

    <div class="feat-section">
      <div class="feat-section-title">Basic</div>
      <div class="feat-grid" id="feat-basic">
        {% for f in ui_basic_bases %}
        <label class="feat-chip {% if f in default_features %}selected{% endif %}" {% if feature_descs.get(f) %}title="{{ feature_descs[f] }}"{% endif %}>
          <input type="checkbox" value="{{ f }}" {% if f in default_features %}checked{% endif %}>
          {{ f }}
        </label>
        {% endfor %}
      </div>
    </div>

    <div class="feat-section">
      <div class="feat-section-title">KenPom</div>
      <div class="feat-grid" id="feat-kp">
        {% for f in ui_kp_bases %}
        <label class="feat-chip kp-only {% if f in default_features %}selected{% endif %}" {% if feature_descs.get(f) %}title="{{ feature_descs[f] }}"{% endif %}>
          <input type="checkbox" value="{{ f }}" {% if f in default_features %}checked{% endif %}>
          {{ f }}
        </label>
        {% endfor %}
      </div>
    </div>

    <div class="feat-section">
      <div class="feat-section-title">BartTorvik</div>
      <div class="feat-grid" id="feat-bt">
        {% for f in ui_bt_bases %}
        <label class="feat-chip bt-only {% if f in default_features %}selected{% endif %}" {% if feature_descs.get(f) %}title="{{ feature_descs[f] }}"{% endif %}>
          <input type="checkbox" value="{{ f }}" {% if f in default_features %}checked{% endif %}>
          {{ f }}
        </label>
        {% endfor %}
      </div>
    </div>

    <div class="feat-section">
      <div class="feat-section-title">Stats (BartTorvik)</div>
      <div class="feat-grid" id="feat-stats">
        {% for f in ui_stats_bases %}
        <label class="feat-chip bt-only {% if f in default_features %}selected{% endif %}" {% if feature_descs.get(f) %}title="{{ feature_descs[f] }}"{% endif %}>
          <input type="checkbox" value="{{ f }}" {% if f in default_features %}checked{% endif %}>
          {{ f }}
        </label>
        {% endfor %}
      </div>
    </div>

    <div class="feat-section">
      <div class="feat-section-title">2-Week BartTorvik</div>
      <div class="feat-grid" id="feat-bt2w">
        {% for f in ui_bt2w_bases %}
        <label class="feat-chip bt2w-only {% if f in default_features %}selected{% endif %}" {% if feature_descs.get(f) %}title="{{ feature_descs[f] }}"{% endif %}>
          <input type="checkbox" value="{{ f }}" {% if f in default_features %}checked{% endif %}>
          {{ f }}
        </label>
        {% endfor %}
      </div>
    </div>

    <div class="feat-section">
      <div class="feat-section-title">Hotness BartTorvik (2-Week minus Season)</div>
      <div class="feat-grid" id="feat-bthot">
        {% for f in ui_bthot_bases %}
        <label class="feat-chip bthot-only {% if f in default_features %}selected{% endif %}" {% if feature_descs.get(f) %}title="{{ feature_descs[f] }}"{% endif %}>
          <input type="checkbox" value="{{ f }}" {% if f in default_features %}checked{% endif %}>
          {{ f }}
        </label>
        {% endfor %}
      </div>
    </div>

    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;margin-top:4px">
      <input type="checkbox" id="norm-years-check" onchange="if(this.checked)document.getElementById('norm-all-check').checked=false" style="accent-color:#3b82f6;cursor:pointer;width:14px;height:14px">
      <span style="font-size:13px;color:#e2e8f0">Normalize within year</span>
      <span style="font-size:10px;color:#475569">(Z-score each feature per year before training)</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
      <input type="checkbox" id="norm-all-check" onchange="if(this.checked)document.getElementById('norm-years-check').checked=false" style="accent-color:#10b981;cursor:pointer;width:14px;height:14px" checked>
      <span style="font-size:13px;color:#e2e8f0">Normalize across all years</span>
      <span style="font-size:10px;color:#475569">(single global Z-score scaler across all years)</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      <input type="checkbox" id="calibrate-check" style="accent-color:#f97316;cursor:pointer;width:14px;height:14px">
      <span style="font-size:13px;color:#e2e8f0">Calibrate probabilities</span>
      <span style="font-size:10px;color:#475569">(Platt scaling &mdash; corrects over/under-confident win probs)</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      <input type="checkbox" id="delta-feats-check" style="accent-color:#a855f7;cursor:pointer;width:14px;height:14px" checked>
      <span style="font-size:13px;color:#e2e8f0">Delta features</span>
      <span style="font-size:10px;color:#475569">(collapse numeric __1 and __2 into a single team1 &minus; team2 difference)</span>
    </div>
    <hr style="border:none;border-top:1px solid #334155;margin:14px 0">
    <label class="field-label">Model Name <span style="color:#f87171">*</span></label>
    <input type="text" id="run-name-input"
      style="width:100%;background:#0f172a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;padding:8px 10px;font-size:13px;outline:none"
      placeholder="e.g. RF_AdjEM_NY  (required — used as Predictions folder name)">
    <p class="hint">This will be the folder name in <code style="color:#fbbf24">Predictions/</code>. Choose a unique, descriptive name.</p>
    <button id="run-btn" onclick="runPrediction()">&#9654; Run Prediction</button>
  </div>

  <!-- ===== OUTPUT ===== -->
  <div class="output-panel">

    <!-- Live log -->
    <div class="panel-card">
      <div class="panel-title" style="display:flex;justify-content:space-between;align-items:center">
        <span>Live Output</span>
        <span id="status-badge" class="status-badge status-idle">Idle</span>
      </div>
      <div id="log-box"></div>
    </div>

    <!-- Results (hidden until done) -->
    <div id="results-section">

      <div class="panel-card">
        <div class="panel-title">Bracket Results by Year</div>
        <div class="year-grid" id="year-links"></div>
      </div>

      <div class="panel-card">
        <div class="panel-title">Summary</div>
        <div id="summary-box"></div>
      </div>

      <!-- Save bracket to group -->
      <div class="panel-card" id="save-bracket-card" style="display:none">
        <div class="panel-title">&#128190; Save Bracket to Groups</div>
        <p style="font-size:12px;color:#64748b;margin-bottom:10px">
          Save this model&#39;s <strong style="color:#fbbf24" id="save-bracket-year-lbl"></strong> predicted bracket as a group bracket entry.
        </p>
        <div class="sim-form">
          <div>
            <label>Name</label>
            <input type="text" id="save-bracket-name"
              style="background:#0f172a;border:1px solid #334155;border-radius:5px;color:#e2e8f0;padding:5px 8px;font-size:12px;outline:none;width:320px"
              placeholder="bracket name">
          </div>
          <div>
            <label>Group</label>
            <input type="text" id="save-bracket-group"
              style="background:#0f172a;border:1px solid #334155;border-radius:5px;color:#e2e8f0;padding:5px 8px;font-size:12px;outline:none;width:120px"
              value="Models">
          </div>
          <button id="save-bracket-btn" onclick="saveModelBracket()">&#128190; Save</button>
        </div>
        <div id="save-bracket-msg" style="font-size:12px;margin-top:6px"></div>
      </div>
      <div class="panel-card" id="sim-card">
        <div class="panel-title" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <span>&#127922; Monte Carlo Simulation</span>
          <span id="sim-status-badge" class="status-badge" style="display:none"></span>
        </div>
        <div id="no-pkl-warning"></div>
        <div class="sim-form">
          <div><label>Year</label><select id="sim-year"></select></div>
          <div><label>Iterations</label><input type="number" id="sim-iters" value="1000" min="100" max="100000"></div>
          <div><label>Seed (opt.)</label><input type="number" id="sim-seed" placeholder="random"></div>
          <button id="sim-btn" onclick="runSimulation()">&#9654; Simulate</button>
        </div>
        <div id="sim-progress-wrap" style="display:none">
          <div class="sim-prog-header">
            <span class="sim-prog-label">Simulating&hellip;</span>
            <span id="sim-progress-text">0 / 0</span>
          </div>
          <div class="sim-prog-track"><div id="sim-progress-bar"></div></div>
        </div>
        <div id="sim-log"></div>
        <div id="sim-prev" style="display:none">
          <div class="sim-prev-title">Previous Simulations</div>
          <div id="sim-links" class="year-grid"></div>
        </div>
      </div>

    </div>
  </div>
</div>

<script>
const ALL_YEARS = {{ ALL_YEARS_JSON }};

// ---- Saved models ----
let activeSavedRow = null;
let currentDirName = null;
let savedModels    = [];
let savedSort      = { key: 'score', dir: -1 };

function savedSortVal(m, key) {
  if (key === 'name')      return (m.dir_name  || '').toLowerCase();
  if (key === 'model')     return (m.model     || '').toLowerCase();
  if (key === 'flags')     return (m.expert    || '').toLowerCase();
  if (key === 'features')  return (m.features  || '').toLowerCase();
  if (key === 'params')    return (m.params    || '').toLowerCase();
  if (key === 'score')     return m.score     != null ? m.score     : -Infinity;
  if (key === 'train_acc') return m.train_acc != null ? m.train_acc : -Infinity;
  if (key === 'test_acc')  return m.test_acc  != null ? m.test_acc  : -Infinity;
  return '';
}

function savedSortBy(key) {
  if (savedSort.key === key) { savedSort.dir *= -1; }
  else { savedSort.key = key; savedSort.dir = (key === 'score' || key === 'train_acc' || key === 'test_acc') ? -1 : 1; }
  renderSavedModels();
}

function renderSavedModels() {
  const sorted = savedModels.slice().sort(function(a, b) {
    const av = savedSortVal(a, savedSort.key), bv = savedSortVal(b, savedSort.key);
    if (av < bv) return savedSort.dir; if (av > bv) return -savedSort.dir; return 0;
  });
  document.querySelectorAll('#saved-table th[data-sort]').forEach(function(th) {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.sort === savedSort.key) th.classList.add(savedSort.dir === 1 ? 'sort-asc' : 'sort-desc');
  });
  const tbody = document.getElementById('saved-tbody');
  tbody.innerHTML = '';
  const fmtAcc = function(v) { return v == null ? '\u2014' : (v * 100).toFixed(1) + '%'; };
  sorted.forEach(function(m, i) {
    const tr = document.createElement('tr');
    tr.className = 'model-row';
    const nyTag  = m.norm_years  ? '<span class="tag tag-ny">NY</span>'   : '';
    const naTag  = m.norm_all    ? '<span class="tag tag-na">NA</span>'   : '';
    const calTag = m.calibrated  ? '<span class="tag tag-cal">CAL</span>' : '';
    const dfTag  = m.delta_feats ? '<span class="tag" style="background:#a855f7;color:#fff">DF</span>' : '';
    const flagsTag = (nyTag + naTag + calTag + dfTag) || '\u2014';
    tr.innerHTML =
      '<td style="color:#475569">' + (i + 1) + '</td>' +
      '<td style="color:#e2e8f0;font-weight:600;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + m.dir_name.replace(/"/g,'&quot;') + '">' + m.dir_name + '</td>' +
      '<td>' + m.model.replace(/_/g, '\u00a0') + '</td>' +
      '<td>' + flagsTag + '</td>' +
      '<td class="feat-tags">' + m.features.replace(/\+/g, ' &middot; ') + '</td>' +
      '<td style="color:#64748b;font-size:10px">' + (m.params || '\u2014') + '</td>' +
      '<td class="score-cell" style="text-align:right">' + m.score + '</td>' +
      '<td style="color:#94a3b8;font-size:11px;text-align:right;font-family:monospace">' + fmtAcc(m.train_acc) + '</td>' +
      '<td style="color:#94a3b8;font-size:11px;text-align:right;font-family:monospace">' + fmtAcc(m.test_acc)  + '</td>';
    tr.addEventListener('click', function() {
      if (activeSavedRow) activeSavedRow.classList.remove('active');
      tr.classList.add('active');
      activeSavedRow = tr;
      loadSavedResults(m.dir_name);
    });
    tbody.appendChild(tr);
  });
}

function loadSavedModels() {
  fetch('/saved_models')
    .then(r => r.json())
    .then(function(models) {
      const table   = document.getElementById('saved-table');
      const empty   = document.getElementById('saved-empty');
      const countEl = document.getElementById('saved-count');
      if (!models.length) { empty.style.display = ''; return; }
      savedModels = models;
      countEl.textContent = models.length + ' model' + (models.length !== 1 ? 's' : '');
      table.style.display = '';
      empty.style.display = 'none';
      renderSavedModels();
    });
}

function loadSavedResults(dirName) {
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('log-box').innerHTML = '';
  setStatus('Loading\u2026', 'status-running');
  fetch('/saved_results/' + dirName)
    .then(r => r.json())
    .then(function(data) {
      setStatus('Done', 'status-done');
      document.getElementById('summary-box').textContent = data.summary || '(no summary)';
      const grid = document.getElementById('year-links');
      grid.innerHTML = '';
      (data.years || []).forEach(function(y) {
        const a = document.createElement('a');
        a.href = '/saved_bracket/' + dirName + '/' + y;
        a.target = '_blank';
        a.textContent = y === 2026 ? y + ' \u2605' : String(y);
        a.className = 'year-btn' + (y === 2026 ? ' current' : '');
        grid.appendChild(a);
      });
      document.getElementById('results-section').style.display = 'block';
      loadSimCard(dirName);
    });
}

loadSavedModels();

// ---- Simulation ----
function loadSimCard(dirName) {
  currentDirName = dirName;
  // Reset state
  document.getElementById('no-pkl-warning').style.display = 'none';
  document.getElementById('sim-log').style.display = 'none';
  document.getElementById('sim-log').innerHTML = '';
  document.getElementById('sim-status-badge').style.display = 'none';
  document.getElementById('sim-btn').disabled = false;

  // Show and populate the save-bracket card
  const saveCard = document.getElementById('save-bracket-card');
  if (saveCard) {
    document.getElementById('save-bracket-name').value = dirName;
    document.getElementById('save-bracket-year-lbl').textContent = {{ THIS_YEAR }} + ' \u2605';
    document.getElementById('save-bracket-msg').textContent = '';
    document.getElementById('save-bracket-btn').disabled = false;
    saveCard.style.display = '';
  }

  // Populate year selector from ALL_YEARS
  const sel = document.getElementById('sim-year');
  sel.innerHTML = '';
  ALL_YEARS.forEach(function(y) {
    const opt = document.createElement('option');
    opt.value = y;
    opt.textContent = y === {{ THIS_YEAR }} ? y + ' \u2605' : String(y);
    if (y === {{ THIS_YEAR }}) opt.selected = true;
    sel.appendChild(opt);
  });

  // Load previous simulation runs
  loadSimPrev(dirName);
}

function saveModelBracket() {
  const name  = document.getElementById('save-bracket-name').value.trim();
  const group = document.getElementById('save-bracket-group').value.trim();
  const msg   = document.getElementById('save-bracket-msg');
  const btn   = document.getElementById('save-bracket-btn');
  if (!name)  { msg.style.color = '#f87171'; msg.textContent = '\u26a0 Enter a bracket name.'; return; }
  if (!group) { msg.style.color = '#f87171'; msg.textContent = '\u26a0 Enter a group name.'; return; }
  if (!currentDirName) return;
  btn.disabled = true;
  msg.style.color = '#94a3b8';
  msg.textContent = 'Saving\u2026';
  fetch('/save_model_bracket', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dir_name: currentDirName, year: {{ THIS_YEAR }}, name, group }),
  })
  .then(r => r.json())
  .then(data => {
    btn.disabled = false;
    if (data.error) {
      msg.style.color = '#f87171';
      msg.textContent = '\u26a0 ' + data.error;
    } else {
      msg.style.color = '#4ade80';
      msg.textContent = '\u2713 Saved to ' + data.path;
    }
  })
  .catch(() => {
    btn.disabled = false;
    msg.style.color = '#f87171';
    msg.textContent = '\u26a0 Failed to save.';
  });
}

function loadSimPrev(dirName) {
  fetch('/sim_list/' + encodeURIComponent(dirName))
    .then(r => r.json())
    .then(function(data) {
      const links = document.getElementById('sim-links');
      const prev  = document.getElementById('sim-prev');
      links.innerHTML = '';
      if (data.files && data.files.length) {
        data.files.forEach(function(fname) {
          const a = document.createElement('a');
          a.href = '/sim_html/' + encodeURIComponent(dirName) + '/' + fname;
          a.target = '_blank';
          a.textContent = fname.replace('_', ' ').replace('iters.html', ' iters');
          a.className = 'sim-file-link';
          links.appendChild(a);
        });
        prev.style.display = '';
      } else {
        prev.style.display = 'none';
      }
    });
}

function runSimulation() {
  if (!currentDirName) return;
  const year     = parseInt(document.getElementById('sim-year').value);
  const numIters = parseInt(document.getElementById('sim-iters').value) || 1000;
  const seedRaw  = document.getElementById('sim-seed').value.trim();
  const seed     = seedRaw !== '' ? parseInt(seedRaw) : null;

  document.getElementById('sim-btn').disabled = true;
  document.getElementById('no-pkl-warning').style.display = 'none';
  const simLog = document.getElementById('sim-log');
  simLog.innerHTML = '';
  simLog.style.display = '';
  const badge = document.getElementById('sim-status-badge');
  badge.style.display = '';
  badge.className = 'status-badge status-running';
  badge.textContent = 'Running\u2026';
  // Reset + show progress bar
  const progWrap = document.getElementById('sim-progress-wrap');
  document.getElementById('sim-progress-bar').style.width = '0%';
  document.getElementById('sim-progress-text').textContent = '0 / ' + numIters.toLocaleString();
  progWrap.style.display = '';

  fetch('/simulate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ dir_name: currentDirName, year, num_iters: numIters, seed }),
  })
  .then(r => r.json())
  .then(function(data) {
    if (data.error) {
      const warn = document.getElementById('no-pkl-warning');
      warn.textContent = data.error;
      warn.style.display = '';
      simLog.style.display = 'none';
      badge.style.display = 'none';
      document.getElementById('sim-btn').disabled = false;
      return;
    }
    startSimStream(data.job_id);
  });
}

function startSimStream(jobId) {
  const simLog   = document.getElementById('sim-log');
  const badge    = document.getElementById('sim-status-badge');
  const progBar  = document.getElementById('sim-progress-bar');
  const progTxt  = document.getElementById('sim-progress-text');
  const progWrap = document.getElementById('sim-progress-wrap');
  const es = new EventSource('/sim_stream/' + jobId);

  es.onmessage = function(e) {
    const msg = JSON.parse(e.data);
    if (msg.type === 'line') {
      // Intercept PROGRESS: lines for the bar — don't show in log
      if (msg.text.startsWith('PROGRESS:')) {
        const parts = msg.text.slice(9).split('/');
        const done  = parseInt(parts[0]);
        const total = parseInt(parts[1]);
        const pct   = total > 0 ? (done / total * 100) : 0;
        progBar.style.width = pct + '%';
        progTxt.textContent = done.toLocaleString() + ' / ' + total.toLocaleString();
        return;
      }
      simLog.textContent += msg.text + '\n';
      simLog.scrollTop = simLog.scrollHeight;
      return;
    }
    if (msg.type === 'done') {
      es.close();
      progWrap.style.display = 'none';
      document.getElementById('sim-btn').disabled = false;
      if (msg.status === 'done') {
        badge.className = 'status-badge status-done';
        badge.textContent = 'Done';
        loadSimPrev(currentDirName);
      } else {
        badge.className = 'status-badge status-error';
        badge.textContent = 'Error';
      }
      return;
    }
    if (msg.type === 'timeout') {
      es.close();
      progWrap.style.display = 'none';
      badge.className = 'status-badge status-error';
      badge.textContent = 'Timeout';
      document.getElementById('sim-btn').disabled = false;
    }
  };
  es.onerror = function() {
    es.close();
    progWrap.style.display = 'none';
    badge.className = 'status-badge status-error';
    badge.textContent = 'Disconnected';
    document.getElementById('sim-btn').disabled = false;
  };
}

// Sync chip appearance with checkbox state on every change
document.querySelectorAll('.feat-chip').forEach(chip => {
  chip.querySelector('input[type="checkbox"]').addEventListener('change', function() {
    chip.classList.toggle('selected', this.checked);
  });
});

function getSelectedFeatures() {
  return [...document.querySelectorAll('.feat-chip input:checked')].map(cb => cb.value);
}

function setStatus(text, cls) {
  const b = document.getElementById('status-badge');
  b.textContent = text;
  b.className = 'status-badge ' + cls;
}

function classifyLine(text) {
  if (/^={3,}/.test(text) || /^-{3,}/.test(text)) return 'log-line-eq';
  if (/^\d{4}$/.test(text.trim())) return 'log-line-year';
  if (/Round \d/.test(text)) return 'log-line-round';
  if (/Model type|Model trained|Train acc|Test acc/.test(text)) return 'log-line-model';
  if (/Year total|Avg bracket|correct|Score/.test(text)) return 'log-line-result';
  if (/\[ERROR\]|Error|Traceback/.test(text)) return 'log-line-err';
  return '';
}

let currentJobId = null;

function runPrediction() {
  const features = getSelectedFeatures();
  if (!features.length) { alert('Select at least one feature.'); return; }

  const runName   = document.getElementById('run-name-input').value.trim();
  if (!runName) {
    document.getElementById('run-name-input').focus();
    document.getElementById('run-name-input').style.borderColor = '#f87171';
    alert('Model Name is required before running.');
    return;
  }
  document.getElementById('run-name-input').style.borderColor = '#334155';

  const model     = document.getElementById('model-select').value;
  const params    = document.getElementById('params-input').value.trim();
  const normYears = document.getElementById('norm-years-check').checked;
  const normAll   = document.getElementById('norm-all-check').checked;
  const calibrate  = document.getElementById('calibrate-check').checked;
  const deltaFeats = document.getElementById('delta-feats-check').checked;

  // Reset UI
  document.getElementById('log-box').innerHTML = '';
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('run-btn').disabled = true;
  setStatus('Running…', 'status-running');

  fetch('/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ run_name: runName, model, params, features, norm_years: normYears, norm_all: normAll, calibrate, delta_feats: deltaFeats }),
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { alert(data.error); document.getElementById('run-btn').disabled = false; return; }
    currentJobId = data.job_id;
    startStream(data.job_id);
  });
}

function startStream(jobId) {
  const logBox = document.getElementById('log-box');
  const es = new EventSource(`/stream/${jobId}`);

  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === 'line') {
      const span = document.createElement('span');
      const cls = classifyLine(msg.text);
      if (cls) span.className = cls;
      span.textContent = msg.text + '\n';
      logBox.appendChild(span);
      logBox.scrollTop = logBox.scrollHeight;
      return;
    }

    if (msg.type === 'done') {
      es.close();
      if (msg.status === 'done') {
        setStatus('Done', 'status-done');
        fetchResults(jobId);
      } else {
        setStatus('Error', 'status-error');
        document.getElementById('run-btn').disabled = false;
      }
      return;
    }

    if (msg.type === 'timeout') {
      es.close();
      setStatus('Timeout', 'status-error');
      document.getElementById('run-btn').disabled = false;
    }
  };

  es.onerror = () => {
    es.close();
    setStatus('Disconnected', 'status-error');
    document.getElementById('run-btn').disabled = false;
  };
}

function fetchResults(jobId) {
  fetch(`/results/${jobId}`)
    .then(r => r.json())
    .then(data => {
      // Summary
      document.getElementById('summary-box').textContent = data.summary || '(no summary)';

      // Year links
      const grid = document.getElementById('year-links');
      grid.innerHTML = '';
      (data.years || []).forEach(y => {
        const a = document.createElement('a');
        a.href = `/bracket/${jobId}/${y}`;
        a.target = '_blank';
        a.textContent = y === {{ THIS_YEAR }} ? `${y} ★` : String(y);
        a.className = 'year-btn' + (y === {{ THIS_YEAR }} ? ' current' : '');
        grid.appendChild(a);
      });

      document.getElementById('results-section').style.display = 'block';
      document.getElementById('run-btn').disabled = false;

      // Load simulation card for this model's output dir
      if (data.dir_name) loadSimCard(data.dir_name);
    });
}
</script>
</body>
</html>
"""

# Inject THIS_YEAR and ALL_YEARS into the template
INDEX_HTML = INDEX_HTML.replace('{{ THIS_YEAR }}', str(THIS_YEAR))
INDEX_HTML = INDEX_HTML.replace('{{ ALL_YEARS_JSON }}', json.dumps(ALL_YEARS))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'Starting server at http://localhost:5050')
    print(f'Repo root: {REPO_ROOT}')
    app.run(debug=False, host='0.0.0.0', port=5050, threaded=True)
