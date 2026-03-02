"""
bracket_html.py

Renders bracket simulation results as a self-contained HTML file and provides
a plain-text CSV formatter for archival output.

Public API:
    format_bracket_html(...)  -> str   # full HTML page
    format_pred_file(...)     -> str   # legacy plain-text CSV string
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_SH = 30   # px per team slot
_CH = 22   # px card height


# ---------------------------------------------------------------------------
# Private HTML primitives
# ---------------------------------------------------------------------------

def _fmt_seed(s) -> str:
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return '?'


def _get_c(correct_by_round, rnd, idx, is_current):
    """Safely retrieve correct_by_round[rnd][idx]; returns None when unavailable."""
    if is_current:
        return None
    lst = correct_by_round[rnd] if rnd < len(correct_by_round) else []
    return lst[idx] if idx < len(lst) else None


def _card(team, seed, prob=None, correct=None, is_winner=True, extra_cls='') -> str:
    """Return an HTML div for one team card."""
    s = _fmt_seed(seed)
    p_html = f'<span class="p"> {prob:.0%}</span>' if (prob is not None and prob == prob) else ''
    mark = ''
    if is_winner:
        if correct is None:
            cls = 'adv'
        elif bool(correct):
            cls = 'ok'
            mark = ' ✓'
        else:
            cls = 'ng'
            mark = ' ✗'
    else:
        cls = 'out'
    return (f'<div class="c {cls} {extra_cls}">'
            f'<span class="s">[{s}]</span>{team}{p_html}{mark}</div>')


def _place(top_px: int, html: str) -> str:
    return f'<div style="position:absolute;top:{top_px}px;left:0;right:0">{html}</div>'


def _col(items: list, label: str, width: int, height: int) -> str:
    inner = ''.join(items)
    return (f'<div class="rnd" style="width:{width}px;height:{height}px">'
            f'<div class="rl">{label}</div>{inner}</div>')


# ---------------------------------------------------------------------------
# Public: HTML bracket
# ---------------------------------------------------------------------------

def format_bracket_html(
    data_root: Path,
    year: int,
    pred_teams_by_round: list,
    pred_seeds_by_round: list,
    pred_probs_by_round: list,
    correct_by_round: list,
    num_correct_by_round: list,
    total_score: int,
    is_current: bool,
    model_key: str,
    feat_bases: List[str],
    ff_pairings: List[Tuple[int, int]],
) -> str:
    """
    Generate a self-contained HTML file showing the full 64-team bracket.

    Layout:  [Left half: R1→R2→R3→R4]  [Center: FF+Champ+FF]  [Right half: R4←R3←R2←R1]
    Rounds 1–4 show teams from both left regions (top/bottom).
    Right half mirrors the left.  Final Four and Championship are in the center.
    Green = correct prediction, red = wrong, blue = current-year pick.
    """
    REGION_SLOTS = 16   # 8 matchups × 2 teams per region
    HALF_SLOTS   = 32   # 2 regions per half
    HALF_H       = HALF_SLOTS * _SH

    # Load all 64 teams from Round 1 -----------------------------------------
    df_r1 = pd.read_csv(
        data_root / 'Data' / 'BracketCombinedData' / str(year) / f'Round1_{year}.csv'
    )
    r1_games = []
    for _, row in df_r1.iterrows():
        actual = None
        if (not is_current and 'Winning_Team' in row.index
                and pd.notna(row['Winning_Team'])):
            actual = str(row['Winning_Team'])
        r1_games.append({
            't1': str(row['Team__1']), 's1': row['Seed__1'],
            't2': str(row['Team__2']), 's2': row['Seed__2'],
            'actual': actual,
        })

    def gc(rnd, idx):
        return _get_c(correct_by_round, rnd, idx, is_current)

    # Vertical-position helpers ------------------------------------------------
    def region_top(r, k):
        return ((1 << (r + 1)) * k + (1 << r) - 1) * _SH

    def half_top(r, region_i, k):
        return region_i * REGION_SLOTS * _SH + region_top(r, k)

    # Build one half (left or right) -------------------------------------------
    def build_half(game_off, r1_off, r2_off, r3_off, r4_off, is_left):
        """
        game_off : index of first R1 game in r1_games for this half (0 or 16)
        rX_off   : slice start in pred_teams_by_round[X-1] for this half
        is_left  : if True, column order R1→R4; if False, R4→R1 (mirrored)
        """
        # R1 – show both teams for each of 16 games
        r1_items = []
        for lg in range(16):
            g      = r1_games[game_off + lg]
            pred_w = pred_teams_by_round[0][r1_off + lg]
            prob   = pred_probs_by_round[0][r1_off + lg]
            corr   = gc(0, r1_off + lg)
            t1_top = lg * 2 * _SH
            t2_top = (lg * 2 + 1) * _SH
            is_w1  = (g['t1'] == pred_w)
            is_w2  = (g['t2'] == pred_w)
            r1_items.append(_place(t1_top, _card(
                g['t1'], g['s1'],
                prob if is_w1 else None,
                corr if is_w1 else None, is_w1)))
            r1_items.append(_place(t2_top, _card(
                g['t2'], g['s2'],
                prob if is_w2 else None,
                corr if is_w2 else None, is_w2)))

        # R2 – 8 advancing teams per half (4 per region)
        r2_items = []
        for ri in range(2):
            for k in range(4):
                idx = r2_off + ri * 4 + k
                if idx >= len(pred_teams_by_round[1]):
                    continue
                top = half_top(1, ri, k)
                r2_items.append(_place(top, _card(
                    pred_teams_by_round[1][idx], pred_seeds_by_round[1][idx],
                    pred_probs_by_round[1][idx], gc(1, idx))))

        # R3 – 4 advancing teams per half (2 per region)
        r3_items = []
        for ri in range(2):
            for k in range(2):
                idx = r3_off + ri * 2 + k
                if idx >= len(pred_teams_by_round[2]):
                    continue
                top = half_top(2, ri, k)
                r3_items.append(_place(top, _card(
                    pred_teams_by_round[2][idx], pred_seeds_by_round[2][idx],
                    pred_probs_by_round[2][idx], gc(2, idx))))

        # R4 – 2 advancing teams per half (1 per region)
        r4_items = []
        for ri in range(2):
            idx = r4_off + ri
            if idx >= len(pred_teams_by_round[3]):
                continue
            top = half_top(3, ri, 0)
            r4_items.append(_place(top, _card(
                pred_teams_by_round[3][idx], pred_seeds_by_round[3][idx],
                pred_probs_by_round[3][idx], gc(3, idx))))

        cols = [
            _col(r1_items, 'First Round',   200, HALF_H),
            _col(r2_items, 'Second Round',  170, HALF_H),
            _col(r3_items, 'Sweet 16',      160, HALF_H),
            _col(r4_items, 'Elite Eight',   155, HALF_H),
        ]
        return cols if is_left else cols[::-1]

    left_cols  = build_half(0,  0,  0, 0, 0, is_left=True)
    right_cols = build_half(16, 16, 8, 4, 2, is_left=False)

    # Build center columns (Final Four + Championship) -------------------------
    FF_WIN_TOP = 15 * _SH   # where FF winner card sits in FF columns
    CHAMP_TOP  = 15 * _SH   # champion card in championship column

    # Left FF column — show only the FF winner (E8 winners already shown in E8 col)
    ff_left = []
    if len(pred_teams_by_round[4]) > 0:
        ff_left.append(_place(FF_WIN_TOP, _card(
            pred_teams_by_round[4][0], pred_seeds_by_round[4][0],
            pred_probs_by_round[4][0], gc(4, 0))))

    # Right FF column — show only the FF winner
    ff_right = []
    if len(pred_teams_by_round[4]) > 1:
        ff_right.append(_place(FF_WIN_TOP, _card(
            pred_teams_by_round[4][1], pred_seeds_by_round[4][1],
            pred_probs_by_round[4][1], gc(4, 1))))

    # Championship column — show only the champion card (finalists are already
    # visible as the advancing winner cards in ff_left / ff_right).
    champ_col = []
    if len(pred_teams_by_round[5]) > 0:
        champ_col.append(_place(CHAMP_TOP, _card(
            pred_teams_by_round[5][0], pred_seeds_by_round[5][0],
            pred_probs_by_round[5][0], gc(5, 0), extra_cls='champ')))

    center_cols = [
        _col(ff_left,   'Final Four',    160, HALF_H),
        _col(champ_col, 'Championship',  185, HALF_H),
        _col(ff_right,  'Final Four',    160, HALF_H),
    ]

    # Header info --------------------------------------------------------------
    feat_str = ' · '.join(feat_bases)
    if not is_current:
        n_total = sum(num_correct_by_round)
        rnd_detail = ' &nbsp; '.join(
            f'R{i + 1}: {num_correct_by_round[i]}/{[32, 16, 8, 4, 2, 1][i]}'
            for i in range(6))
        score_html = (
            f'<div class="score-bar">'
            f'Score: <strong>{total_score}</strong> pts &nbsp;|&nbsp; '
            f'Correct: <strong>{n_total}</strong>/63 &nbsp;|&nbsp; {rnd_detail}'
            f'</div>')
    else:
        score_html = '<div class="score-bar">Current-year prediction — no score available yet.</div>'

    # Assemble -----------------------------------------------------------------
    all_cols = ''.join(left_cols + center_cols + right_cols)

    return f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{year} NCAA Bracket \u2013 {model_key}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Segoe UI",Arial,sans-serif;font-size:11px;background:#111827;color:#e5e7eb;padding:14px;min-width:max-content}}
h1{{font-size:17px;color:#fbbf24;margin-bottom:3px}}
.meta{{color:#6b7280;font-size:11px;margin-bottom:6px}}
.score-bar{{font-size:12px;color:#9ca3af;margin-bottom:14px}}
.score-bar strong{{color:#f9fafb}}
.bracket{{display:flex;flex-direction:row;align-items:flex-start;gap:0}}
.rnd{{flex-shrink:0;border-right:1px solid #1f2937;padding:0 3px;position:relative}}
.rl{{font-size:9px;color:#4b5563;text-transform:uppercase;letter-spacing:.6px;text-align:center;padding-bottom:3px}}
.c{{height:{_CH}px;line-height:{_CH}px;padding:0 5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-radius:3px;font-size:11px;border-left:3px solid transparent}}
.out{{background:#1f2937;color:#4b5563;border-left-color:#374151}}
.adv{{background:#1e3a5f;color:#93c5fd;border-left-color:#3b82f6}}
.ok {{background:#14432a;color:#86efac;border-left-color:#22c55e}}
.ng {{background:#450a0a;color:#fca5a5;border-left-color:#ef4444}}
.champ{{background:#451a03;color:#fde68a;border:2px solid #f59e0b;font-weight:700}}
.ok.champ{{background:#14432a;color:#86efac;border:2px solid #22c55e;font-weight:700}}
.ng.champ{{background:#450a0a;color:#fca5a5;border:2px solid #ef4444;font-weight:700}}
.s{{color:#6b7280;font-size:10px;margin-right:2px}}
.p{{color:#f59e0b;font-size:9px}}
</style></head>
<body>
<h1>{year} NCAA Tournament \u2014 {model_key}</h1>
<div class="meta">Features: {feat_str}</div>
{score_html}
<div class="bracket">{all_cols}</div>
</body></html>'''


# ---------------------------------------------------------------------------
# Public: plain-text CSV formatter (legacy / archival)
# ---------------------------------------------------------------------------

def format_pred_file(
    pred_teams_by_round,
    pred_seeds_by_round,
    pred_probs_by_round,
    correct_by_round,
    num_correct_by_round,
    total_score,
    is_current: bool,
) -> str:
    """
    Produce a compact plain-text representation of bracket predictions.
    Each line corresponds to one round; advancing teams are comma-separated.
    """
    lines = []
    num_correct_total = 0
    for rnd_idx in range(6):
        parts = []
        n = len(pred_teams_by_round[rnd_idx])
        if not is_current:
            n_cor = num_correct_by_round[rnd_idx]
            round_score = n_cor * (2 ** rnd_idx) * 10
            num_correct_total += n_cor
            parts.append(f'{n_cor} for {n}')
            parts.append(str(round_score))
        for j in range(n):
            prob = pred_probs_by_round[rnd_idx][j]
            prob_tag = f'({prob:.2f})' if prob is not None else ''
            entry = f'[{pred_seeds_by_round[rnd_idx][j]}]{pred_teams_by_round[rnd_idx][j]}{prob_tag}'
            if not is_current:
                entry += f'({int(correct_by_round[rnd_idx][j])})'
            parts.append(entry)
        lines.append(','.join(parts))
    result = '\n'.join(lines)
    if not is_current:
        result += f'\n{num_correct_total} for 63,{total_score}'
    return result
