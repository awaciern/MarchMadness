#!/usr/bin/env python3
"""
generate_sim_data.py
--------------------
Generate simulated training data from real tournament game rows.
Three generation methods are supported:

  noise     (original)
            Add independent Gaussian noise to Score__1 and Score__2 and
            recompute the winner.  Simple, but perturbs total game scoring
            (which is driven by tempo, a real feature) along with the
            competitive margin.

  margin    (new)
            Perturb only the point spread while keeping the total game score
            fixed.  More physically meaningful: pace/tempo determines how
            many total points are scored; the competitive uncertainty should
            be a perturbation of the margin alone.
              margin_sim = (s1 - s2) + N(0, sigma)
              s1_sim = (s_total + margin_sim) / 2
              s2_sim = (s_total - margin_sim) / 2

  logistic  (new)
            Sample win/loss outcomes directly from a win probability that is
            grounded in team-quality features -- no score corruption at all.
            Uses the Bradley-Terry model applied to BartTorvik Barthag ratings:
              p_win1 = odds1 / (odds1 + odds2),  odds_i = Barthag_i / (1-Barthag_i)
            Then samples Win__1 ~ Bernoulli(p_win1) independently for each
            simulated row.  Synthetic scores that are consistent with the
            sampled outcome are generated from KenPom AdjO / AdjD / AdjT so
            that score columns remain plausible (needed if downstream code reads
            scores, though they are not model features):
              tempo   = (AdjT1 + AdjT2) / 2
              s1_exp  = AdjO1 * tempo / 100
              s2_exp  = AdjO2 * tempo / 100
            A correlated noise term (Bivariate Normal with rho=0.5) is added
            to reflect the real positive correlation between the two teams'
            scores.  If the noisy scores contradict the sampled outcome, the
            margin sign is corrected without re-drawing.

Output is written to:
    Data/SimulatedData<identifier>/All.csv

Usage examples:
    # margin method, std=8, 15 copies per game
    python3 Python/generate_sim_data.py --method margin --identifier Margin8 --std 8 --n 15

    # logistic method (Barthag-grounded), 20 copies per game
    python3 Python/generate_sim_data.py --method logistic --identifier BT20 --n 20

    # original Gaussian noise
    python3 Python/generate_sim_data.py --method noise --identifier Noise5 --std 5 --n 15

    # dry run
    python3 Python/generate_sim_data.py --method logistic --identifier BT20 --n 20 --dry-run
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Generate simulated tournament game data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--method', default='noise',
        choices=['noise', 'margin', 'logistic'],
        help=(
            'Simulation method: '
            '"noise" = independent Gaussian score noise (original); '
            '"margin" = margin-only perturbation, total score preserved; '
            '"logistic" = feature-grounded Bernoulli outcome sampling via Barthag.'
        ),
    )
    p.add_argument(
        '--identifier', required=True,
        help='Unique name for this dataset (e.g. "Margin8"). '
             'Output is written to Data/SimulatedData<identifier>/All.csv.',
    )
    p.add_argument(
        '--std', type=float, default=None,
        help='Standard deviation of the perturbation noise.  Required for '
             '"noise" and "margin" methods.  Not used for "logistic".',
    )
    p.add_argument(
        '--n', type=int, required=True,
        help='Number of simulated rows to produce per real game row.',
    )
    p.add_argument(
        '--score-std', type=float, default=6.0,
        help='(logistic only) Std-dev of per-team score noise around the '
             'KenPom-expected score.  Default: 6.0 points.',
    )
    p.add_argument(
        '--source', default=None,
        help='Explicit path to source All.csv. '
             'Defaults to <data-root>/Data/GameCombinedData/All.csv.',
    )
    p.add_argument(
        '--data-root', default=None,
        help='Path to repo root (directory containing Data/). '
             'Inferred from this script\'s location if omitted.',
    )
    p.add_argument(
        '--seed', type=int, default=42,
        help='NumPy random seed for reproducibility (default: 42).',
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help='Print stats without writing any files.',
    )
    return p


# ---------------------------------------------------------------------------
# Method 1 (original): independent Gaussian score noise
# ---------------------------------------------------------------------------

def generate_noise(
    df: pd.DataFrame,
    std: float,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add independent N(0,sigma) noise to Score__1 and Score__2."""
    total = len(df)
    noise1 = rng.normal(0.0, std, size=(total, n)).reshape(-1)
    noise2 = rng.normal(0.0, std, size=(total, n)).reshape(-1)

    idx_rep   = np.repeat(np.arange(total), n)
    sim_df    = df.iloc[idx_rep].copy().reset_index(drop=True)

    s1_base   = pd.to_numeric(df['Score__1'], errors='coerce').fillna(0).values
    s2_base   = pd.to_numeric(df['Score__2'], errors='coerce').fillna(0).values
    s1_sim    = np.repeat(s1_base, n) + noise1
    s2_sim    = np.repeat(s2_base, n) + noise2
    t1_names  = df['Team__1'].values
    t2_names  = df['Team__2'].values

    sim_df['Score__1']     = s1_sim
    sim_df['Score__2']     = s2_sim
    team1_wins             = s1_sim > s2_sim
    sim_df['Win__1']       = team1_wins
    sim_df['Winning_Team'] = np.where(
        team1_wins,
        np.repeat(t1_names, n),
        np.repeat(t2_names, n),
    )
    return sim_df


# ---------------------------------------------------------------------------
# Method 2 (new): margin-only perturbation
# ---------------------------------------------------------------------------

def generate_margin(
    df: pd.DataFrame,
    std: float,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Perturb the point spread by N(0,sigma) while holding the total game score fixed.

    Outcome:
        margin_sim = (s1 - s2) + N(0, sigma)
        s1_sim = (s_total + margin_sim) / 2
        s2_sim = (s_total - margin_sim) / 2

    This is more physically grounded than independent score noise because:
    - Total points scored is determined by tempo (a real feature), not luck.
    - Only the competitive margin -- the quantity that determines who wins --
      is perturbed.
    """
    total    = len(df)
    noise    = rng.normal(0.0, std, size=(total, n)).reshape(-1)

    idx_rep  = np.repeat(np.arange(total), n)
    sim_df   = df.iloc[idx_rep].copy().reset_index(drop=True)

    s1_base  = pd.to_numeric(df['Score__1'], errors='coerce').fillna(0).values
    s2_base  = pd.to_numeric(df['Score__2'], errors='coerce').fillna(0).values
    s_total  = s1_base + s2_base           # preserved across simulations
    margin   = s1_base - s2_base

    margin_sim  = np.repeat(margin, n) + noise
    total_rep   = np.repeat(s_total, n)
    s1_sim      = (total_rep + margin_sim) / 2.0
    s2_sim      = (total_rep - margin_sim) / 2.0
    t1_names    = df['Team__1'].values
    t2_names    = df['Team__2'].values

    sim_df['Score__1']     = s1_sim
    sim_df['Score__2']     = s2_sim
    team1_wins             = margin_sim > 0
    sim_df['Win__1']       = team1_wins
    sim_df['Winning_Team'] = np.where(
        team1_wins,
        np.repeat(t1_names, n),
        np.repeat(t2_names, n),
    )
    return sim_df


# ---------------------------------------------------------------------------
# Method 3 (new): feature-grounded Bernoulli sampling
# ---------------------------------------------------------------------------

def _bt_win_prob(barthag1: np.ndarray, barthag2: np.ndarray) -> np.ndarray:
    """
    Bradley-Terry win probability for team 1 from Barthag power ratings.

    Barthag_i = P(team i beats an average D1 team), so the team's log-odds
    quality is logit(Barthag_i).  Under Bradley-Terry:
        P(1 beats 2) = odds_1 / (odds_1 + odds_2)
                     = sigmoid(logit(B1) - logit(B2))

    This is the natural probabilistic mapping and requires no calibration
    constant -- it uses the features' own probability scale.
    """
    eps = 1e-6
    b1  = np.clip(barthag1, eps, 1 - eps)
    b2  = np.clip(barthag2, eps, 1 - eps)
    log_odds_diff = np.log(b1 / (1 - b1)) - np.log(b2 / (1 - b2))
    return 1.0 / (1.0 + np.exp(-log_odds_diff))


def generate_logistic(
    df: pd.DataFrame,
    n: int,
    score_std: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sample win/loss outcomes from a Bradley-Terry probability derived from
    BartTorvik Barthag ratings, then synthesize plausible KenPom-consistent
    scores.

    Why this is better than score perturbation:
    - Win probabilities are grounded in team quality features, not score noise.
    - A 1-seed vs 16-seed matchup will almost always yield the expected winner;
      a 5 vs 12 matchup will reflect realistic ~35% upset rates automatically.
    - The feature columns remain completely untouched; only the outcome labels
      and scores change -- the feature-to-outcome mapping stays consistent.

    Score synthesis:
        tempo   = (AdjT__1 + AdjT__2) / 2          (avg possessions per game)
        s_exp_i = KP__AdjO__i * tempo / 100        (expected points from KP)
        (n1, n2) ~ BivariateNormal(0, score_std, rho=0.5)
          -- positive correlation reflects shared high/low-scoring game effects

    If the noisy scores contradict the sampled Bernoulli outcome, a small
    delta is added to align the score sign without re-drawing, preserving the
    Bernoulli win probability distribution.
    """
    BARTHAG1 = 'BT__Barthag__1'
    BARTHAG2 = 'BT__Barthag__2'
    ADJ_O1   = 'KP__AdjO__1'
    ADJ_O2   = 'KP__AdjO__2'
    ADJ_T1   = 'KP__AdjT__1'
    ADJ_T2   = 'KP__AdjT__2'

    missing = [c for c in [BARTHAG1, BARTHAG2, ADJ_O1, ADJ_O2, ADJ_T1, ADJ_T2]
               if c not in df.columns]
    if missing:
        raise ValueError(
            f'logistic method requires columns: {missing}. '
            'Make sure you are using GameCombinedData (not raw GameData).'
        )

    total     = len(df)
    b1        = pd.to_numeric(df[BARTHAG1], errors='coerce').fillna(0.5).values
    b2        = pd.to_numeric(df[BARTHAG2], errors='coerce').fillna(0.5).values
    p_win1    = _bt_win_prob(b1, b2)

    # KenPom-expected scores (offensive efficiency * pace / 100)
    tempo_est = (
        pd.to_numeric(df[ADJ_T1], errors='coerce').fillna(68).values +
        pd.to_numeric(df[ADJ_T2], errors='coerce').fillna(68).values
    ) / 2.0
    adj_o1    = pd.to_numeric(df[ADJ_O1], errors='coerce').fillna(100).values
    adj_o2    = pd.to_numeric(df[ADJ_O2], errors='coerce').fillna(100).values
    s1_exp    = adj_o1 * tempo_est / 100.0
    s2_exp    = adj_o2 * tempo_est / 100.0

    t1_names  = df['Team__1'].values
    t2_names  = df['Team__2'].values

    # Repeat base arrays n times
    idx_rep   = np.repeat(np.arange(total), n)
    sim_df    = df.iloc[idx_rep].copy().reset_index(drop=True)

    N         = total * n
    p_rep     = np.repeat(p_win1, n)
    s1_rep    = np.repeat(s1_exp, n)
    s2_rep    = np.repeat(s2_exp, n)

    # Sample Bernoulli win outcomes
    team1_wins = rng.random(N) < p_rep   # True with probability p_win1

    # Bivariate Normal score noise (rho=0.5: realistic positive score correlation)
    rho       = 0.5
    z1        = rng.standard_normal(N)
    z2        = rng.standard_normal(N)
    w1        = z1
    w2        = rho * z1 + np.sqrt(1 - rho ** 2) * z2   # correlated with w1
    s1_sim    = s1_rep + score_std * w1
    s2_sim    = s2_rep + score_std * w2

    # Ensure scores are consistent with the sampled outcome.
    # Where they conflict, shift the margin by a small delta rather than
    # re-drawing (avoids biasing the score distribution).
    score_says_1_wins = s1_sim > s2_sim
    mismatch          = team1_wins != score_says_1_wins
    delta             = np.abs(s1_sim - s2_sim) + 0.5
    s1_sim[mismatch & team1_wins]  += delta[mismatch & team1_wins]
    s2_sim[mismatch & ~team1_wins] += delta[mismatch & ~team1_wins]

    sim_df['Score__1']     = s1_sim
    sim_df['Score__2']     = s2_sim
    sim_df['Win__1']       = team1_wins
    sim_df['Winning_Team'] = np.where(
        team1_wins,
        np.repeat(t1_names, n),
        np.repeat(t2_names, n),
    )
    return sim_df


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------

def print_diagnostics(df_real: pd.DataFrame, df_sim: pd.DataFrame, method: str) -> None:
    real_rate = float(pd.to_numeric(df_real['Win__1'], errors='coerce').mean())
    sim_rate  = float(df_sim['Win__1'].mean())
    print(f'Real Win__1 rate : {real_rate:.4f}')
    print(f'Sim  Win__1 rate : {sim_rate:.4f}  (delta={sim_rate - real_rate:+.4f})')

    if method == 'logistic':
        seed_col1 = 'Seed__1' if 'Seed__1' in df_sim.columns else None
        seed_col2 = 'Seed__2' if 'Seed__2' in df_sim.columns else None
        if seed_col1 and seed_col2:
            df_sim = df_sim.copy()
            df_sim['_seed_gap'] = (
                pd.to_numeric(df_sim[seed_col2], errors='coerce') -
                pd.to_numeric(df_sim[seed_col1], errors='coerce')
            )
            for label, mask in [
                ('higher-seed fav  (gap > 4)',   df_sim['_seed_gap'] >  4),
                ('close matchup    (|gap| <= 4)', df_sim['_seed_gap'].abs() <= 4),
                ('underdog favored (gap < -4)',   df_sim['_seed_gap'] < -4),
            ]:
                sub = df_sim[mask]
                if len(sub):
                    print(f'  {label}: win1_rate={sub["Win__1"].mean():.3f}  (n={len(sub)})')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_arg_parser()
    args   = parser.parse_args()

    # Validate method-specific args
    if args.method in ('noise', 'margin') and args.std is None:
        parser.error(f'--std is required for --method {args.method}')
    if args.method == 'logistic' and args.std is not None:
        print(f'Note: --std is ignored for --method logistic '
              f'(using --score-std={args.score_std} for synthetic score generation)')
    if args.method in ('noise', 'margin') and args.std is not None and args.std <= 0:
        parser.error('--std must be positive.')
    if args.n < 1:
        parser.error('--n must be at least 1.')

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    data_root  = Path(args.data_root) if args.data_root else script_dir.parent
    source     = Path(args.source) if args.source else (
        data_root / 'Data' / 'GameCombinedData' / 'All.csv'
    )
    out_dir    = data_root / 'Data' / f'SimulatedData{args.identifier}'
    out_path   = out_dir / 'All.csv'

    if not source.exists():
        print(f'ERROR: Source file not found: {source}', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(source)

    print(f'Source         : {source}')
    print(f'Real games     : {len(df)} rows  ({df["Year"].nunique()} years)')
    print(f'Method         : {args.method}')
    print(f'Identifier     : SimulatedData{args.identifier}')
    if args.method in ('noise', 'margin'):
        print(f'Noise sigma    : {args.std}')
    if args.method == 'logistic':
        print(f'Score noise    : {args.score_std}  (synthetic score generation only)')
    print(f'Copies/game    : {args.n}')
    print(f'Total sim rows : {len(df) * args.n}')
    print(f'Random seed    : {args.seed}')

    if args.dry_run:
        print('\n--dry-run: no files written.')
        return

    rng = np.random.default_rng(args.seed)

    if args.method == 'noise':
        sim_df = generate_noise(df, args.std, args.n, rng)
    elif args.method == 'margin':
        sim_df = generate_margin(df, args.std, args.n, rng)
    elif args.method == 'logistic':
        sim_df = generate_logistic(df, args.n, args.score_std, rng)
    else:
        raise ValueError(f'Unknown method: {args.method}')

    print_diagnostics(df, sim_df, args.method)

    out_dir.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(out_path, index=False)
    print(f'\nOutput         : {out_path}')
    print('Done.')


if __name__ == '__main__':
    main()
