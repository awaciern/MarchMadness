#!/usr/bin/env python3
"""
generate_sim_data.py
--------------------
Generate simulated training data from real tournament game rows.
Eight generation methods are supported, grouped by what they perturb:

SCORE-PERTURBATION (original 3 — perturb game outcomes only, features unchanged)
─────────────────────────────────────────────────────────────────────────────────
  noise
        Add independent Gaussian noise to Score__1 and Score__2 and
        recompute the winner.  Simple, but conflates tempo-driven total
        scoring with competitive uncertainty.

  margin
        Perturb only the point spread while keeping the total game score
        fixed.  More physically grounded: pace/tempo determines how many
        total points are scored; only the competitive margin is uncertain.
          margin_sim = (s1 - s2) + N(0, sigma)
          s1_sim = (s_total + margin_sim) / 2  ;  s2_sim = (s_total - margin_sim) / 2

  logistic
        Sample win/loss outcomes from a Bradley-Terry win probability derived
        from BartTorvik Barthag ratings.  Scores are synthesised from KenPom
        AdjO/AdjT with correlated Bivariate Normal noise (rho=0.5).

FEATURE-PERTURBATION (5 new — modify the team statistics themselves)
──────────────────────────────────────────────────────────────────────
  feature_noise                                       (--feat-noise-frac)
        Add independent Gaussian noise to every numeric feature column
        (KenPom, BartTorvik, 2-week BT, hotness delta, win records, …).
        Noise std = noise_frac × std(column).  Win outcome re-derived via
        Bradley-Terry from the noisy Barthag values.

  correlated                                          (--feat-noise-frac)
        Multivariate Gaussian noise using a Ledoit-Wolf regularised empirical
        covariance matrix fitted on the team-1 feature set from the real data.
        Related statistics (AdjO / Barthag / EFG%) move together rather than
        independently, generating more realistic synthetic teams.  The same
        covariance structure is applied independently to team-2 features.

  smote                              (--k-neighbors, --pca-components)
        K-NN SMOTE-style interpolation: for each real row, find its k nearest
        neighbours in PCA-compressed feature space, pick one uniformly at
        random, and linearly interpolate at λ ~ U(0,1) in the original space.
        Creates genuinely new matchup scenarios that lie between real games.

  mixup                                               (--mixup-alpha)
        Random-pair convex combination (Zhang et al. 2018): two rows drawn
        independently and blended as  x = λ·xᵢ + (1−λ)·xⱼ  with
        λ ~ Beta(alpha, alpha).  Strong regularisation effect; the blended
        feature vectors occupy regions of space that no real team has occupied.

  swap
        Mirror every game row by swapping team-1 and team-2 (all __1 / __2
        column pairs are exchanged, Win__1 is flipped).  Doubles the dataset
        with zero information fabrication and eliminates slot-assignment bias.
        Always produces exactly one swap per real row (ignores --n).

Output:  Data/SimulatedData<identifier>/All.csv

Usage examples
──────────────
  # Feature noise (5% of each column's std)
  python3 Python/generate_sim_data.py --method feature_noise --identifier FeatNoise --n 10

  # Correlated MVN noise (8% of each column's std)
  python3 Python/generate_sim_data.py --method correlated --identifier Corr8 --n 10 --feat-noise-frac 0.08

  # SMOTE interpolation, 5 neighbours
  python3 Python/generate_sim_data.py --method smote --identifier SMOTE5 --n 5

  # Mixup (alpha=2)
  python3 Python/generate_sim_data.py --method mixup --identifier Mixup2 --n 10

  # Team-order swap (n is ignored)
  python3 Python/generate_sim_data.py --method swap --identifier Swap --n 1

  # Original methods (unchanged)
  python3 Python/generate_sim_data.py --method margin --identifier Margin8 --std 8 --n 15
  python3 Python/generate_sim_data.py --method logistic --identifier BT20 --n 20
  python3 Python/generate_sim_data.py --method noise --identifier Noise5 --std 5 --n 15
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Feature-space methods live in a companion module to keep this file readable
_SYN_METHODS_AVAILABLE = False
try:
    from syn_feature_methods import (
        generate_feature_noise,
        generate_correlated_noise,
        generate_smote,
        generate_mixup,
        generate_swap,
        print_feature_diagnostics,
    )
    _SYN_METHODS_AVAILABLE = True
except ImportError:
    pass  # will error at dispatch time if user tries to use these methods


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
        choices=['noise', 'margin', 'logistic',
                 'feature_noise', 'correlated', 'smote', 'mixup', 'swap'],
        help=(
            'Simulation method.  Score-perturbation: noise, margin, logistic.  '
            'Feature-perturbation: feature_noise, correlated, smote, mixup, swap.'
        ),
    )
    p.add_argument(
        '--identifier', required=True,
        help='Unique name for this dataset (e.g. "FeatNoise"). '
             'Output is written to Data/SimulatedData<identifier>/All.csv.',
    )
    p.add_argument(
        '--std', type=float, default=None,
        help='Noise std-dev for "noise" and "margin" methods.',
    )
    p.add_argument(
        '--n', type=int, required=True,
        help='Number of simulated rows per real game row '
             '(ignored for "swap", which always generates exactly 1 copy).',
    )
    p.add_argument(
        '--score-std', type=float, default=6.0,
        help='(logistic only) Std-dev of score noise around KenPom expectation. '
             'Default: 6.0.',
    )
    # ---- feature-perturbation knobs ----
    p.add_argument(
        '--feat-noise-frac', type=float, default=0.05,
        dest='feat_noise_frac',
        help='(feature_noise / correlated) Noise level as a fraction of each '
             'column\'s std dev.  Default: 0.05 (5%%).',
    )
    p.add_argument(
        '--k-neighbors', type=int, default=5,
        dest='k_neighbors',
        help='(smote) Number of nearest neighbours to consider per row. '
             'Default: 5.',
    )
    p.add_argument(
        '--pca-components', type=int, default=20,
        dest='pca_components',
        help='(smote) Number of PCA components used for the kNN index. '
             'Default: 20.',
    )
    p.add_argument(
        '--mixup-alpha', type=float, default=2.0,
        dest='mixup_alpha',
        help='(mixup) Beta distribution concentration parameter. '
             'Higher = more blending.  Default: 2.0.',
    )
    # ---- common options ----
    p.add_argument(
        '--source', default=None,
        help='Explicit path to source All.csv. '
             'Defaults to <data-root>/Data/GameCombinedData/All.csv.',
    )
    p.add_argument(
        '--data-root', default=None,
        help='Path to repo root.  Inferred from this script\'s location if omitted.',
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

    # ---- method-specific validation ----
    score_methods   = {'noise', 'margin', 'logistic'}
    feature_methods = {'feature_noise', 'correlated', 'smote', 'mixup', 'swap'}

    if args.method in ('noise', 'margin') and args.std is None:
        parser.error(f'--std is required for --method {args.method}')
    if args.method == 'logistic' and args.std is not None:
        print(f'Note: --std is ignored for --method logistic '
              f'(using --score-std={args.score_std} for score generation)')
    if args.method in ('noise', 'margin') and args.std is not None and args.std <= 0:
        parser.error('--std must be positive.')
    if args.method in feature_methods and not _SYN_METHODS_AVAILABLE:
        parser.error(
            f'Method "{args.method}" requires syn_feature_methods.py to be present '
            'in the same directory as generate_sim_data.py.'
        )
    if args.n < 1:
        parser.error('--n must be at least 1.')
    if args.method == 'swap' and args.n != 1:
        print(f'Note: --n is ignored for "swap" (always generates 1 copy per row).')

    # ---- resolve paths ----
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

    # ---- print run summary ----
    print(f'Source         : {source}')
    print(f'Real games     : {len(df)} rows  ({df["Year"].nunique()} years)')
    print(f'Method         : {args.method}')
    print(f'Identifier     : SimulatedData{args.identifier}')

    _is_swap = args.method == 'swap'
    expected_rows = len(df) if _is_swap else len(df) * args.n

    if args.method in ('noise', 'margin'):
        print(f'Noise sigma    : {args.std}')
    if args.method == 'logistic':
        print(f'Score noise    : {args.score_std}  (synthetic score generation only)')
    if args.method in ('feature_noise', 'correlated'):
        print(f'Feat noise frac: {args.feat_noise_frac}  (×std per column)')
    if args.method == 'smote':
        print(f'K neighbours   : {args.k_neighbors}')
        print(f'PCA components : {args.pca_components}')
    if args.method == 'mixup':
        print(f'Mixup alpha    : {args.mixup_alpha}')
    if not _is_swap:
        print(f'Copies/game    : {args.n}')
    print(f'Total sim rows : {expected_rows}')
    print(f'Random seed    : {args.seed}')

    if args.dry_run:
        print('\n--dry-run: no files written.')
        return

    rng = np.random.default_rng(args.seed)

    # ---- dispatch ----
    if args.method == 'noise':
        sim_df = generate_noise(df, args.std, args.n, rng)
        print_diagnostics(df, sim_df, args.method)

    elif args.method == 'margin':
        sim_df = generate_margin(df, args.std, args.n, rng)
        print_diagnostics(df, sim_df, args.method)

    elif args.method == 'logistic':
        sim_df = generate_logistic(df, args.n, args.score_std, rng)
        print_diagnostics(df, sim_df, args.method)

    elif args.method == 'feature_noise':
        sim_df = generate_feature_noise(df, args.n, rng,
                                        noise_frac=args.feat_noise_frac)
        print_feature_diagnostics(df, sim_df, args.method)

    elif args.method == 'correlated':
        sim_df = generate_correlated_noise(df, args.n, rng,
                                           noise_frac=args.feat_noise_frac)
        print_feature_diagnostics(df, sim_df, args.method)

    elif args.method == 'smote':
        sim_df = generate_smote(df, args.n, rng,
                                k_neighbors=args.k_neighbors,
                                pca_components=args.pca_components)
        print_feature_diagnostics(df, sim_df, args.method)

    elif args.method == 'mixup':
        sim_df = generate_mixup(df, args.n, rng,
                                alpha=args.mixup_alpha)
        print_feature_diagnostics(df, sim_df, args.method)

    elif args.method == 'swap':
        sim_df = generate_swap(df)
        print_feature_diagnostics(df, sim_df, args.method)

    else:
        raise ValueError(f'Unknown method: {args.method}')

    out_dir.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(out_path, index=False)
    print(f'\nOutput         : {out_path}')
    print('Done.')


if __name__ == '__main__':
    main()
