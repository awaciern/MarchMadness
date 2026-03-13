#!/usr/bin/env python3
"""
generate_sim_data.py
--------------------
Generate simulated training data by adding independent Gaussian noise to each
game's final scores and recomputing which team won.

The feature columns (team stats, seeds, conference, etc.) are kept exactly as
in the real game — only Score__1, Score__2, Win__1, and Winning_Team are
modified.  This lets models train on the augmented dataset while ensuring the
learned relationship between features and outcomes is preserved (but with
softened certainty near close matchups).

Output is written to:
    Data/SimulatedData<identifier>/All.csv

Usage examples:
    # Standard-deviation 5, 15 copies per game, identifier "Noise"
    python3 Python/generate_sim_data.py --identifier Noise --std 5 --n 15

    # Preview count without writing
    python3 Python/generate_sim_data.py --identifier Noise --std 5 --n 15 --dry-run
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Generate Gaussian-noise score simulations from real game data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--identifier', required=True,
        help='Unique name for this dataset (e.g. "Noise"). '
             'Output is written to Data/SimulatedData<identifier>/All.csv.',
    )
    p.add_argument(
        '--std', type=float, required=True,
        help='Standard deviation (σ) of the Gaussian score perturbation. Mean is always 0.',
    )
    p.add_argument(
        '--n', type=int, required=True,
        help='Number of simulated game entries to produce per real game row.',
    )
    p.add_argument(
        '--source', default=None,
        help='Explicit path to source All.csv. '
             'Defaults to <data-root>/Data/GameCombinedData/All.csv.',
    )
    p.add_argument(
        '--data-root', default=None,
        help='Path to repo root (the directory containing Data/). '
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
# Core generation logic
# ---------------------------------------------------------------------------

def generate(
    df: pd.DataFrame,
    std: float,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    For every row in *df*, produce *n* simulated rows where:
      - Score__1_sim  = Score__1 + N(0, σ)
      - Score__2_sim  = Score__2 + N(0, σ)  (independent draw)
      - Win__1        = Score__1_sim > Score__2_sim
      - Winning_Team  = Team__1 if Win__1 else Team__2

    All other columns are copied unchanged from the original row.
    """
    if 'Score__1' not in df.columns or 'Score__2' not in df.columns:
        raise ValueError("Source CSV must have 'Score__1' and 'Score__2' columns.")

    total = len(df)
    # Pre-allocate noise arrays for speed: shape (total, n)
    noise1 = rng.normal(0.0, std, size=(total, n))
    noise2 = rng.normal(0.0, std, size=(total, n))

    # Repeat each row n times: (total*n) rows
    idx_repeated = np.repeat(np.arange(total), n)
    sim_df = df.iloc[idx_repeated].copy().reset_index(drop=True)

    # Flat noise vectors
    flat_noise1 = noise1.reshape(-1)
    flat_noise2 = noise2.reshape(-1)

    s1_base = pd.to_numeric(df['Score__1'], errors='coerce').fillna(0).values
    s2_base = pd.to_numeric(df['Score__2'], errors='coerce').fillna(0).values
    s1_sim = np.repeat(s1_base, n) + flat_noise1
    s2_sim = np.repeat(s2_base, n) + flat_noise2

    team1_names = df['Team__1'].values
    team2_names = df['Team__2'].values

    sim_df['Score__1']     = s1_sim
    sim_df['Score__2']     = s2_sim
    team1_wins             = s1_sim > s2_sim
    sim_df['Win__1']       = team1_wins
    sim_df['Winning_Team'] = np.where(
        team1_wins,
        np.repeat(team1_names, n),
        np.repeat(team2_names, n),
    )

    return sim_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_arg_parser()
    args   = parser.parse_args()

    if args.std <= 0:
        parser.error('--std must be positive.')
    if args.n < 1:
        parser.error('--n must be at least 1.')

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    data_root  = Path(args.data_root) if args.data_root else script_dir.parent
    source     = Path(args.source) if args.source else (
        data_root / 'Data' / 'GameCombinedData' / 'All.csv'
    )
    out_dir  = data_root / 'Data' / f'SimulatedData{args.identifier}'
    out_path = out_dir / 'All.csv'

    if not source.exists():
        print(f'ERROR: Source file not found: {source}', file=sys.stderr)
        sys.exit(1)

    # Load real data
    df = pd.read_csv(source)
    print(f'Source:         {source}')
    print(f'Real games:     {len(df)} rows  ({df["Year"].nunique()} years)')
    print(f'Identifier:     SimulatedData{args.identifier}')
    print(f'Noise σ:        {args.std}')
    print(f'Copies/game:    {args.n}')
    print(f'Total sim rows: {len(df) * args.n}')
    print(f'Random seed:    {args.seed}')

    # Compute roughly how many upsets-flipped
    real_win_rate = float(pd.to_numeric(df['Win__1'], errors='coerce').mean())
    print(f'Real Win__1 rate: {real_win_rate:.4f}')

    if args.dry_run:
        print('\n--dry-run: no files written.')
        return

    # Generate
    rng    = np.random.default_rng(args.seed)
    sim_df = generate(df, args.std, args.n, rng)

    sim_win_rate = float(sim_df['Win__1'].mean())
    print(f'Sim  Win__1 rate: {sim_win_rate:.4f}  '
          f'(delta={sim_win_rate - real_win_rate:+.4f})')

    # Write
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(out_path, index=False)
    print(f'\nOutput: {out_path}')
    print('Done.')


if __name__ == '__main__':
    main()
