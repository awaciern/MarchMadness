#!/usr/bin/env python3
"""Downsample simulated All.csv files per-year to match real data counts.

Creates a new folder next to each simulated dataset named <orig>_downsampled
and writes All.csv with rows sampled per-year to match counts from
Data/GameCombinedData/All.csv.

Usage: python Python/downsample_sim_per_year.py [--seed SEED] [sim_folder ...]
If no sim_folder arguments are given, the script processes all folders in
Data/ starting with "SimulatedData" and not already ending with "_downsampled".
"""

import argparse
import csv
import glob
import os
import random
from collections import defaultdict


def read_real_year_counts(real_csv_path):
    counts = defaultdict(int)
    with open(real_csv_path, newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        try:
            year_idx = header.index('Year')
        except ValueError:
            # try common alternatives
            year_idx = None
            for i, col in enumerate(header):
                if col.strip().lower() == 'year':
                    year_idx = i
                    break
            if year_idx is None:
                raise RuntimeError('Could not find Year column in %s' % real_csv_path)
        for row in reader:
            if len(row) <= year_idx:
                continue
            try:
                y = int(row[year_idx])
            except Exception:
                continue
            counts[y] += 1
    return counts


def process_sim_folder(sim_folder, real_counts, seed=None):
    sim_csv = os.path.join(sim_folder, 'All.csv')
    if not os.path.exists(sim_csv):
        print(f"Skipping {sim_folder}: All.csv not found")
        return

    with open(sim_csv, newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        try:
            year_idx = header.index('Year')
        except ValueError:
            year_idx = None
            for i, col in enumerate(header):
                if col.strip().lower() == 'year':
                    year_idx = i
                    break
            if year_idx is None:
                print(f"Skipping {sim_folder}: Year column not found")
                return

        rows_by_year = defaultdict(list)
        for row in reader:
            if len(row) <= year_idx:
                continue
            try:
                y = int(row[year_idx])
            except Exception:
                continue
            rows_by_year[y].append(row)

    rng = random.Random(seed)
    out_folder = sim_folder + '_downsampled'
    os.makedirs(out_folder, exist_ok=True)
    out_csv = os.path.join(out_folder, 'All.csv')

    total_in = sum(len(v) for v in rows_by_year.values())
    total_out = 0
    selected_rows = []

    for year, rows in sorted(rows_by_year.items()):
        real_n = real_counts.get(year, 0)
        if real_n <= 0:
            # if no real rows for this year, skip (or keep zero)
            take = 0
        else:
            take = min(len(rows), real_n)
        if take == 0:
            continue
        selected = rng.sample(rows, take) if take < len(rows) else list(rows)
        selected_rows.extend(selected)
        total_out += len(selected)

    # write output preserving header
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(selected_rows)

    print(f"Processed {sim_folder}: in={total_in} out={total_out} wrote {out_csv}")


def find_sim_folders(args_folders):
    if args_folders:
        return args_folders
    base = 'Data'
    pattern = os.path.join(base, 'SimulatedData*')
    all_folders = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    # exclude already downsampled folders
    return [p for p in all_folders if not p.endswith('_downsampled')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sim_folders', nargs='*', help='Simulated data folders to process')
    p.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    p.add_argument('--real-csv', default=os.path.join('Data', 'GameCombinedData', 'All.csv'), help='Path to real All.csv')
    args = p.parse_args()

    if not os.path.exists(args.real_csv):
        raise SystemExit(f"Real All.csv not found at {args.real_csv}")

    real_counts = read_real_year_counts(args.real_csv)
    sim_folders = find_sim_folders(args.sim_folders)
    if not sim_folders:
        print('No simulated data folders found to process')
        return

    for folder in sim_folders:
        process_sim_folder(folder, real_counts, seed=args.seed)


if __name__ == '__main__':
    main()
