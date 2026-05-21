"""
Match z=0 galaxies (from a saved embedding) to their main progenitors at one or
more target snapshots using the local SubLink merger trees.

Output: CSV with one row per z=0 galaxy and one column per target snapshot.
Columns: [idx_z0, subfind_z0, subfind_snap_<S1>, subfind_snap_<S2>, ...]
Empty cells indicate that the galaxy has no main progenitor at that snapshot.

Example:
    # Single target snapshot
    python match_progenitors.py --target_snapshots 51 --output progenitors_snap51.csv

    # Multiple snapshots in one pass (each tree load gives full MPB for free)
    python match_progenitors.py --target_snapshots 84 67 51 33 17 \\
        --output progenitors_multi.csv
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

try:
    import illustris_python as il
except ImportError:
    sys.exit("illustris_python is required. Install it from https://github.com/illustristng/illustris_python")

SNAP_Z0 = 99
BOX_CKPC = 75000.0  # TNG100 box: 75 Mpc/h


def load_mpb(basePath, sid, fields):
    """Load main progenitor branch for one z=0 subhalo. Returns dict or None."""
    return il.sublink.loadTree(basePath, SNAP_Z0, sid, fields=fields, onlyMPB=True)


def periodic_distance(p1, p2, box):
    delta = p1 - p2
    delta -= box * np.round(delta / box)
    return float(np.linalg.norm(delta))


def verify_positions(basePath, output_path, target_snapshots, n_sample, seed=42):
    """Sanity-check: for n_sample random matched pairs, fetch positions from
    the merger tree (no API) and print the periodic comoving distance between
    descendant and progenitor."""
    rows = []
    with open(output_path, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No rows in CSV to verify.")
        return

    rng = np.random.default_rng(seed)
    sample = rng.choice(len(rows), min(n_sample, len(rows)), replace=False)

    for target in target_snapshots:
        col = f'subfind_snap_{target}'
        matched_rows = [rows[i] for i in sample
                        if rows[i].get(col, '') not in ('', None)]
        if not matched_rows:
            print(f"\nSnap {target}: no matches in sampled rows.")
            continue
        print(f"\nVerifying {len(matched_rows)} pairs for snap {target} "
              f"(positions in ckpc/h, box={BOX_CKPC:.0f}):")
        print(f"{'sid_z0':>10} {'sid_prog':>10} {'dist (ckpc/h)':>14}")
        distances = []
        for row in matched_rows:
            sid_z0 = int(row['subfind_z0'])
            tree = load_mpb(basePath, sid_z0,
                            fields=['SnapNum', 'SubfindID', 'SubhaloPos'])
            if tree is None:
                continue
            snaps = list(tree['SnapNum'])
            i0 = snaps.index(SNAP_Z0)
            it = snaps.index(target)
            d = periodic_distance(tree['SubhaloPos'][i0],
                                  tree['SubhaloPos'][it], BOX_CKPC)
            distances.append(d)
            print(f"{sid_z0:>10} {int(row[col]):>10} {d:>14.0f}")
        if distances:
            d = np.array(distances)
            print(f"  summary: median={np.median(d):.0f}, min={d.min():.0f}, "
                  f"max={d.max():.0f} ckpc/h")


def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path, 'r', newline='') as f:
        return {int(r['subfind_z0']) for r in csv.DictReader(f)
                if r.get('subfind_z0', '').isdigit()}


def main():
    parser = argparse.ArgumentParser(
        description='Match z=0 galaxies to progenitors at one or more target snapshots '
                    'using local SubLink trees.')
    parser.add_argument('--target_snapshots', type=int, nargs='+', required=True,
                        help='One or more target snapshot numbers (e.g. 84 67 51 33 17)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV path (resumable)')
    parser.add_argument('--basePath', type=str, default='../data',
                        help='TNG basePath (contains postprocessing/trees/SubLink/)')
    parser.add_argument('--z0_embedding', type=str, default='snapshot99_embedding.npz',
                        help='Path to z=0 embedding npz with original_index')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Flush to disk every N processed galaxies')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only the first N galaxies (debug)')
    parser.add_argument('--verify', type=int, default=5,
                        help='Sanity-check N random pairs after matching (0 to skip)')
    parser.add_argument('--verify_only', action='store_true',
                        help='Skip matching, only verify an existing CSV')
    args = parser.parse_args()

    targets = sorted(set(args.target_snapshots))
    for t in targets:
        if t >= SNAP_Z0:
            sys.exit(f"target_snapshot ({t}) must be < {SNAP_Z0} (z=0)")

    if args.verify_only:
        if args.verify <= 0:
            sys.exit("--verify_only requires --verify N (N > 0)")
        verify_positions(args.basePath, args.output, targets, args.verify)
        return

    print(f"Loading z=0 embedding from {args.z0_embedding}...")
    data = np.load(args.z0_embedding)
    orig_z0 = np.asarray(data['original_index']).astype(int)
    if args.limit is not None:
        orig_z0 = orig_z0[:args.limit]
    print(f"  {len(orig_z0)} z=0 galaxies, target snapshots: {targets}")

    done = load_done(args.output)
    if done:
        print(f"Resuming: {len(done)} already in {args.output}")

    new_file = not os.path.exists(args.output)
    f = open(args.output, 'a', newline='')
    writer = csv.writer(f)
    header = ['idx_z0', 'subfind_z0'] + [f'subfind_snap_{t}' for t in targets]
    if new_file:
        writer.writerow(header)
        f.flush()

    t0 = time.time()
    processed = 0
    matched_per_snap = {t: 0 for t in targets}
    try:
        for idx, sid in enumerate(orig_z0):
            sid = int(sid)
            if sid in done:
                continue
            tree = load_mpb(args.basePath, sid,
                            fields=['SnapNum', 'SubfindID'])
            row = [idx, sid]
            if tree is None:
                row.extend([''] * len(targets))
            else:
                snaps = list(tree['SnapNum'])
                sids = tree['SubfindID']
                for t in targets:
                    if t in snaps:
                        prog = int(sids[snaps.index(t)])
                        row.append(prog)
                        matched_per_snap[t] += 1
                    else:
                        row.append('')
            writer.writerow(row)
            processed += 1
            if processed % args.save_every == 0:
                f.flush()
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = len(orig_z0) - len(done) - processed
                eta = remaining / rate if rate > 0 else float('inf')
                print(f"  [{processed}] rate={rate:.1f}/s eta={eta:.1f}s "
                      f"matched per snap: {matched_per_snap}")
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")
    finally:
        f.flush()
        f.close()

    elapsed = time.time() - t0
    print(f"\nDone. Processed {processed} galaxies in {elapsed:.1f} s.")
    for t in targets:
        print(f"  snap {t}: {matched_per_snap[t]} matched")
    print(f"Output: {args.output}")

    if args.verify > 0:
        verify_positions(args.basePath, args.output, targets, args.verify)


if __name__ == '__main__':
    main()