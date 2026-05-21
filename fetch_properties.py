"""
Fetch physical properties for matched galaxies across multiple snapshots
using local TNG group catalogs.

Input:  progenitors CSV from match_progenitors.py (columns idx_z0, subfind_z0,
        and one subfind_snap_<S> column per target snapshot).

Output: wide CSV. For each snapshot S the descendant/progenitor lives at, this
        script adds columns:
            sfr_snap_<S>
            mass_stars_snap_<S>      (SubhaloMassType[:, 4])
            mass_total_snap_<S>      (SubhaloMass — total bound subhalo mass)
            m200_snap_<S>            (Group_M_Crit200 of parent FoF group)
            starmet_snap_<S>         (SubhaloStarMetallicity)
            gasmet_snap_<S>          (SubhaloGasMetallicity)

Units: all values are in TNG code units. Masses are in 1e10 Msun/h. SFR is
       in Msun/yr (no h dependence). Metallicities are mass fractions.

The descendant (snap 99) is always included automatically using subfind_z0.

Example:
    python fetch_properties.py \\
        --input progenitors_multi.csv \\
        --output progenitor_properties.csv
"""

import argparse
import csv
import sys
import time

import numpy as np

try:
    import illustris_python as il
except ImportError:
    sys.exit("illustris_python is required.")

SNAP_Z0 = 99

SUBHALO_FIELDS = [
    'SubhaloSFR',
    'SubhaloMassType',
    'SubhaloMass',
    'SubhaloStarMetallicity',
    'SubhaloGasMetallicity',
    'SubhaloGrNr',
]
GROUP_FIELDS = ['Group_M_Crit200']

PROP_SUFFIXES = ['sfr', 'mass_stars', 'mass_total', 'm200', 'starmet', 'gasmet']


def load_snapshot_props(basePath, snap):
    """Return dict mapping SubfindID -> (sfr, mstar, mtot, m200, starmet, gasmet)
    by loading the full group catalog for one snapshot."""
    print(f"  Loading subhalo catalog for snap {snap}...")
    subs = il.groupcat.loadSubhalos(basePath, snap, fields=SUBHALO_FIELDS)
    print(f"  Loading FoF group catalog for snap {snap}...")
    grps = il.groupcat.loadHalos(basePath, snap, fields=GROUP_FIELDS)
    # When loadHalos is given a single field, it returns the array directly
    # rather than a dict. Handle both cases.
    if isinstance(grps, dict):
        m200_arr = grps['Group_M_Crit200']
        n_grp = grps['count']
    else:
        m200_arr = grps
        n_grp = len(grps)

    return {
        'sfr': subs['SubhaloSFR'],
        'mass_stars': subs['SubhaloMassType'][:, 4],
        'mass_total': subs['SubhaloMass'],
        'starmet': subs['SubhaloStarMetallicity'],
        'gasmet': subs['SubhaloGasMetallicity'],
        'grnr': subs['SubhaloGrNr'],
        'm200_by_group': m200_arr,
        'n_sub': subs['count'],
        'n_grp': n_grp,
    }


def lookup(props, sid):
    """Return ordered prop values for one SubfindID, or empty strings if invalid."""
    n_sub = props['n_sub']
    n_grp = props['n_grp']
    if sid < 0 or sid >= n_sub:
        return [''] * len(PROP_SUFFIXES)
    grnr = int(props['grnr'][sid])
    if 0 <= grnr < n_grp:
        m200 = float(props['m200_by_group'][grnr])
    else:
        m200 = ''
    return [
        float(props['sfr'][sid]),
        float(props['mass_stars'][sid]),
        float(props['mass_total'][sid]),
        m200,
        float(props['starmet'][sid]),
        float(props['gasmet'][sid]),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV from match_progenitors.py')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV with properties added')
    parser.add_argument('--basePath', type=str, default='../data',
                        help='TNG basePath')
    parser.add_argument('--snapshots', type=int, nargs='+', default=None,
                        help='Restrict to these snapshots (e.g. 17 33 51 99). '
                             'Defaults to all snapshots found in the input CSV '
                             'plus snap 99 (z=0 descendants).')
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    with open(args.input, 'r', newline='') as f:
        reader = csv.DictReader(f)
        in_fieldnames = reader.fieldnames
        rows = list(reader)
    print(f"  {len(rows)} rows")

    # Identify snapshot columns available in the input
    snap_cols = [c for c in in_fieldnames if c.startswith('subfind_snap_')]
    available_snaps = {int(c.replace('subfind_snap_', '')) for c in snap_cols}

    if args.snapshots is not None:
        snaps = sorted(set(args.snapshots))
        missing = [s for s in snaps if s != SNAP_Z0 and s not in available_snaps]
        if missing:
            print(f"  WARNING: requested snapshots not in input CSV: {missing} "
                  f"(those columns will be empty)")
    else:
        snaps = sorted(available_snaps | {SNAP_Z0})
    print(f"  Snapshots to fetch properties for: {snaps}")

    # Load all required snapshot catalogs once
    t0 = time.time()
    all_props = {}
    for s in snaps:
        all_props[s] = load_snapshot_props(args.basePath, s)
    print(f"All catalogs loaded in {time.time()-t0:.1f} s")

    # Build output columns
    out_fieldnames = list(in_fieldnames)
    for s in snaps:
        for suf in PROP_SUFFIXES:
            out_fieldnames.append(f'{suf}_snap_{s}')

    print(f"Writing {args.output}...")
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for s in snaps:
                if s == SNAP_Z0:
                    sid_str = row['subfind_z0']
                else:
                    sid_str = row.get(f'subfind_snap_{s}', '')
                if sid_str in ('', None):
                    vals = [''] * len(PROP_SUFFIXES)
                else:
                    vals = lookup(all_props[s], int(sid_str))
                for suf, v in zip(PROP_SUFFIXES, vals):
                    out[f'{suf}_snap_{s}'] = v
            writer.writerow(out)
    print(f"Done. Wrote {len(rows)} rows × {len(out_fieldnames)} columns.")


if __name__ == '__main__':
    main()