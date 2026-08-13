"""
Generate MDS embedding for a TNG100 snapshot -- fast version. Written with Claude Code.

Same algorithm and same results as generate_embedding.py, but avoids ever building
an N x N physical distance matrix and avoids the O(N^3) double-centering.

Three changes, all exact (no approximation):

1. KD-tree neighbour search (scipy.spatial.cKDTree with periodic `boxsize`) instead
   of the O(N^2) `periodic_distance_matrix` loop. Neighbour *distances* are then
   recomputed with the identical float32 minimum-image arithmetic the original used,
   so the numbers are bit-for-bit the same rather than merely close.

2. K-column truncation of the sorted-neighbour vectors. In the original, each sorted
   row is (N - n_i) leading zeros followed by n_i neighbour distances. The leading
   zeros are identical in every row, so they contribute nothing to any pairwise
   Euclidean distance and can be dropped. We keep only K = max_i(n_i) columns.
   This is lossless, not an approximation.

3. Rank-1 double-centering. The original forms C = I - J/N explicitly and computes
   C @ D @ C, which is O(N^3) and allocates two extra N x N arrays. The algebraically
   identical rank-1 form B = -0.5 * (D - row_means - col_means + grand_mean) is O(N^2)
   and allocates nothing extra.

What still scales as N^2: the similarity matrix `D` between sorted-neighbour vectors,
and the eigendecomposition. That caps this version at roughly 100k galaxies on a
128 GB machine. Going beyond that needs a matrix-free eigensolver.

Example usage:
    python generate_embedding_faster.py --snapshot 99 --output snapshot99_embedding.npz
    python generate_embedding_faster.py --snapshot 99 --threshold 2.0 --top_n 30000 \
        --output snapshot99_embedding_r2_all.npz

Loading saved embeddings:
    data = np.load('snapshot99_embedding.npz')
    embedding = data['embedding']                # MDS embedding (N x 10)
    original_index = data['original_index']      # SubfindIDs for merger tree lookups
    sorted_neighbors = data['sorted_neighbors']  # N x K, right-aligned (see note below)
    positions = data['positions']                # Filtered subhalo positions

NOTE on `sorted_neighbors`: this file stores the N x K truncated form, not the N x N
form the original script wrote. The columns are still right-aligned (leading zeros,
then ascending neighbour distances), so downstream code that truncates to the last
columns -- as embedding_evolution.ipynb and the Nystrom projection already do -- works
unchanged, provided it truncates to at most K columns. `n_neighbors_max` is also saved.
"""

import argparse
import time
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.linalg import eigsh
from astropy import units as u
import illustris_python as il


def impose_cut(subhalos, boolean_array):
    """Filter all subhalo arrays by boolean mask."""
    for key in subhalos.keys():
        if key != 'count':
            subhalos[key] = subhalos[key][boolean_array]
        else:
            subhalos[key] = sum(boolean_array)
    return subhalos


def load_and_filter_subhalos(basePath, snapshot, top_n=10000, min_star_particles=500):
    """Load subhalos, apply quality cuts, and keep top_n most massive by stellar mass."""
    print(f"Loading subhalos from snapshot {snapshot}...")

    fields = ['SubhaloFlag', 'SubhaloGrNr', 'SubhaloMass', 'SubhaloMassType',
              'SubhaloParent', 'SubhaloPos', 'SubhaloLenType',
              'SubhaloStarMetallicity', 'SubhaloSFR', 'SubhaloGasMetallicity']
    subhalos = il.groupcat.loadSubhalos(basePath, snapshot, fields=fields)

    print(f"Loaded {subhalos['count']} subhalos")

    # Track original indices before any cuts
    subhalos['original_index'] = np.arange(subhalos['count'])

    # Convert positions from kpc to Mpc
    for i in range(3):
        subhalos['SubhaloPos'][:, i] = ((subhalos['SubhaloPos'][:, i]) * u.kpc).to(u.Mpc).value

    # Quality cuts
    print("Applying quality cuts...")
    subhalos = impose_cut(subhalos, subhalos['SubhaloFlag'] > 0)
    print(f"  After SubhaloFlag cut: {subhalos['count']}")

    subhalos = impose_cut(subhalos, subhalos['SubhaloLenType'][:, 4] > min_star_particles)
    print(f"  After star particle cut (>{min_star_particles}): {subhalos['count']}")

    # Keep top_n most massive by stellar mass
    stellar_mass = subhalos['SubhaloMassType'][:, 4]
    n_available = subhalos['count']
    if n_available > top_n:
        top_idx = np.argsort(stellar_mass)[-top_n:]
        mask = np.zeros(n_available, dtype=bool)
        mask[top_idx] = True
        subhalos = impose_cut(subhalos, mask)
        print(f"  After top-{top_n} stellar mass selection: {subhalos['count']}")
    else:
        print(f"  Only {n_available} galaxies survive quality cuts (< top_n={top_n}); keeping all")

    min_kept_mass = subhalos['SubhaloMassType'][:, 4].min() * 1e10
    print(f"  Minimum stellar mass kept: {min_kept_mass:.2e} M_sun")

    return subhalos


def _minimum_image_distances(points, i, candidates, boxsize):
    """Distance from point i to `candidates`, in the original's float32 arithmetic.

    Mirrors periodic_distance_matrix in generate_embedding.py exactly: `points` is the
    float32 SubhaloPos array, so every operation here stays in float32 and reproduces
    the original values bit-for-bit. Letting the KD-tree return its own float64
    distances instead would differ at the ~1e-7 level.
    """
    delta = points[candidates] - points[i]
    delta = delta - boxsize * np.round(delta / boxsize)
    return np.sqrt(np.sum(delta ** 2, axis=1))


def sorted_neighbor_vectors(positions, boxsize, threshold):
    """Build the N x K right-aligned sorted-neighbour matrix via a KD-tree.

    Equivalent to np.sort(np.where(D < threshold, D, 0), axis=1) on the full N x N
    periodic distance matrix, with the all-zero leading columns dropped.

    Each original row is a multiset: one 0 for the galaxy itself, n_i neighbour
    distances, and (N - n_i - 1) zeros for everything outside the radius. Sorting makes
    the row (zeros..., ascending distances), so keeping the last K = max_i(n_i) columns
    preserves every nonzero value in every row.
    """
    num_points = positions.shape[0]

    # cKDTree's periodic mode requires coordinates in [0, boxsize). Wrapping does not
    # change minimum-image distances, and distances are recomputed from the unwrapped
    # `positions` below regardless.
    wrapped = np.asarray(positions, dtype=np.float64) % float(boxsize)

    print(f"  Building KD-tree over {num_points} points...")
    tree = cKDTree(wrapped, boxsize=float(boxsize))

    # Query with a slack radius, then apply the original's exact `< threshold` test in
    # float32. The slack covers the float32-vs-float64 discrepancy so that no galaxy
    # sitting right at the boundary is missed by the tree before the exact test runs.
    slack = float(threshold) * 1e-5 + 1e-5
    print(f"  Querying neighbours within {threshold} (+{slack:.2e} slack)...")
    candidate_lists = tree.query_ball_point(wrapped, r=float(threshold) + slack, workers=-1)

    # First pass: exact neighbour distances, and the max neighbour count (sets K).
    print("  Computing neighbour distances (float32, minimum image)...")
    per_galaxy = []
    for i in range(num_points):
        candidates = np.asarray(candidate_lists[i], dtype=np.int64)
        candidates = candidates[candidates != i]  # self contributes a 0, same as a non-neighbour
        if candidates.size == 0:
            per_galaxy.append(np.empty(0, dtype=positions.dtype))
            continue
        d = _minimum_image_distances(positions, i, candidates, boxsize)
        per_galaxy.append(np.sort(d[d < threshold]))

    n_neighbors = np.array([v.size for v in per_galaxy])
    K = int(n_neighbors.max()) if num_points else 0
    print(f"  Max neighbours = {K}; keeping {K} of {num_points} columns "
          f"({num_points - K} all-zero columns dropped)")
    if (n_neighbors == 0).any():
        n_zero = int((n_neighbors == 0).sum())
        print(f"  Note: {n_zero} galaxies ({100 * n_zero / num_points:.1f}%) have zero "
              f"neighbours; their vectors are all-zero and degenerate in the embedding")

    # Second pass: right-align into the N x K matrix. float64 to match the original,
    # whose dist_matrix was a float64 array holding float32-precision values.
    sorted_neighbors = np.zeros((num_points, K), dtype=np.float64)
    for i, v in enumerate(per_galaxy):
        if v.size:
            sorted_neighbors[i, K - v.size:] = v

    return sorted_neighbors


def compute_embedding(subhalos, threshold=5, n_components=10, boxsize=75.0):
    """Compute MDS embedding from subhalo positions.

    `boxsize` is the periodic box side length in Mpc/h -- 75 for TNG100 at every
    snapshot, since positions are comoving. generate_embedding.py instead inferred it
    as round(max(positions[:, 0])), which happens to give 75 for all of snapshots
    17/33/50/51/67/99 but would silently give the wrong box on a sparse enough sample
    (no galaxy within 0.5 Mpc/h of the edge), corrupting every periodic distance.
    """
    positions = subhalos['SubhaloPos']

    print(f"Building sorted-neighbour vectors (boxsize={boxsize} Mpc/h, "
          f"threshold={threshold} Mpc/h)...")
    sorted_neighbors = sorted_neighbor_vectors(positions, boxsize, threshold)

    print("Computing similarity distances...")
    D = euclidean_distances(sorted_neighbors)

    print(f"Computing MDS embedding ({n_components} components)...")
    # Rank-1 double-centering: identical to -0.5 * C @ D @ C for C = I - J/N, but
    # O(N^2) instead of O(N^3) and with no extra N x N allocations.
    row_means = D.mean(axis=1, keepdims=True)
    col_means = D.mean(axis=0, keepdims=True)
    grand_mean = D.mean()
    B = D
    B -= row_means
    B -= col_means
    B += grand_mean
    B *= -0.5

    evals, evecs = eigsh(B, k=n_components, which='LA')

    # Sort from largest to smallest
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # Scale by sqrt of eigenvalues
    embedding = evecs * np.sqrt(evals)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Explained variance (first 3): {np.sum(evals[:3]) / np.sum(evals) * 100:.1f}%")

    return embedding, sorted_neighbors


def main():
    parser = argparse.ArgumentParser(description='Generate MDS embedding for TNG100 snapshot (fast)')
    parser.add_argument('--snapshot', type=int, required=True, help='Snapshot number (e.g., 99 for z=0, 51 for z=0.95)')
    parser.add_argument('--output', type=str, required=True, help='Output filename (e.g., snapshot99_embedding.npz)')
    parser.add_argument('--basePath', type=str, default='../data', help='Path to TNG data')
    parser.add_argument('--threshold', type=float, default=5.0, help='Neighborhood threshold in Mpc/h')
    parser.add_argument('--n_components', type=int, default=10, help='Number of embedding dimensions')
    parser.add_argument('--top_n', type=int, default=10000, help='Keep this many most massive galaxies (by stellar mass)')
    parser.add_argument('--min_star_particles', type=int, default=500, help='Minimum number of star particles')
    parser.add_argument('--boxsize', type=float, default=75.0, help='Periodic box side length in Mpc/h (75 for TNG100, comoving, at every snapshot)')

    args = parser.parse_args()

    start_time = time.time()

    # Load and filter subhalos
    subhalos = load_and_filter_subhalos(
        args.basePath,
        args.snapshot,
        top_n=args.top_n,
        min_star_particles=args.min_star_particles
    )

    # Compute embedding
    embedding, sorted_neighbors = compute_embedding(
        subhalos,
        threshold=args.threshold,
        n_components=args.n_components,
        boxsize=args.boxsize
    )

    # Save results
    print(f"Saving to {args.output}...")
    np.savez(
        args.output,
        embedding=embedding,
        original_index=subhalos['original_index'],
        sorted_neighbors=sorted_neighbors,
        positions=subhalos['SubhaloPos'],
        snapshot=args.snapshot,
        threshold=args.threshold,
        n_neighbors_max=sorted_neighbors.shape[1],
        boxsize=args.boxsize
    )

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Done! Total time: {int(minutes)}m {seconds:.1f}s")


if __name__ == '__main__':
    main()
