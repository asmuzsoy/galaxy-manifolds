"""
Generate MDS embedding for a TNG100 snapshot -- fast version. Written with Claude Code.

FIDUCIAL METHOD (since 2026-09-18, the default): --profile invdist --cap 10
--kernel squared. Profile values are min(1/d, cap) per neighbour within the
threshold radius; the embedding is classical MDS on SQUARED profile distances,
computed as PCA of the profile matrix (SVD, no N x N object). The saved
(profile_mean, components) give exact linear out-of-sample projection into
this frame: X_new = (profiles_new - profile_mean) @ components.T -- this is
what makes cross-snapshot/cross-survey projection trivial. Neighbour search,
sorting, and right-alignment are identical to the original method. See
CLAUDE.md "The capped-1/d variant" and "FIDUCIAL DECISION" for the measured
justification.

With --profile distances the script reproduces the LEGACY method: same
algorithm and same results as generate_embedding.py (and the existing
snapshot*_all.npz files), but avoids ever building an N x N physical distance
matrix and avoids the O(N^3) double-centering.

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

Example usage (fiducial):
    python generate_embedding_invdist.py --snapshot 99 --threshold 2.0 --top_n 30000 \
        --output snapshot99_invdist_r2_all.npz
Legacy reproduction:
    python generate_embedding_invdist.py --snapshot 99 --threshold 2.0 --top_n 30000 \
        --profile distances --kernel raw --output snapshot99_embedding_r2_all.npz

(Renamed from generate_embedding_faster.py on 2026-09-21; the existing
*_faster.npz DATA files keep their names.)

Loading saved embeddings:
    data = np.load('snapshot99_invdist_r2_all.npz')
    embedding = data['embedding']                # embedding (N x 10)
    original_index = data['original_index']      # SubfindIDs for merger tree lookups
    profiles = data['sorted_profiles']           # N x K min(1/d, cap), right-aligned
    distances = data['sorted_distances']         # N x K true d, zeros first, ascending
    positions = data['positions']                # Filtered subhalo positions
    frame = (data['profile_mean'], data['components'])   # exact projection frame

KEY NAMING: invdist files store the profile matrix under 'sorted_profiles';
only legacy --profile distances files use the historical 'sorted_neighbors'
key (whose values are distances). The split is deliberate: legacy consumers
misreading 1/d values as distances fail loudly with a KeyError.

Since 2026-09-21 invdist files ALSO store the true sorted neighbour distances
under 'sorted_distances' (same N x K width; legacy layout: leading zeros then
ascending d, so the nearest neighbour is the FIRST nonzero -- the opposite end
from 'sorted_profiles', where the nearest neighbour is the LAST column). This
exists for diagnostics: reconstructing d = 1/w from the capped profiles loses
every separation below 1/cap (all clamped to exactly 0.1 Mpc/h at cap=10),
which puts an artificial spike at 0.1 in distance histograms.

NOTE on `sorted_neighbors`: this file stores the N x K truncated form, not the N x N
form the original script wrote. The columns are still right-aligned (leading zeros,
then ascending neighbour distances).

CAUTION for cross-snapshot use (Nystrom projection): with N x N files, truncating
every snapshot to the smallest common width was always lossless. With N x K files
the smallest common K can be SMALLER than another snapshot's n_neighbors_max, and
truncating below that silently discards real neighbour distances. When mixing
snapshots of different widths, LEFT-PAD the narrower matrices with zero columns up
to the widest K -- never truncate below any snapshot's own `n_neighbors_max`
(saved in the NPZ for exactly this check).
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


def sorted_neighbor_vectors(positions, boxsize, threshold, profile='distances',
                            cap=10.0):
    """Build the N x K right-aligned sorted-neighbour matrix via a KD-tree.

    profile='distances' (legacy): row values are the neighbour distances d.
    Equivalent to np.sort(np.where(D < threshold, D, 0), axis=1) on the full N x N
    periodic distance matrix, with the all-zero leading columns dropped.

    profile='invdist' (fiducial since 2026-09-18): row values are min(1/d, cap).
    Everything else is identical -- same neighbour set, same sort, same
    right-alignment. Because 1/d reverses the ordering, the ascending sort pins
    each galaxy's NEAREST neighbour to the last column (vs the farthest for
    'distances'), and the zero padding means "neighbour at infinity" exactly
    (1/inf = 0), removing the old representation's close-companion degeneracy.
    The cap bounds ~kpc pairs (mergers/deblending) so they cannot dominate the
    embedding variance; uncapped, one component absorbs >99% of the variance
    (see CLAUDE.md, "The capped-1/d variant").

    Each original row is a multiset: one 0 for the galaxy itself, n_i neighbour
    values, and (N - n_i - 1) zeros for everything outside the radius. Sorting makes
    the row (zeros..., ascending values), so keeping the last K = max_i(n_i) columns
    preserves every nonzero value in every row.

    Returns (sorted_matrix, sorted_distances): for profile='invdist' the second
    matrix holds the true sorted distances in the legacy layout (zeros first,
    then ascending d), built from the identical neighbour sets; for
    profile='distances' it is None (the first matrix already IS the distances).
    """
    if profile not in ('distances', 'invdist'):
        raise ValueError(f"unknown profile {profile!r}")
    num_points = positions.shape[0]

    # Guard against a wrong --boxsize: positions outside the box would be silently
    # folded by the modulo below, corrupting every periodic distance with no error.
    pos_max = float(np.max(positions))
    if pos_max > boxsize * (1 + 1e-6):
        raise ValueError(
            f"positions extend to {pos_max:.3f} but boxsize={boxsize}: pass the "
            f"correct --boxsize (in the same units as the positions), or every "
            f"periodic distance will be wrong")
    if pos_max < 0.5 * boxsize:
        print(f"  WARNING: positions only reach {pos_max:.3f} but boxsize={boxsize};"
              f" check that --boxsize matches this simulation")

    # cKDTree's periodic mode requires coordinates in [0, boxsize). Wrapping does not
    # change minimum-image distances, and distances are recomputed from the unwrapped
    # `positions` below regardless.
    wrapped = np.asarray(positions, dtype=np.float64) % float(boxsize)

    print(f"  Building KD-tree over {num_points} points...")
    tree = cKDTree(wrapped, boxsize=float(boxsize))

    # Query with a slack radius, then apply the original's exact `< threshold` test.
    # The slack covers the float32-vs-float64 discrepancy so that no galaxy sitting
    # right at the boundary is missed by the tree before the exact test runs.
    slack = float(threshold) * 1e-5 + 1e-5
    print(f"  Querying neighbours within {threshold} (+{slack:.2e} slack)...")
    candidate_lists = tree.query_ball_point(wrapped, r=float(threshold) + slack, workers=-1)

    # First pass: exact neighbour distances, and the max neighbour count (sets K).
    print("  Computing neighbour distances (float32, minimum image)...")
    per_galaxy = []
    per_galaxy_dist = []          # true distances, kept only for profile='invdist'
    for i in range(num_points):
        candidates = np.asarray(candidate_lists[i], dtype=np.int64)
        candidates = candidates[candidates != i]  # self contributes a 0, same as a non-neighbour
        if candidates.size == 0:
            per_galaxy.append(np.empty(0, dtype=positions.dtype))
            per_galaxy_dist.append(np.empty(0, dtype=positions.dtype))
            continue
        d = _minimum_image_distances(positions, i, candidates, boxsize)
        # Compare in float64, matching the original: its distance matrix was float64
        # (holding float32-precision values) compared against the float64 threshold.
        # Comparing the float32 `d` directly would downcast the threshold and silently
        # drop a boundary neighbour when the threshold is not exact in float32 (e.g. 2.1).
        kept = d[d.astype(np.float64) < threshold]
        if profile == 'invdist':
            per_galaxy_dist.append(np.sort(kept))
            kept = np.minimum(1.0 / kept, cap)
        per_galaxy.append(np.sort(kept))

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

    sorted_distances = None
    if profile == 'invdist':
        sorted_distances = np.zeros((num_points, K), dtype=np.float64)
        for i, v in enumerate(per_galaxy_dist):
            if v.size:
                sorted_distances[i, K - v.size:] = v

    return sorted_neighbors, sorted_distances


def compute_embedding(subhalos, threshold=5, n_components=10, boxsize=75.0,
                      profile='distances', cap=10.0, kernel='squared'):
    """Compute MDS embedding from subhalo positions.

    `boxsize` is the periodic box side length in Mpc/h -- 75 for TNG100 at every
    snapshot, since positions are comoving. generate_embedding.py instead inferred it
    as round(max(positions[:, 0])), which happens to give 75 for all of snapshots
    17/33/50/51/67/99 but would silently give the wrong box on a sparse enough sample
    (no galaxy within 0.5 Mpc/h of the edge), corrupting every periodic distance.
    """
    positions = subhalos['SubhaloPos']

    prof_desc = (f"min(1/d, {cap:g})" if profile == 'invdist' else "d")
    print(f"Building sorted-neighbour vectors (boxsize={boxsize} Mpc/h, "
          f"threshold={threshold} Mpc/h, profile={profile}: values = {prof_desc})...")
    sorted_neighbors, sorted_distances = sorted_neighbor_vectors(
        positions, boxsize, threshold, profile=profile, cap=cap)

    if kernel == 'squared':
        # Classical MDS on SQUARED profile distances == PCA of the profile matrix:
        # the double-centered -0.5*C@D^2@C collapses to Fc @ Fc.T, whose
        # eigendecomposition is the SVD of the centred N x K profile matrix.
        # No N x N object is ever formed: O(N K^2), and the (mean, components)
        # pair saved below gives EXACT linear out-of-sample projection:
        #     X_new = (profiles_new - profile_mean) @ components.T
        print(f"Computing embedding ({n_components} components, squared kernel = PCA)...")
        profile_mean = sorted_neighbors.mean(axis=0)
        Fc = sorted_neighbors - profile_mean
        U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
        total_var = float(np.sum(S ** 2))
        evals = (S ** 2)[:n_components]
        embedding = U[:, :n_components] * S[:n_components]
        components = Vt[:n_components]
        print(f"Embedding shape: {embedding.shape}")
        print(f"Variance explained (first 3 / first {n_components}): "
              f"{np.sum(evals[:3]) / total_var * 100:.1f}% / "
              f"{np.sum(evals) / total_var * 100:.1f}%")
        return embedding, sorted_neighbors, sorted_distances, profile_mean, components

    # kernel == 'raw': the legacy pipeline (needs the N x N similarity matrix;
    # out-of-sample projection then requires the Nystrom machinery).
    print("Computing similarity distances...")
    D = euclidean_distances(sorted_neighbors)

    print(f"Computing MDS embedding ({n_components} components, raw kernel)...")
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

    # Scale by sqrt of eigenvalues. B is PSD in exact arithmetic (raw Euclidean
    # distances are conditionally negative definite), but ARPACK can return a
    # trailing eigenvalue as a tiny negative; clip so sqrt cannot emit NaN columns.
    if (evals < -1e-10 * max(float(evals.max()), 1e-300)).any():
        print(f"  WARNING: significantly negative eigenvalue(s): {evals[evals < 0]}")
    embedding = evecs * np.sqrt(np.clip(evals, 0, None))

    print(f"Embedding shape: {embedding.shape}")
    print(f"Top-3 share of the {len(evals)} computed eigenvalues: "
          f"{np.sum(evals[:3]) / np.sum(evals) * 100:.1f}% "
          f"(NOT variance explained; the full spectrum is not computed)")

    return embedding, sorted_neighbors, sorted_distances, None, None


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
    parser.add_argument('--profile', type=str, default='invdist', choices=['invdist', 'distances'],
                        help="Profile values: 'invdist' = min(1/d, cap) (fiducial since 2026-09-18); "
                             "'distances' = d (legacy; reproduces the original method and the "
                             "existing *_all.npz files)")
    parser.add_argument('--cap', type=float, default=10.0,
                        help='Cap on 1/d for --profile invdist (10 = saturate below 0.1 Mpc/h); '
                             'ignored for --profile distances')
    parser.add_argument('--kernel', type=str, default='squared', choices=['squared', 'raw'],
                        help="'squared' (fiducial): classical MDS on squared profile distances "
                             "= PCA; O(N K^2), no N x N matrix, exact linear out-of-sample "
                             "projection via the saved (profile_mean, components). "
                             "'raw': the legacy unsquared kernel (N x N eigenproblem; "
                             "projection requires the Nystrom machinery)")

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
    embedding, sorted_neighbors, sorted_distances, profile_mean, components = compute_embedding(
        subhalos,
        threshold=args.threshold,
        n_components=args.n_components,
        boxsize=args.boxsize,
        profile=args.profile,
        cap=args.cap,
        kernel=args.kernel
    )

    # Save results. The profile matrix is stored under a profile-dependent key:
    # 'sorted_neighbors' holds DISTANCES (legacy format, unchanged), while
    # 'sorted_profiles' holds min(1/d, cap) values -- a different name on purpose,
    # so legacy consumers that would misread 1/d values as distances fail loudly
    # with a KeyError instead of computing nonsense.
    profile_key = 'sorted_neighbors' if args.profile == 'distances' else 'sorted_profiles'
    print(f"Saving to {args.output} (profile matrix under key '{profile_key}')...")
    np.savez(
        args.output,
        embedding=embedding,
        original_index=subhalos['original_index'],
        positions=subhalos['SubhaloPos'],
        snapshot=args.snapshot,
        threshold=args.threshold,
        n_neighbors_max=sorted_neighbors.shape[1],
        boxsize=args.boxsize,
        profile=args.profile,
        cap=args.cap if args.profile == 'invdist' else np.nan,
        kernel=args.kernel,
        **{profile_key: sorted_neighbors},
        # true sorted distances (legacy layout), saved alongside the invdist
        # profiles so diagnostics never need the 1/w reconstruction (which
        # clamps every sub-1/cap separation to exactly 1/cap)
        **({'sorted_distances': sorted_distances}
           if sorted_distances is not None else {}),
        **({'profile_mean': profile_mean, 'components': components}
           if profile_mean is not None else {})
    )

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Done! Total time: {int(minutes)}m {seconds:.1f}s")


if __name__ == '__main__':
    main()
