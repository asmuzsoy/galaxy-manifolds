"""
Generate MDS embedding for a TNG100 snapshot. Written with Claude Code.

Example usage:
    # Generate embedding for z=0 (snapshot 99)
    python generate_embedding.py --snapshot 99 --output snapshot99_embedding.npz

    # Generate embedding for z=0.95 (snapshot 51)
    python generate_embedding.py --snapshot 51 --output snapshot51_embedding.npz

    # Custom threshold and data path
    python generate_embedding.py --snapshot 99 --output snapshot99_r3.npz --threshold 3.0 --basePath /path/to/TNG100

Loading saved embeddings:
    data = np.load('snapshot99_embedding.npz')
    embedding = data['embedding']           # MDS embedding (N x 10)
    original_index = data['original_index'] # SubfindIDs for merger tree lookups
    sorted_neighbors = data['sorted_neighbors']  # For OT cost matrix
    positions = data['positions']           # Filtered subhalo positions
"""

import argparse
import time
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.linalg import eigsh
from astropy import units as u
import illustris_python as il


def periodic_distance_matrix(xs, ys, zs, boxsize):
    """Compute distance matrix with periodic boundary conditions."""
    points = np.vstack((xs, ys, zs)).T
    num_points = points.shape[0]
    dist_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        delta = points - points[i]
        delta = delta - boxsize * np.round(delta / boxsize)
        dist_matrix[i] = np.sqrt(np.sum(delta**2, axis=1))
    return dist_matrix


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


def compute_embedding(subhalos, threshold=5, n_components=10):
    """Compute MDS embedding from subhalo positions."""
    positions = subhalos['SubhaloPos']
    boxsize = round(max(positions[:, 0]))

    print(f"Computing distance matrix (boxsize={boxsize} Mpc/h)...")
    distance_matrix = periodic_distance_matrix(
        positions[:, 0], positions[:, 1], positions[:, 2], boxsize
    )

    print(f"Thresholding at {threshold} Mpc/h...")
    nearest_neighbor_matrix = np.where(distance_matrix < threshold, distance_matrix, 0)

    print("Computing similarity distances...")
    sorted_neighbors = np.sort(nearest_neighbor_matrix, axis=1)
    similarity_distances = euclidean_distances(sorted_neighbors)

    print(f"Computing MDS embedding ({n_components} components)...")
    D = similarity_distances
    num_points = D.shape[0]
    C = np.eye(num_points) - (1 / num_points) * np.ones_like(D)
    B = -0.5 * np.matmul(C, np.matmul(D, C))

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
    parser = argparse.ArgumentParser(description='Generate MDS embedding for TNG100 snapshot')
    parser.add_argument('--snapshot', type=int, required=True, help='Snapshot number (e.g., 99 for z=0, 51 for z=0.95)')
    parser.add_argument('--output', type=str, required=True, help='Output filename (e.g., snapshot99_embedding.npz)')
    parser.add_argument('--basePath', type=str, default='../data', help='Path to TNG data')
    parser.add_argument('--threshold', type=float, default=5.0, help='Neighborhood threshold in Mpc/h')
    parser.add_argument('--n_components', type=int, default=10, help='Number of embedding dimensions')
    parser.add_argument('--top_n', type=int, default=10000, help='Keep this many most massive galaxies (by stellar mass)')
    parser.add_argument('--min_star_particles', type=int, default=500, help='Minimum number of star particles')

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
        n_components=args.n_components
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
        threshold=args.threshold
    )

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Done! Total time: {int(minutes)}m {seconds:.1f}s")


if __name__ == '__main__':
    main()
