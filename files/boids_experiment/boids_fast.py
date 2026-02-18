"""
Optimized Boids Simulation - O(n) using spatial hashing + NumPy vectorization

Supports 500+ boids efficiently.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.spatial import cKDTree


@dataclass
class BoidsConfig:
    """Configuration for Boids simulation"""
    n_boids: int = 500
    world_size: float = 200.0
    boundary: str = "periodic"
    max_speed: float = 2.0
    dt: float = 0.1
    perception_radius: float = 15.0
    alignment_weight: float = 1.0
    cohesion_weight: float = 0.0  # Default off for FCC test
    separation_weight: float = 1.0
    separation_radius: float = 3.0
    noise_std: float = 0.1

    def to_dict(self):
        return self.__dict__.copy()


class BoidsFlock:
    """Optimized Boids using KD-tree for O(n log n) neighbor queries"""

    def __init__(self, config: BoidsConfig):
        self.config = config
        self.positions = None
        self.velocities = None
        self.reset()

    def reset(self):
        """Initialize random positions and velocities"""
        c = self.config
        self.positions = np.random.uniform(0, c.world_size, (c.n_boids, 2))
        angles = np.random.uniform(0, 2 * np.pi, c.n_boids)
        speeds = np.random.uniform(0.5, 1.0, c.n_boids) * c.max_speed
        self.velocities = np.column_stack([
            speeds * np.cos(angles),
            speeds * np.sin(angles)
        ])

    def _wrap_positions(self, positions):
        """Handle periodic boundary"""
        return positions % self.config.world_size

    def _periodic_distance(self, pos1, pos2):
        """Compute distances with periodic boundary"""
        diff = pos1 - pos2
        ws = self.config.world_size
        diff = np.where(diff > ws/2, diff - ws, diff)
        diff = np.where(diff < -ws/2, diff + ws, diff)
        return diff

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run one simulation step using vectorized operations"""
        c = self.config
        n = c.n_boids

        # Build KD-tree for efficient neighbor queries
        # For periodic boundaries, we need to handle wrapping
        tree = cKDTree(self.positions, boxsize=c.world_size)

        # Query all neighbors within perception radius
        neighbor_lists = tree.query_ball_point(self.positions, c.perception_radius)

        # Initialize acceleration arrays
        alignment_acc = np.zeros((n, 2))
        cohesion_acc = np.zeros((n, 2))
        separation_acc = np.zeros((n, 2))
        neighbor_counts = np.zeros(n)

        # Process each boid
        for i in range(n):
            neighbors = neighbor_lists[i]
            neighbors = [j for j in neighbors if j != i]  # Exclude self

            if len(neighbors) == 0:
                continue

            neighbor_counts[i] = len(neighbors)
            neighbor_idx = np.array(neighbors)

            # Get neighbor data
            neighbor_pos = self.positions[neighbor_idx]
            neighbor_vel = self.velocities[neighbor_idx]

            # Periodic position differences
            diff = self._periodic_distance(neighbor_pos, self.positions[i])

            # Alignment: steer toward average heading
            avg_vel = neighbor_vel.mean(axis=0)
            alignment_acc[i] = avg_vel - self.velocities[i]

            # Cohesion: steer toward center of mass
            if c.cohesion_weight > 0:
                center_diff = diff.mean(axis=0)
                cohesion_acc[i] = center_diff * 0.1

            # Separation: steer away from close neighbors
            distances = np.linalg.norm(diff, axis=1)
            close_mask = distances < c.separation_radius
            if close_mask.any():
                close_diff = diff[close_mask]
                close_dist = distances[close_mask].reshape(-1, 1)
                # Inverse square repulsion
                repulsion = -close_diff / (close_dist ** 2 + 0.01)
                separation_acc[i] = repulsion.sum(axis=0)

        # Combine accelerations
        total_acc = (
            c.alignment_weight * alignment_acc +
            c.cohesion_weight * cohesion_acc +
            c.separation_weight * separation_acc
        )

        # Add noise
        total_acc += np.random.normal(0, c.noise_std, (n, 2))

        # Update velocities
        self.velocities = self.velocities + total_acc * c.dt

        # Limit speed (vectorized)
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        too_fast = speeds > c.max_speed
        self.velocities = np.where(
            too_fast,
            self.velocities * c.max_speed / (speeds + 1e-10),
            self.velocities
        )

        # Update positions
        self.positions = self.positions + self.velocities * c.dt
        self.positions = self._wrap_positions(self.positions)

        return self.positions.copy(), self.velocities.copy()

    def get_state(self) -> dict:
        return {
            'positions': self.positions.tolist(),
            'velocities': self.velocities.tolist()
        }


def run_simulation(config: BoidsConfig, n_steps: int, record_every: int = 1) -> List[dict]:
    """Run simulation and record states"""
    flock = BoidsFlock(config)
    history = []

    for step in range(n_steps):
        positions, velocities = flock.step()
        if step % record_every == 0:
            history.append({
                'step': step,
                'positions': positions.copy(),
                'velocities': velocities.copy()
            })

    return history


# Even faster version using pure NumPy (no KD-tree, uses grid hashing)
class BoidsFastGrid:
    """Ultra-fast Boids using spatial grid hashing - O(n) average"""

    def __init__(self, config: BoidsConfig):
        self.config = config
        self.positions = None
        self.velocities = None
        self.cell_size = config.perception_radius
        self.grid_size = int(np.ceil(config.world_size / self.cell_size))
        self.reset()

    def reset(self):
        c = self.config
        self.positions = np.random.uniform(0, c.world_size, (c.n_boids, 2))
        angles = np.random.uniform(0, 2 * np.pi, c.n_boids)
        speeds = np.random.uniform(0.5, 1.0, c.n_boids) * c.max_speed
        self.velocities = np.column_stack([
            speeds * np.cos(angles),
            speeds * np.sin(angles)
        ])

    def _get_cell_indices(self, positions):
        """Get grid cell indices for all positions"""
        return (positions / self.cell_size).astype(int) % self.grid_size

    def _build_grid(self):
        """Build spatial hash grid"""
        cell_idx = self._get_cell_indices(self.positions)
        # Convert 2D cell index to 1D hash
        cell_hash = cell_idx[:, 0] * self.grid_size + cell_idx[:, 1]

        # Build dictionary: cell_hash -> list of boid indices
        grid = {}
        for i, h in enumerate(cell_hash):
            if h not in grid:
                grid[h] = []
            grid[h].append(i)

        return grid, cell_idx

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        c = self.config
        n = c.n_boids
        ws = c.world_size

        # Build spatial grid
        grid, cell_indices = self._build_grid()

        # Initialize accumulators
        alignment_acc = np.zeros((n, 2))
        cohesion_acc = np.zeros((n, 2))
        separation_acc = np.zeros((n, 2))

        # Process each boid
        for i in range(n):
            cx, cy = cell_indices[i]

            # Check neighboring cells (3x3 neighborhood)
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx = (cx + dx) % self.grid_size
                    ny = (cy + dy) % self.grid_size
                    cell_hash = nx * self.grid_size + ny
                    if cell_hash in grid:
                        neighbors.extend(grid[cell_hash])

            # Remove self and filter by actual distance
            pos_i = self.positions[i]
            valid_neighbors = []

            for j in neighbors:
                if j == i:
                    continue
                # Periodic distance
                diff = self.positions[j] - pos_i
                if diff[0] > ws/2: diff[0] -= ws
                if diff[0] < -ws/2: diff[0] += ws
                if diff[1] > ws/2: diff[1] -= ws
                if diff[1] < -ws/2: diff[1] += ws
                dist = np.sqrt(diff[0]**2 + diff[1]**2)

                if dist < c.perception_radius:
                    valid_neighbors.append((j, diff, dist))

            if len(valid_neighbors) == 0:
                continue

            # Compute forces
            neighbor_vels = np.array([self.velocities[j] for j, _, _ in valid_neighbors])
            neighbor_diffs = np.array([d for _, d, _ in valid_neighbors])
            neighbor_dists = np.array([dist for _, _, dist in valid_neighbors])

            # Alignment
            avg_vel = neighbor_vels.mean(axis=0)
            alignment_acc[i] = avg_vel - self.velocities[i]

            # Cohesion
            if c.cohesion_weight > 0:
                cohesion_acc[i] = neighbor_diffs.mean(axis=0) * 0.1

            # Separation
            close_mask = neighbor_dists < c.separation_radius
            if close_mask.any():
                close_diffs = neighbor_diffs[close_mask]
                close_dists = neighbor_dists[close_mask].reshape(-1, 1)
                separation_acc[i] = -(close_diffs / (close_dists**2 + 0.01)).sum(axis=0)

        # Combine and update
        total_acc = (
            c.alignment_weight * alignment_acc +
            c.cohesion_weight * cohesion_acc +
            c.separation_weight * separation_acc +
            np.random.normal(0, c.noise_std, (n, 2))
        )

        self.velocities = self.velocities + total_acc * c.dt

        # Limit speed
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        too_fast = speeds > c.max_speed
        self.velocities = np.where(too_fast, self.velocities * c.max_speed / (speeds + 1e-10), self.velocities)

        # Update positions with periodic boundary
        self.positions = (self.positions + self.velocities * c.dt) % ws

        return self.positions.copy(), self.velocities.copy()


def run_simulation_fast(config: BoidsConfig, n_steps: int, record_every: int = 1) -> List[dict]:
    """Run optimized simulation"""
    flock = BoidsFastGrid(config)
    history = []

    for step in range(n_steps):
        positions, velocities = flock.step()
        if step % record_every == 0:
            history.append({
                'step': step,
                'positions': positions.copy(),
                'velocities': velocities.copy()
            })

    return history


if __name__ == "__main__":
    import time

    # Benchmark
    for n_boids in [100, 500, 1000]:
        config = BoidsConfig(n_boids=n_boids, world_size=200.0)

        start = time.time()
        history = run_simulation_fast(config, n_steps=100, record_every=10)
        elapsed = time.time() - start

        print(f"n_boids={n_boids}: {elapsed:.2f}s for 100 steps ({100/elapsed:.1f} steps/sec)")
