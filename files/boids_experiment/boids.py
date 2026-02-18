"""
Boids Flocking Simulation for FCC Falsification Test

Core simulation with tunable alignment parameter (alpha) for phase transition testing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List
import json


@dataclass
class BoidsConfig:
    """Configuration for Boids simulation"""
    # World
    n_boids: int = 100
    world_size: float = 100.0
    boundary: str = "periodic"  # "periodic" or "reflective"

    # Dynamics
    max_speed: float = 2.0
    dt: float = 0.1

    # Feedback (fixed)
    perception_radius: float = 10.0

    # Constraint (swept) - PRIMARY PARAMETER
    alignment_weight: float = 1.0  # alpha
    cohesion_weight: float = 1.0
    separation_weight: float = 1.5
    separation_radius: float = 2.0

    # Noise
    noise_std: float = 0.1

    def to_dict(self):
        return {
            'n_boids': self.n_boids,
            'world_size': self.world_size,
            'boundary': self.boundary,
            'max_speed': self.max_speed,
            'dt': self.dt,
            'perception_radius': self.perception_radius,
            'alignment_weight': self.alignment_weight,
            'cohesion_weight': self.cohesion_weight,
            'separation_weight': self.separation_weight,
            'separation_radius': self.separation_radius,
            'noise_std': self.noise_std
        }


class BoidsFlock:
    """Boids flocking simulation"""

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

    def get_neighbors(self, idx: int) -> np.ndarray:
        """Get indices of neighbors within perception radius"""
        c = self.config
        pos = self.positions[idx]

        # Handle periodic boundary
        if c.boundary == "periodic":
            diff = self.positions - pos
            diff = np.where(diff > c.world_size/2, diff - c.world_size, diff)
            diff = np.where(diff < -c.world_size/2, diff + c.world_size, diff)
            distances = np.linalg.norm(diff, axis=1)
        else:
            distances = np.linalg.norm(self.positions - pos, axis=1)

        neighbors = np.where((distances < c.perception_radius) & (distances > 0))[0]
        return neighbors

    def align(self, idx: int, neighbors: np.ndarray) -> np.ndarray:
        """Alignment: steer toward average heading of neighbors"""
        if len(neighbors) == 0:
            return np.zeros(2)
        avg_velocity = self.velocities[neighbors].mean(axis=0)
        return avg_velocity - self.velocities[idx]

    def cohere(self, idx: int, neighbors: np.ndarray) -> np.ndarray:
        """Cohesion: steer toward center of mass of neighbors"""
        if len(neighbors) == 0:
            return np.zeros(2)
        c = self.config
        center = self.positions[neighbors].mean(axis=0)

        # Handle periodic boundary
        diff = center - self.positions[idx]
        if c.boundary == "periodic":
            if diff[0] > c.world_size/2: diff[0] -= c.world_size
            if diff[0] < -c.world_size/2: diff[0] += c.world_size
            if diff[1] > c.world_size/2: diff[1] -= c.world_size
            if diff[1] < -c.world_size/2: diff[1] += c.world_size

        return diff * 0.1  # Scale factor

    def separate(self, idx: int, neighbors: np.ndarray) -> np.ndarray:
        """Separation: steer away from close neighbors"""
        c = self.config
        if len(neighbors) == 0:
            return np.zeros(2)

        steer = np.zeros(2)
        pos = self.positions[idx]

        for n in neighbors:
            diff = pos - self.positions[n]
            # Handle periodic boundary
            if c.boundary == "periodic":
                if diff[0] > c.world_size/2: diff[0] -= c.world_size
                if diff[0] < -c.world_size/2: diff[0] += c.world_size
                if diff[1] > c.world_size/2: diff[1] -= c.world_size
                if diff[1] < -c.world_size/2: diff[1] += c.world_size

            dist = np.linalg.norm(diff)
            if dist < c.separation_radius and dist > 0:
                steer += diff / (dist * dist)  # Inverse square

        return steer

    def limit_speed(self, velocity: np.ndarray) -> np.ndarray:
        """Limit velocity to max_speed"""
        speed = np.linalg.norm(velocity)
        if speed > self.config.max_speed:
            return velocity * (self.config.max_speed / speed)
        return velocity

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run one simulation step, return new positions and velocities"""
        c = self.config
        new_velocities = np.zeros_like(self.velocities)

        for i in range(c.n_boids):
            neighbors = self.get_neighbors(i)

            if len(neighbors) > 0:
                alignment = c.alignment_weight * self.align(i, neighbors)
                cohesion = c.cohesion_weight * self.cohere(i, neighbors)
                separation = c.separation_weight * self.separate(i, neighbors)
                delta_v = alignment + cohesion + separation
            else:
                delta_v = np.zeros(2)

            # Add noise
            delta_v += np.random.normal(0, c.noise_std, 2)

            new_v = self.velocities[i] + delta_v * c.dt
            new_velocities[i] = self.limit_speed(new_v)

        self.velocities = new_velocities
        self.positions = self.positions + self.velocities * c.dt

        # Apply boundary conditions
        if c.boundary == "periodic":
            self.positions = self.positions % c.world_size
        else:
            # Reflective boundary
            for d in range(2):
                mask_low = self.positions[:, d] < 0
                mask_high = self.positions[:, d] > c.world_size
                self.positions[mask_low, d] = -self.positions[mask_low, d]
                self.velocities[mask_low, d] = -self.velocities[mask_low, d]
                self.positions[mask_high, d] = 2*c.world_size - self.positions[mask_high, d]
                self.velocities[mask_high, d] = -self.velocities[mask_high, d]

        return self.positions.copy(), self.velocities.copy()

    def get_state(self) -> dict:
        """Get current state as JSON-serializable dict"""
        return {
            'positions': self.positions.tolist(),
            'velocities': self.velocities.tolist(),
            'config': self.config.to_dict()
        }


def run_simulation(config: BoidsConfig, n_steps: int,
                   record_every: int = 1) -> List[dict]:
    """
    Run simulation and record states.

    Returns list of {positions, velocities} at each recorded step.
    """
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


if __name__ == "__main__":
    # Quick test
    config = BoidsConfig(n_boids=50, alignment_weight=1.5)
    history = run_simulation(config, n_steps=100)
    print(f"Ran {len(history)} steps")
    print(f"Final positions shape: {history[-1]['positions'].shape}")
