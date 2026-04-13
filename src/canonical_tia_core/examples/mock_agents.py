from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

from canonical_tia_core.core import BaseTIAAgent

@dataclass
class MockEnvironment:
    state: float = 0.0
    drift: float = 0.03
    def observe(self): return np.array([self.state], dtype=float)
    def step(self, action):
        self.state += float(action[0]) + self.drift
        self.state = float(np.clip(self.state, -1.5, 1.5))
    def reset(self): self.state = 0.0

class _BaseMockAgent(BaseTIAAgent):
    def __init__(self, latent_dim=4, seed=0, *, self_weight=0.5, counterfactual_depth=0.4, access_base=0.2):
        super().__init__(latent_dim=latent_dim, seed=seed)
        self.self_weight = self_weight; self.counterfactual_depth = counterfactual_depth; self.access_base = access_base; self.boundary_integrity = 0.5 + self_weight * 0.4
    def observe(self, environment: MockEnvironment): return environment.observe()
    def encoder(self, x_t, h_prev):
        z = np.zeros(self.latent_dim); z[0] = float(x_t[0]); z[1] = float(h_prev[0]) if len(h_prev) else 0.0; z[2] = float(self.counterfactual_depth); z[3] = float(self.boundary_integrity); return z
    def self_model(self, z_t, s_prev, h_prev): return self.self_weight * z_t + (1.0 - self.self_weight) * s_prev
    def loss_fn(self, z_t, s_t, goals_t: Dict[str, float]):
        target = float(goals_t.get("target", 0.0)); return float((z_t[0] - target) ** 2 + 0.2 * (1.0 - self.boundary_integrity))
    def grad_z(self, z_t, s_t, goals_t: Dict[str, float]):
        target = float(goals_t.get("target", 0.0)); grad = np.zeros_like(z_t); grad[0] = -2.0 * (z_t[0] - target); grad[3] = self.self_weight * (self.boundary_integrity - 1.0); return grad
    def integration_metric(self, z_t, z_prev):
        delta = np.linalg.norm(z_t - z_prev); return float(1.0 / (1.0 + delta))
    def access_metric(self, model_state): return float(np.clip(self.access_base + (1.0 - self.boundary_integrity) * 0.2, 0.0, 1.0))
    def interface_synthesis(self, z_t, s_t, v_t, phi_t, a_t): return 0.4 * z_t + 0.3 * s_t + 0.2 * v_t + 0.1 * np.array([phi_t, a_t, self.counterfactual_depth, self.boundary_integrity])
    def policy(self, z_t, s_t, q_t): return np.array([-0.5 * z_t[0] - 0.2 * q_t[0]], dtype=float)

class MockTaskAgent(_BaseMockAgent):
    def __init__(self, latent_dim=4, seed=0):
        super().__init__(latent_dim=latent_dim, seed=seed, self_weight=0.15, counterfactual_depth=0.2, access_base=0.35); self.boundary_integrity = 0.35

class MockInterfaceAgent(_BaseMockAgent):
    def __init__(self, latent_dim=4, seed=0):
        super().__init__(latent_dim=latent_dim, seed=seed, self_weight=0.7, counterfactual_depth=0.75, access_base=0.12); self.boundary_integrity = 0.82
