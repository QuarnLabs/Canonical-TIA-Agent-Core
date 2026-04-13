from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Dict, List
import copy
import numpy as np

from .metrics import interface_chi, mutual_information_proxy
from .state import AgentTimestep

Array = np.ndarray

class BaseTIAAgent(ABC):
    def __init__(self, latent_dim: int = 4, seed: int = 0):
        self.latent_dim = latent_dim
        self.rng = np.random.default_rng(seed)
        self.reset_internal_state()

    def reset_internal_state(self) -> None:
        self.h_prev = np.zeros(self.latent_dim)
        self.z_prev = np.zeros(self.latent_dim)
        self.s_prev = np.zeros(self.latent_dim)
        self.history: List[AgentTimestep] = []
        self.counterfactual_depth = 0.0
        self.boundary_integrity = 0.5
        self.recovery_time = 0.0

    def clone(self):
        return copy.deepcopy(self)

    @abstractmethod
    def observe(self, environment) -> Array: ...
    @abstractmethod
    def encoder(self, x_t: Array, h_prev: Array) -> Array: ...
    @abstractmethod
    def self_model(self, z_t: Array, s_prev: Array, h_prev: Array) -> Array: ...
    @abstractmethod
    def loss_fn(self, z_t: Array, s_t: Array, goals_t: Dict[str, float]) -> float: ...
    @abstractmethod
    def grad_z(self, z_t: Array, s_t: Array, goals_t: Dict[str, float]) -> Array: ...
    @abstractmethod
    def integration_metric(self, z_t: Array, z_prev: Array) -> float: ...
    @abstractmethod
    def access_metric(self, model_state: Dict[str, float]) -> float: ...
    @abstractmethod
    def interface_synthesis(self, z_t: Array, s_t: Array, v_t: Array, phi_t: float, a_t: float) -> Array: ...
    @abstractmethod
    def policy(self, z_t: Array, s_t: Array, q_t: Array) -> Array: ...

    def update_parameters(self) -> None:
        return None

    def model_state(self) -> Dict[str, float]:
        return {
            "counterfactual_depth": float(self.counterfactual_depth),
            "boundary_integrity": float(self.boundary_integrity),
            "recovery_time": float(self.recovery_time),
        }

    def step(self, environment, goals_t: Dict[str, float]) -> AgentTimestep:
        x_t = self.observe(environment)
        z_t = self.encoder(x_t, self.h_prev)
        s_t = self.self_model(z_t, self.s_prev, self.h_prev)
        L_t = self.loss_fn(z_t, s_t, goals_t)
        v_t = self.grad_z(z_t, s_t, goals_t)
        phi_t = self.integration_metric(z_t, self.z_prev)
        a_t = self.access_metric(self.model_state())
        q_t = self.interface_synthesis(z_t, s_t, v_t, phi_t, a_t)
        chi_t = interface_chi(phi_t, s_t, z_t, v_t, a_t)
        u_t = self.policy(z_t, s_t, q_t)
        environment.step(u_t)
        self.update_parameters()
        ts = AgentTimestep(
            x_t=x_t, z_t=z_t, s_t=s_t, L_t=float(L_t), v_t=v_t, Phi_t=float(phi_t), A_t=float(a_t), q_t=q_t, chi_t=float(chi_t), u_t=u_t,
            metadata={
                "self_coupling": mutual_information_proxy(s_t, z_t),
                "counterfactual_depth": float(self.counterfactual_depth),
                "boundary_integrity": float(self.boundary_integrity),
                "recovery_time": float(self.recovery_time),
            },
        )
        self.history.append(ts)
        self.h_prev = q_t.copy(); self.z_prev = z_t.copy(); self.s_prev = s_t.copy()
        return ts

class CanonicalTIAAgent(BaseTIAAgent):
    def __init__(self, latent_dim: int, seed: int = 0, *, observe_fn: Callable, encoder_fn: Callable, self_model_fn: Callable, loss_fn_impl: Callable, grad_fn: Callable, integration_metric_fn: Callable, access_metric_fn: Callable, interface_synthesis_fn: Callable, policy_fn: Callable):
        super().__init__(latent_dim=latent_dim, seed=seed)
        self._observe = observe_fn; self._encoder = encoder_fn; self._self_model = self_model_fn; self._loss = loss_fn_impl; self._grad = grad_fn; self._integration = integration_metric_fn; self._access = access_metric_fn; self._synthesis = interface_synthesis_fn; self._policy = policy_fn

    def observe(self, environment): return self._observe(environment)
    def encoder(self, x_t, h_prev): return self._encoder(x_t, h_prev)
    def self_model(self, z_t, s_prev, h_prev): return self._self_model(z_t, s_prev, h_prev)
    def loss_fn(self, z_t, s_t, goals_t): return self._loss(z_t, s_t, goals_t)
    def grad_z(self, z_t, s_t, goals_t): return self._grad(z_t, s_t, goals_t)
    def integration_metric(self, z_t, z_prev): return self._integration(z_t, z_prev)
    def access_metric(self, model_state): return self._access(model_state)
    def interface_synthesis(self, z_t, s_t, v_t, phi_t, a_t): return self._synthesis(z_t, s_t, v_t, phi_t, a_t)
    def policy(self, z_t, s_t, q_t): return self._policy(z_t, s_t, q_t)
