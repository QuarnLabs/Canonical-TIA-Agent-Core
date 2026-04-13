from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np

Array = np.ndarray

@dataclass
class AgentTimestep:
    x_t: Array
    z_t: Array
    s_t: Array
    L_t: float
    v_t: Array
    Phi_t: float
    A_t: float
    q_t: Array
    chi_t: float
    u_t: Array
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InterfaceTelemetry:
    mean_chi: float
    mean_phi: float
    mean_self_coupling: float
    mean_valence_intensity: float
    mean_counterfactual_depth: float
    mean_boundary_integrity: float
    mean_recovery_time: float
    interface_score: float
    raw: Dict[str, float] = field(default_factory=dict)
