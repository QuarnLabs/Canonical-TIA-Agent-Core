from __future__ import annotations
import math
import numpy as np

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def mutual_information_proxy(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a.flatten().astype(float)
    b = b.flatten().astype(float)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(max(0.0, np.dot(a, b) / denom))

def interface_chi(phi_t: float, s_t: np.ndarray, z_t: np.ndarray, v_t: np.ndarray, a_t: float,
                  w_phi: float = 1.0, w_mi: float = 1.0, w_v: float = 1.0, w_a: float = 1.0) -> float:
    x = (w_phi * phi_t) + (w_mi * mutual_information_proxy(s_t, z_t)) + (w_v * safe_norm(v_t)) - (w_a * a_t)
    return sigmoid(x)

def interface_score(mean_chi: float, mean_phi: float, mean_self_coupling: float,
                    mean_counterfactual_depth: float, mean_boundary_integrity: float,
                    mean_recovery_time: float) -> float:
    recovery_penalty = 1.0 / (1.0 + max(0.0, mean_recovery_time))
    score = (0.22 * mean_chi + 0.18 * mean_phi + 0.22 * mean_self_coupling + 0.18 * mean_counterfactual_depth + 0.15 * mean_boundary_integrity + 0.05 * recovery_penalty)
    return float(score)
