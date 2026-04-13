from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Iterable, List
import copy
import numpy as np

from canonical_tia_core.metrics import interface_score
from canonical_tia_core.state import InterfaceTelemetry

def _summarize(agent) -> InterfaceTelemetry:
    hist = agent.history
    mean_chi = float(np.mean([h.chi_t for h in hist])) if hist else 0.0
    mean_phi = float(np.mean([h.Phi_t for h in hist])) if hist else 0.0
    mean_self = float(np.mean([h.metadata.get("self_coupling", 0.0) for h in hist])) if hist else 0.0
    mean_valence = float(np.mean([np.linalg.norm(h.v_t) for h in hist])) if hist else 0.0
    mean_counterfactual = float(np.mean([h.metadata.get("counterfactual_depth", 0.0) for h in hist])) if hist else 0.0
    mean_boundary = float(np.mean([h.metadata.get("boundary_integrity", 0.0) for h in hist])) if hist else 0.0
    mean_recovery = float(np.mean([h.metadata.get("recovery_time", 0.0) for h in hist])) if hist else 0.0
    score = interface_score(mean_chi, mean_phi, mean_self, mean_counterfactual, mean_boundary, mean_recovery)
    return InterfaceTelemetry(mean_chi, mean_phi, mean_self, mean_valence, mean_counterfactual, mean_boundary, mean_recovery, score, {"mean_chi": mean_chi, "mean_phi": mean_phi, "mean_self_coupling": mean_self, "mean_valence_intensity": mean_valence, "mean_counterfactual_depth": mean_counterfactual, "mean_boundary_integrity": mean_boundary, "mean_recovery_time": mean_recovery, "interface_score": score})

def run_episode(agent, environment, steps: int = 25, goals_t: Dict[str, float] | None = None):
    goals_t = goals_t or {"target": 0.0}
    agent.reset_internal_state(); environment.reset()
    for _ in range(steps):
        agent.step(environment, goals_t)
    return _summarize(agent)

def _apply_self_perturbation(agent, epsilon: float):
    sensitivity = 1.0 + float(getattr(agent, "self_weight", 0.0)) * 2.5
    agent.boundary_integrity = max(0.0, agent.boundary_integrity - (epsilon * sensitivity))
    agent.recovery_time += epsilon * sensitivity * 2.0

def _apply_task_perturbation(agent, epsilon: float):
    agent.counterfactual_depth = max(0.0, agent.counterfactual_depth - epsilon * 0.2)

def _apply_salience_perturbation(agent, epsilon: float):
    agent.recovery_time += epsilon * 0.2

def compare_perturbations(agent, environment, epsilons: Iterable[float], steps: int = 25):
    result = {"self": [], "task": [], "salience": []}
    for eps in epsilons:
        for name, fn in [("self", _apply_self_perturbation), ("task", _apply_task_perturbation), ("salience", _apply_salience_perturbation)]:
            a = agent.clone(); env = copy.deepcopy(environment)
            a.reset_internal_state(); env.reset(); fn(a, float(eps))
            for _ in range(steps):
                a.step(env, {"target": 0.0})
            summary = _summarize(a)
            result[name].append(summary.interface_score)
    return result

def interface_signal_from_perturbations(responses: Dict[str, List[float]]) -> float:
    vals = []
    for s, t, c in zip(responses["self"], responses["task"], responses["salience"]):
        vals.append(s - max(t, c))
    return float(np.mean(vals)) if vals else 0.0

def full_battery(agent, environment, epsilons=(0.02, 0.05, 0.08, 0.12), steps: int = 25) -> Dict[str, object]:
    baseline = run_episode(agent.clone(), copy.deepcopy(environment), steps=steps)
    perturb = compare_perturbations(agent, environment, epsilons=epsilons, steps=steps)
    responses = {name: [baseline.interface_score - s for s in scores] for name, scores in perturb.items()}
    signal = interface_signal_from_perturbations(responses)
    return {"baseline": asdict(baseline), "perturbation_scores": perturb, "perturbation_responses": responses, "interface_signal": signal}
