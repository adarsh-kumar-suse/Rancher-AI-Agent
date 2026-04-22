from __future__ import annotations

import json
from typing import Any, Dict, List

from schemas import ClusterSnapshot, Plan, Action
from llm import OllamaClient


def build_candidate_actions(snapshot: ClusterSnapshot) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []

    for dep in snapshot.deployments:
        replicas = dep.replicas or 0
        cpu_util = dep.cpu_utilization or 0.0
        container = dep.containers[0].name if dep.containers else "main"

        if cpu_util >= 0.80 and replicas < 10:
            actions.append(
                {
                    "type": "scale",
                    "namespace": dep.namespace,
                    "name": dep.name,
                    "replicas": replicas + 1,
                    "reason": f"CPU utilization is high ({cpu_util:.2f}). Scale up by 1.",
                }
            )

        if cpu_util <= 0.25 and replicas > 1:
            actions.append(
                {
                    "type": "scale",
                    "namespace": dep.namespace,
                    "name": dep.name,
                    "replicas": replicas - 1,
                    "reason": f"CPU utilization is low ({cpu_util:.2f}). Scale down by 1.",
                }
            )

        if dep.containers:
            c = dep.containers[0]
            if c.cpu_request_m > 0 and cpu_util >= 1.10:
                new_req = int(max(c.cpu_request_m * 1.25, c.cpu_request_m + 10))
                new_lim = int(max(new_req, c.cpu_limit_m or (new_req * 2)))
                actions.append(
                    {
                        "type": "patch_resources",
                        "namespace": dep.namespace,
                        "name": dep.name,
                        "container": container,
                        "cpu_request_m": new_req,
                        "cpu_limit_m": new_lim,
                        "reason": "CPU usage is above request, increase request to reduce throttling.",
                    }
                )

    return actions


def build_plan(snapshot: ClusterSnapshot, ollama: OllamaClient) -> Plan:
    candidates = build_candidate_actions(snapshot)

    if not candidates:
        return Plan(summary="No optimization needed right now.", actions=[])

    if not ollama.is_available():
        return Plan(summary="Ollama unavailable; using heuristic actions only.", actions=[Action(**a) for a in candidates])

    system = (
        "You are a Kubernetes workload optimizer. "
        "You will receive a snapshot and candidate actions. "
        "Return strict JSON with keys: summary (string), chosen_indices (array of integers). "
        "Choose only from the candidate actions."
    )

    user = json.dumps(
        {
            "snapshot": snapshot.model_dump(),
            "candidate_actions": candidates,
            "goal": "Optimize workload efficiency and keep changes minimal.",
        },
        indent=2,
    )

    try:
        result = ollama.chat_json(system=system, user=user)
        summary = str(result.get("summary", "")).strip() or "Optimization plan generated."
        chosen = result.get("chosen_indices", [])
        selected: List[Action] = []

        for idx in chosen:
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                selected.append(Action(**candidates[idx]))

        if not selected:
            selected = [Action(**a) for a in candidates]

        return Plan(summary=summary, actions=selected)

    except Exception:
        return Plan(summary="Model output could not be parsed; using heuristic actions.", actions=[Action(**a) for a in candidates])
