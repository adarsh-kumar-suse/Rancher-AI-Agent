from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from kubernetes import client, config
from kubernetes.client import ApiException
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("k8s-optimizer", json_response=True)


def _load_k8s() -> tuple[client.AppsV1Api, client.CoreV1Api, client.CustomObjectsApi]:
    kubeconfig_path = os.getenv("KUBECONFIG_PATH")

    if kubeconfig_path and os.path.exists(kubeconfig_path):
        config.load_kube_config(config_file=kubeconfig_path)
    else:
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()

    return client.AppsV1Api(), client.CoreV1Api(), client.CustomObjectsApi()


def _parse_cpu_to_m(value: Optional[str]) -> float:
    if not value:
        return 0.0
    v = value.strip()
    if v.endswith("m"):
        return float(v[:-1])
    return float(v) * 1000.0


def _parse_mem_to_mib(value: Optional[str]) -> float:
    if not value:
        return 0.0
    v = value.strip()
    if v.endswith("Ki"):
        return float(v[:-2]) / 1024.0
    if v.endswith("Mi"):
        return float(v[:-2])
    if v.endswith("Gi"):
        return float(v[:-2]) * 1024.0
    if v.endswith("Ti"):
        return float(v[:-2]) * 1024.0 * 1024.0
    return 0.0


def _pod_metrics(custom_api: client.CustomObjectsApi) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    try:
        data = custom_api.list_cluster_custom_object("metrics.k8s.io", "v1beta1", "pods")
    except Exception:
        return out

    for item in data.get("items", []):
        ns = item.get("metadata", {}).get("namespace", "")
        name = item.get("metadata", {}).get("name", "")
        cpu = 0.0
        mem = 0.0
        for c in item.get("containers", []):
            usage = c.get("usage", {})
            cpu += _parse_cpu_to_m(usage.get("cpu"))
            mem += _parse_mem_to_mib(usage.get("memory"))
        out[f"{ns}/{name}"] = {"cpu_m": cpu, "memory_mib": mem}
    return out


@mcp.tool()
def list_namespaces() -> str:
    apps, v1, custom = _load_k8s()
    namespaces = [ns.metadata.name for ns in v1.list_namespace().items]
    return json.dumps({"namespaces": namespaces}, indent=2)


@mcp.tool()
def list_deployments(namespace: str) -> str:
    apps, v1, custom = _load_k8s()
    deployments = []
    try:
        for dep in apps.list_namespaced_deployment(namespace).items:
            deployments.append(dep.metadata.name)
    except ApiException as exc:
        return json.dumps({"error": str(exc), "deployments": []}, indent=2)
    return json.dumps({"namespace": namespace, "deployments": deployments}, indent=2)


@mcp.tool()
def get_deployment_snapshot(namespace: str, name: str) -> str:
    apps, v1, custom = _load_k8s()
    metrics = _pod_metrics(custom)

    dep = apps.read_namespaced_deployment(name=name, namespace=namespace)
    selector = dep.spec.selector.match_labels or {}
    label_selector = ",".join([f"{k}={v}" for k, v in selector.items()]) if selector else ""

    pods = []
    cpu_total = 0.0
    mem_total = 0.0

    if label_selector:
        for pod in v1.list_namespaced_pod(namespace, label_selector=label_selector).items:
            key = f"{namespace}/{pod.metadata.name}"
            m = metrics.get(key, {"cpu_m": 0.0, "memory_mib": 0.0})
            cpu_total += m["cpu_m"]
            mem_total += m["memory_mib"]
            pods.append(
                {
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "cpu_m": m["cpu_m"],
                    "memory_mib": m["memory_mib"],
                }
            )

    containers = []
    cpu_req_total = 0.0
    mem_req_total = 0.0

    for c in dep.spec.template.spec.containers:
        req = (c.resources.requests or {}) if c.resources else {}
        lim = (c.resources.limits or {}) if c.resources else {}

        cpu_req = _parse_cpu_to_m(req.get("cpu"))
        cpu_lim = _parse_cpu_to_m(lim.get("cpu"))
        mem_req = _parse_mem_to_mib(req.get("memory"))
        mem_lim = _parse_mem_to_mib(lim.get("memory"))

        cpu_req_total += cpu_req
        mem_req_total += mem_req

        containers.append(
            {
                "name": c.name,
                "cpu_request_m": cpu_req,
                "cpu_limit_m": cpu_lim,
                "memory_request_mib": mem_req,
                "memory_limit_mib": mem_lim,
            }
        )

    cpu_util = (cpu_total / cpu_req_total) if cpu_req_total > 0 else 0.0
    mem_util = (mem_total / mem_req_total) if mem_req_total > 0 else 0.0

    snap = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "namespace": namespace,
        "name": name,
        "replicas": dep.spec.replicas or 0,
        "available_replicas": dep.status.available_replicas or 0,
        "selector": selector,
        "containers": containers,
        "pods": pods,
        "avg_cpu_m": cpu_total,
        "avg_memory_mib": mem_total,
        "cpu_utilization": cpu_util,
        "memory_utilization": mem_util,
    }
    return json.dumps(snap, indent=2)


@mcp.tool()
def scale_deployment(namespace: str, name: str, replicas: int) -> str:
    apps, v1, custom = _load_k8s()
    body = {"spec": {"replicas": replicas}}
    resp = apps.patch_namespaced_deployment(name=name, namespace=namespace, body=body)
    return json.dumps(
        {"ok": True, "namespace": namespace, "name": name, "replicas": resp.spec.replicas},
        indent=2,
    )


@mcp.tool()
def patch_cpu_request(
    namespace: str,
    name: str,
    container: str,
    cpu_request_m: int,
    cpu_limit_m: int,
) -> str:
    apps, v1, custom = _load_k8s()
    patch = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": container,
                            "resources": {
                                "requests": {"cpu": f"{cpu_request_m}m"},
                                "limits": {"cpu": f"{cpu_limit_m}m"},
                            },
                        }
                    ]
                }
            }
        }
    }
    resp = apps.patch_namespaced_deployment(name=name, namespace=namespace, body=patch)
    return json.dumps(
        {"ok": True, "namespace": namespace, "name": name, "container": container},
        indent=2,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
