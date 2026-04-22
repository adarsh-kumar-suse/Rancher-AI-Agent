from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ContainerSnapshot(BaseModel):
    name: str
    cpu_request_m: float = 0.0
    cpu_limit_m: float = 0.0
    memory_request_mib: float = 0.0
    memory_limit_mib: float = 0.0


class PodUsage(BaseModel):
    name: str
    cpu_m: float = 0.0
    memory_mib: float = 0.0


class DeploymentSnapshot(BaseModel):
    namespace: str
    name: str
    replicas: int = 0
    available_replicas: int = 0
    selector: Dict[str, str] = Field(default_factory=dict)
    containers: List[ContainerSnapshot] = Field(default_factory=list)
    pods: List[PodUsage] = Field(default_factory=list)
    avg_cpu_m: float = 0.0
    avg_memory_mib: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0


class ClusterSnapshot(BaseModel):
    captured_at: str
    deployments: List[DeploymentSnapshot] = Field(default_factory=list)


class Action(BaseModel):
    type: Literal["scale", "patch_resources"]
    namespace: str
    name: str
    reason: str = ""
    replicas: Optional[int] = None
    container: Optional[str] = None
    cpu_request_m: Optional[int] = None
    cpu_limit_m: Optional[int] = None
    memory_request_mib: Optional[int] = None
    memory_limit_mib: Optional[int] = None


class Plan(BaseModel):
    summary: str = ""
    actions: List[Action] = Field(default_factory=list)
