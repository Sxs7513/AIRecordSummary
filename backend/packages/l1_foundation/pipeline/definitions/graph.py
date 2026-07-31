from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from l1_foundation.pipeline.contracts import RetryPolicy


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    """Name an input artifact consumed by a node."""

    name: str
    artifact_type: str
    from_node: str | None = None


@dataclass(frozen=True, slots=True)
class PipelineNode:
    name: str
    stage_name: str
    stage_version: str
    retry_policy: RetryPolicy
    depends_on: tuple[str, ...] = ()
    required: bool = True
    input_payload: dict[str, Any] | None = None
    input_artifacts: tuple[ArtifactBinding, ...] = ()
    output_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PipelineDefinition:
    name: str
    version: str
    nodes: tuple[PipelineNode, ...]

    def __post_init__(self) -> None:
        node_names = {node.name for node in self.nodes}
        if not self.nodes:
            raise ValueError("Pipeline definition requires at least one node")
        if len(node_names) != len(self.nodes):
            raise ValueError("Pipeline node names must be unique")
        for node in self.nodes:
            missing_dependencies = set(node.depends_on) - node_names
            if missing_dependencies:
                raise ValueError(f"Node {node.name} references missing dependencies: {sorted(missing_dependencies)}")
            for binding in node.input_artifacts:
                if binding.from_node is not None and binding.from_node not in node.depends_on:
                    raise ValueError(f"Node {node.name} must depend on artifact producer {binding.from_node}")
        self.topologically_sorted_nodes()

    def topologically_sorted_nodes(self) -> tuple[PipelineNode, ...]:
        nodes_by_name = {node.name: node for node in self.nodes}
        remaining_dependencies = {node.name: set(node.depends_on) for node in self.nodes}
        ordered: list[PipelineNode] = []
        while remaining_dependencies:
            ready_names = sorted(name for name, dependencies in remaining_dependencies.items() if not dependencies)
            if not ready_names:
                raise ValueError("Pipeline definition contains a dependency cycle")
            for name in ready_names:
                ordered.append(nodes_by_name[name])
                del remaining_dependencies[name]
            completed = set(ready_names)
            for dependencies in remaining_dependencies.values():
                dependencies.difference_update(completed)
        return tuple(ordered)
