from typing import List, Dict, Any, Literal, Optional

from dataclasses import dataclass
from pydantic import BaseModel, Field


class ExecutionRecord(BaseModel):
    """Individual execution data captured at runtime."""
    args: Dict[str, Any]  # Explicit input params
    return_value: Any
    duration: float
    timestamp: float
    coverage: Optional[float] = 0.0  # Code coverage percentage


class CallGraphNode(BaseModel):
    """Attributes for a node representing a function or class."""
    # Static metadata
    fqn: str = Field(..., description="Fully qualified name, e.g., src.module.Class.method")
    # NOTE: "module" and "external" are used to support maximal graphs.
    type: Literal["function_definition", "class_definition", "module", "external"]
    file: str
    start_line: int
    end_line: int
    description: Optional[str] = ""  # Semantic description of the function

    # Dynamic metadata (Optional depending on analysis_type)
    execution_count: int = 0
    executions: List[ExecutionRecord] = []
    total_duration: Optional[float] = 0.0
    avg_duration: Optional[float] = 0.0

    # Hybrid metadata
    covered: bool = False
    suspiciousness: Optional[float] = -1.0

    analysis_type: Literal["static", "dynamic", "hybrid"] = "static"


class CallGraphEdge(BaseModel):
    """Attributes for an edge representing a call relationship."""
    source: str = Field(..., description="FQN of the caller")
    target: str = Field(..., description="FQN of the callee")

    # Dynamic metrics
    call_count: Optional[int] = 0
    avg_call_duration: Optional[float] = 0.0

    # Hybrid metadata
    analysis_type: Literal["static", "dynamic", "hybrid"]


class CallGraph(BaseModel):
    # Source/Target flags
    static: bool = False
    dynamic: bool = False

    nodes: List[CallGraphNode] = []
    edges: List[CallGraphEdge] = []
    
    def get_suspicious_nodes(
        self,
        min_suspiciousness: float = 0.0,
        limit: Optional[int] = None
    ) -> List[CallGraphNode]:
        """
        Returns nodes sorted by suspiciousness score in descending order.
        
        Args:
            min_suspiciousness: Minimum suspiciousness threshold (default: 0.0)
            limit: Maximum number of nodes to return (default: None, returns all)
        
        Returns:
            List of CallGraphNode objects sorted by suspiciousness (highest first)
        """
        # Filter nodes by minimum suspiciousness
        suspicious = [
            node for node in self.nodes
            if node.suspiciousness is not None and node.suspiciousness >= min_suspiciousness
        ]
        
        # Sort by suspiciousness in descending order
        suspicious.sort(key=lambda n: n.suspiciousness or 0.0, reverse=True)
        
        # Apply limit if specified
        if limit is not None:
            suspicious = suspicious[:limit]
        
        return suspicious


@dataclass(frozen=True)
class _ActiveCall:
    fqn: str
    caller_fqn: Optional[str]
    start_ts: float
    start_rel_ts: float
    args: Dict[str, Any]


@dataclass
class RepoDefinition:
    """Minimal definition of a repository to be traced."""
    repo_path: str
    trace_script: str
    install_command: Optional[str] = None


@dataclass
class DockerTracerConfig:
    """Configuration for the Dockerized tracer pipeline."""
    image_name: str = "python:3.11-slim"
    keep_alive: bool = False
    output_file: str = "call_graph.json"