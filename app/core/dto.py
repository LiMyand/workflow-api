from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional


class WorkflowEventType(Enum):
    NODE_START = "node_start"
    NODE_RUNNING = "node_running"
    NODE_OUTPUT = "node_output"
    NODE_END = "node_end"
    NODE_ERROR = "node_error"


@dataclass
class WorkflowData:
    content: Any
    metadata: Dict[str, Any] = None
    prompt: Optional[str] = None
    additional_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.additional_params is None:
            self.additional_params = {}


@dataclass
class WorkflowEvent:
    event_type: WorkflowEventType
    node_id: str
    data: Optional[Any] = None
    metadata: Dict[str, Any] = None
    timestamp: float = None
