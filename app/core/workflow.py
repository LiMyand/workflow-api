from typing import Dict, List, Optional
from .dto import WorkflowData
from ..agents.base import BaseAgent
from enum import Enum
from .logger import logger  # 导入 logger
import json


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowNode:
    def __init__(
        self,
        id: str,
        agent: BaseAgent,
        prompt_template: str = None,
        config: Dict = None,
    ):
        self.id = id
        self.agent = agent
        self.prompt_template = prompt_template
        self.config = config or {}
        self.status = NodeStatus.PENDING
        self.error = None
        self.start_time = None
        self.end_time = None
        self.input_data: Optional[List[WorkflowData]] = None
        self.output_data: Optional[WorkflowData] = None
        self.next_nodes: List[str] = []
        self.type = config.get("type", "llmNode") if config else "llmNode"

        # 记录节点信息到日志
        node_info = {
            "id": self.id,
            "type": self.type,
            "agent_type": type(self.agent).__name__,
            "config": self.config,
            "next_nodes": self.next_nodes,
        }

        logger.info(
            f"DAG节点信息: {json.dumps(node_info, ensure_ascii=False, indent=2)}"
        )


class WorkflowEdge:
    def __init__(
        self,
        source_id: str,
        target_id: str,
        data_mapping: Dict[str, str] = None,
        condition: str = None,
        transform_function: str = None,
        aggregation_mode: str = None,
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.data_mapping = data_mapping or {}
        self.condition = condition
        self.transform_function = transform_function
        self.aggregation_mode = aggregation_mode

        # 记录边信息到日志
        edge_info = {
            "source": source_id,
            "target": target_id,
            "mapping": data_mapping,
            "condition": condition,
            "transform": transform_function,
            "aggregation": aggregation_mode,
        }

        logger.info(f"DAG边信息: {json.dumps(edge_info, ensure_ascii=False, indent=2)}")


class StreamState(Enum):
    INIT = "init"
    NODE_PREPARING = "preparing"
    NODE_STARTED = "started"
    NODE_RUNNING = "running"
    NODE_PROCESSING = "processing"
    NODE_COMPLETED = "completed"
    NODE_ERROR = "error"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_ERROR = "workflow_error"


class StreamStateMachine:
    def __init__(self):
        self.current_state = StreamState.INIT
        self.transitions = {
            StreamState.INIT: [StreamState.NODE_PREPARING, StreamState.WORKFLOW_ERROR],
            StreamState.NODE_PREPARING: [
                StreamState.NODE_STARTED,
                StreamState.NODE_ERROR,
            ],
            StreamState.NODE_STARTED: [
                StreamState.NODE_RUNNING,
                StreamState.NODE_ERROR,
            ],
            StreamState.NODE_RUNNING: [
                StreamState.NODE_PROCESSING,
                StreamState.NODE_ERROR,
            ],
            StreamState.NODE_PROCESSING: [
                StreamState.NODE_COMPLETED,
                StreamState.NODE_ERROR,
            ],
            StreamState.NODE_COMPLETED: [
                StreamState.NODE_PREPARING,
                StreamState.WORKFLOW_COMPLETED,
                StreamState.NODE_ERROR,
            ],
            StreamState.NODE_ERROR: [
                StreamState.NODE_PREPARING,
                StreamState.WORKFLOW_ERROR,
            ],
            StreamState.WORKFLOW_ERROR: [],  # 终止状态
            StreamState.WORKFLOW_COMPLETED: [],  # 终止状态
        }


class WorkflowEventType(Enum):
    NODE_START = "node_start"
    NODE_RUNNING = "node_running"
    NODE_OUTPUT = "node_output"
    NODE_END = "node_end"
    NODE_ERROR = "node_error"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    WORKFLOW_ERROR = "workflow_error"
