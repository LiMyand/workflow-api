from enum import Enum
from typing import Dict, Set
from dataclasses import dataclass


class StreamState(Enum):
    INIT = "init"  # 初始状态
    NODE_PREPARING = "preparing"  # 节点准备阶段
    NODE_STARTED = "started"  # 节点开始执行
    NODE_RUNNING = "running"  # 节点运行中
    NODE_PROCESSING = "processing"  # 节点处理输出
    NODE_COMPLETED = "completed"  # 节点完成
    NODE_ERROR = "error"  # 节点错误
    WORKFLOW_COMPLETED = "workflow_completed"  # 工作流完成
    WORKFLOW_ERROR = "workflow_error"  # 工作流错误


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


@dataclass
class NodeStateInfo:
    state: StreamState
    error: str = None


class ParallelStreamStateMachine(StreamStateMachine):
    def __init__(self):
        super().__init__()
        self.node_states = {}  # 存储每个节点的状态
        self.parallel_groups = {}  # 存储并行节点组
        self.errors = {}  # 存储节点错误信息

    def _get_parallel_nodes(self, node_id: str) -> set:
        """
        获取与指定节点在同一并行组的其他节点
        :param node_id: 节点ID
        :return: 同组的并行节点集合
        """
        for group, nodes in self.parallel_groups.items():
            if node_id in nodes:
                return set(nodes)
        return set()

    def add_parallel_group(self, nodes: list):
        """
        添加一组并行执行的节点
        :param nodes: 并行节点ID列表
        """
        group_id = len(self.parallel_groups)
        self.parallel_groups[group_id] = set(nodes)

    def update_node_state(
        self, node_id: str, new_state: StreamState, error: str = None
    ):
        """
        更新指定节点的状态
        :param node_id: 节点ID
        :param new_state: 新状态
        :param error: 错误信息（如果有）
        """
        # 更新节点状态
        self.node_states[node_id] = NodeStateInfo(state=new_state, error=error)

        # 如果是错误状态，记录错误信息
        if new_state == StreamState.NODE_ERROR:
            self.errors[node_id] = error

        # 获取同组的并行节点
        parallel_nodes = self._get_parallel_nodes(node_id)

        # 如果没有并行节点，直接更新当前状态
        if not parallel_nodes:
            self.current_state = new_state
            return

        # 检查所有并行节点的状态
        all_completed = True
        any_error = False

        for node in parallel_nodes:
            if node not in self.node_states:
                all_completed = False
                break
            node_state = self.node_states[node].state
            if node_state == StreamState.NODE_ERROR:
                any_error = True
                break
            if node_state != StreamState.NODE_COMPLETED:
                all_completed = False

        # 根据并行节点状态更新当前状态
        if any_error:
            self.current_state = StreamState.NODE_ERROR
        elif all_completed:
            self.current_state = StreamState.NODE_COMPLETED
        else:
            self.current_state = new_state
