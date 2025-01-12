from typing import List, Dict, AsyncGenerator
import asyncio
from enum import Enum
from dataclasses import dataclass


class StreamChunkType(Enum):
    CONTENT = "content"
    STATUS = "status"
    ERROR = "error"


@dataclass
class StreamChunk:
    node_id: str
    content: str
    type: StreamChunkType
    status: StreamState = None
    error: str = None


class StreamWorkflowManager:
    async def _execute_parallel_nodes(
        self, nodes: List[str], context: dict
    ) -> AsyncGenerator:
        """并行执行多个节点的流式响应"""
        # 创建所有节点的任务
        tasks = []
        for node_id in nodes:
            input_data = self._prepare_node_input(node_id, context)
            task = self._execute_node_stream(node_id, input_data)
            tasks.append((node_id, asyncio.create_task(task)))

        # 使用 asyncio.as_completed 处理流式响应
        for completed_task in asyncio.as_completed([task for _, task in tasks]):
            try:
                async for chunk in await completed_task:
                    yield chunk
            except Exception as e:
                # 处理错误
                logger.error(f"Stream execution error: {str(e)}")
                yield StreamChunk(
                    node_id=node_id,
                    content="",
                    type=StreamChunkType.ERROR,
                    error=str(e),
                )

    async def _execute_node_stream(
        self, node_id: str, input_data: dict
    ) -> AsyncGenerator:
        """执行单个节点的流式响应"""
        try:
            # 更新节点状态为开始执行
            self.state_manager.update_node_state(node_id, StreamState.NODE_RUNNING)
            yield StreamChunk(
                node_id=node_id,
                content="",
                type=StreamChunkType.STATUS,
                status=StreamState.NODE_RUNNING,
            )

            # 获取节点执行器
            node = self.nodes[node_id]
            executor = self._get_node_executor(node)

            # 执行节点并获取流式响应
            async for chunk in executor.execute_stream(input_data):
                yield StreamChunk(
                    node_id=node_id, content=chunk.content, type=StreamChunkType.CONTENT
                )

            # 更新节点状态为完成
            self.state_manager.update_node_state(node_id, StreamState.NODE_COMPLETED)
            yield StreamChunk(
                node_id=node_id,
                content="",
                type=StreamChunkType.STATUS,
                status=StreamState.NODE_COMPLETED,
            )

        except Exception as e:
            # 处理错误状态
            self.state_manager.update_node_state(
                node_id, StreamState.NODE_ERROR, error=str(e)
            )
            yield StreamChunk(
                node_id=node_id, content="", type=StreamChunkType.ERROR, error=str(e)
            )

    async def execute_workflow_stream(
        self, start_node_id: str, initial_context: dict = None
    ) -> AsyncGenerator:
        """执行整个工作流的流式响应"""
        context = initial_context or {}
        execution_levels = self._get_execution_levels(start_node_id)

        for level in execution_levels:
            # 检查是否有并行节点
            parallel_nodes = [
                node_id
                for node_id in level
                if self.nodes[node_id].type == "splitterNode"
                or self.nodes[node_id].metadata.get("parallel_execution")
            ]

            if parallel_nodes:
                # 并行执行节点
                async for chunk in self._execute_parallel_nodes(
                    parallel_nodes, context
                ):
                    yield chunk
                    if chunk.type == StreamChunkType.CONTENT:
                        # 更新上下文
                        context.setdefault(chunk.node_id, "")
                        context[chunk.node_id] += chunk.content

            # 执行其他顺序节点
            sequential_nodes = [n for n in level if n not in parallel_nodes]
            for node_id in sequential_nodes:
                async for chunk in self._execute_node_stream(node_id, context):
                    yield chunk
                    if chunk.type == StreamChunkType.CONTENT:
                        # 更新上下文
                        context.setdefault(node_id, "")
                        context[node_id] += chunk.content
