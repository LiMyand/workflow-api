from typing import List, Dict, Set
import asyncio
from .workflow import WorkflowNode
from .dto import WorkflowData, WorkflowEvent, WorkflowEventType
from .logger import logger
from ..agents.base import BaseAgent, StartAgent, EndAgent
from ..agents.delay_agent import DelayAgent
from ..agents.aggregator_agent import AggregatorAgent
from ..agents.loop_agent import LoopAgent
from ..agents.parallel_agent import ParallelAgent
from ..agents.llm_agents import LLMAgentFactory
from .stream_state import StreamState, StreamStateMachine, ParallelStreamStateMachine
from ..agents.splitter_agent import SplitterAgent
from ..agents.branch_agent import BranchAgent
import traceback


class WorkflowManager:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.node_results = {}
        logger.info("初始化新的工作流管理器")

    def add_node(
        self,
        node_id: str,
        node_type: str,
        config: Dict = None,
        prompt_template: str = None,
        model_name: str = None,
    ):
        """
        添加工作流节点
        :param node_id: 节点ID
        :param node_type: 节点类型
        :param config: 节点配置
        :param prompt_template: 提示词模板
        :param model_name: 模型名称
        """
        config = config or {}

        # 添加分支节点的处理
        if node_type == "branchNode":
            agent = BranchAgent(evaluation_mode=config.get("evaluation_mode", "direct"))
        elif node_type == "startNode":
            agent = StartAgent()
        elif node_type == "endNode":
            agent = EndAgent()
        elif node_type == "delayNode":
            agent = DelayAgent()
        elif node_type == "aggregatorNode":
            agent = AggregatorAgent(config.get("aggregation_type", "concat"))

        elif node_type == "loopNode":
            agent = LoopAgent()
            # 确保 evaluation_prompt 从配置中正确传递
            if "evaluation_prompt" not in config:
                # 如果 evaluation_prompt 在 data 层级，移动到 config 层级
                if (
                    isinstance(prompt_template, dict)
                    and "evaluation_prompt" in prompt_template
                ):
                    config["evaluation_prompt"] = prompt_template["evaluation_prompt"]
                elif (
                    isinstance(config, dict)
                    and "data" in config
                    and "evaluation_prompt" in config["data"]
                ):
                    config["evaluation_prompt"] = config["data"]["evaluation_prompt"]
        elif node_type == "parallelNode":
            agent = ParallelAgent(config.get("parallel_config"))
        elif node_type == "splitterNode":
            agent = SplitterAgent(config.get("split_config"))
        elif node_type == "splitterNode":
            agent = SplitterAgent(config.get("split_config"))
        else:
            if not model_name:
                raise ValueError(f"节点 {node_id} ({node_type}) 需要指定模型名称")
            agent = LLMAgentFactory.create_agent(model_name)

        # 创建工作流节点
        node = WorkflowNode(
            id=node_id,
            agent=agent,
            prompt_template=prompt_template,
            config={"type": node_type, **config},
        )

        self.nodes[node_id] = node
        logger.info(f"添加节点: {node_id} ({node_type})")

    def add_edge(
        self,
        from_node_id: str,
        to_node_id: str,
        params_mapping: dict = None,
        condition: str = None,
    ):
        """
        添加带条件的边
        """
        # 验证节点是否存在
        if from_node_id not in self.nodes:
            raise ValueError(f"源节点 {from_node_id} 不存在")
        if to_node_id not in self.nodes:
            raise ValueError(f"目标节点 {to_node_id} 不存在")

        logger.info(f"添加边: {from_node_id} -> {to_node_id}")
        edge = {
            "from": from_node_id,
            "to": to_node_id,
            "params_mapping": params_mapping or {},
            "condition": condition,
        }
        self.edges.append(edge)

        # 如果是结束节点，不需要添加 next_nodes
        if to_node_id == "end_node":
            return

        # 为源节点添加 next_nodes 属性（如果还没有的话）
        if not hasattr(self.nodes[from_node_id], "next_nodes"):
            self.nodes[from_node_id].next_nodes = []
        self.nodes[from_node_id].next_nodes.append(
            {"node_id": to_node_id, "condition": condition}
        )

    async def _execute_node(self, node_id: str, input_data: List[WorkflowData]):
        """执行单个节点的具体逻辑"""
        try:
            node = self.nodes[node_id]
            logger.info(f"执行节点 {node_id} (类型: {node.type})")

            return await node.agent.execute(
                inputs=input_data, prompt_template=node.prompt_template, **node.config
            )
        except Exception as e:
            logger.error(f"节点 {node_id} 执行失败: {str(e)}")
            raise

    def _get_execution_levels(self, start_node_id: str) -> List[List[str]]:
        """获取节点的执行层级，支持并行执行标记"""
        levels = []
        visited = set()
        current_level = {start_node_id}

        while current_level:
            levels.append(list(current_level))
            visited.update(current_level)

            next_level_candidates = set()
            for node_id in current_level:
                node = self.nodes[node_id]

                # 检查是否是分割节点
                if node.type == "splitterNode":
                    # 获取所有目标节点并设置为同一层级
                    routing = node.config.get("routing", {})
                    next_level_candidates.update(routing.values())
                else:
                    # 常规节点处理
                    next_nodes = {
                        edge["to"] for edge in self.edges if edge["from"] == node_id
                    }
                    next_level_candidates.update(next_nodes)

            # 过滤已访问的节点
            current_level = {
                node for node in next_level_candidates if node not in visited
            }

        return levels

    async def execute_workflow_stream(
        self, start_node_id: str, initial_context: Dict = None
    ):
        """以流式方式执行工作流"""
        state_machine = ParallelStreamStateMachine()

        try:
            context = initial_context or {}
            # 如果存在 input_text，将其添加到上下文中
            if "input_text" in context:
                context["start_node"] = WorkflowData(
                    content=context["input_text"], metadata={"type": "text_input"}
                )

            visited = set()
            async for event in self._execute_node_stream(
                start_node_id, context, visited, state_machine
            ):
                yield event

        except Exception as e:
            logger.error(f"工作流执行错误: {str(e)}", exc_info=True)
            yield WorkflowEvent(
                event_type=WorkflowEventType.NODE_ERROR,
                node_id="workflow",
                data=str(e),
                metadata={
                    "error_type": "WorkflowError",
                    "details": {"traceback": traceback.format_exc()},
                },
            )

    async def _execute_node_stream(
        self,
        node_id: str,
        context: Dict,
        visited: set,
        state_machine: ParallelStreamStateMachine,
        level_index: int = None,
        total_nodes: int = None,
    ):
        """执行单个节点并生成事件流"""
        node = self.nodes[node_id]

        try:
            dependencies = {
                edge["from"] for edge in self.edges if edge["to"] == node_id
            }
            missing_deps = dependencies - visited
            if missing_deps:
                logger.info(f"节点 {node_id} 等待依赖节点完成: {missing_deps}")
                return

            if node_id in visited:
                logger.info(f"节点 {node_id} 已经执行过，跳过")
                return

            async for event in self._execute_node_stream_core(
                node_id, node, context, visited, state_machine
            ):
                yield event

        except Exception as e:
            logger.error(f"节点 {node_id} 执行错误: {str(e)}", exc_info=True)
            state_machine.update_node_state(
                node_id, StreamState.NODE_ERROR, error=str(e)
            )
            yield WorkflowEvent(
                event_type=WorkflowEventType.NODE_ERROR,
                node_id=node_id,
                data=str(e),
                metadata={
                    "error_type": type(e).__name__,
                    "state": StreamState.NODE_ERROR.value,
                },
            )
            raise

    async def _execute_node_stream_core(
        self,
        node_id: str,
        node,
        context: Dict,
        visited: set,
        state_machine: ParallelStreamStateMachine,
        level_index: int = None,
        total_nodes: int = None,
    ):
        """节点流式执行的核心逻辑"""
        visited.add(node_id)

        # 构建输入数据
        input_data = []
        if node_id == "start_node":
            if "start_node" in context:
                input_data = [context["start_node"]]
        else:
            # 对于其他节点，从其依赖节点获取数据
            for edge in self.edges:
                if edge["to"] == node_id:
                    from_node_id = edge["from"]
                    if from_node_id in context:
                        result = context[from_node_id]

                        # 检查是否是分割节点的输出数据
                        if (
                            isinstance(result, WorkflowData)
                            and "segment_contents" in result.metadata
                            and node_id in result.metadata["segment_contents"]
                        ):
                            # 只获取当前节点对应的段落内容
                            segment_data = result.metadata["segment_contents"][node_id]
                            input_data.append(
                                WorkflowData(
                                    content=segment_data["content"],
                                    metadata={
                                        "source_key": segment_data["source_key"],
                                        "source_node": from_node_id,
                                    },
                                )
                            )
                        else:
                            input_data.append(result)

        logger.info(f"节点 {node_id} 的输入数据: {input_data}")

        try:
            yield WorkflowEvent(
                event_type=WorkflowEventType.NODE_START,
                node_id=node_id,
                metadata={"state": "started"},
            )
            yield WorkflowEvent(
                event_type=WorkflowEventType.NODE_RUNNING,
                node_id=node_id,
                metadata={"state": "started"},
            )
            # 执行节点
            result = await self._execute_node(node_id, input_data)

            # 更新上下文
            context[node_id] = result

            # 修改分支节点的处理逻辑
            if node.config["type"] == "branchNode":
                # 获取分支配置
                true_branch = node.config.get("true_branch")
                false_branch = node.config.get("false_branch")
                condition = node.config.get("condition", {})

                # 根据分支节点的结果决定下一个节点
                try:
                    # 从结果的metadata中获取实际的条件判断结果
                    is_true_branch = result.metadata.get("condition_result", False)
                    selected_branch = true_branch if is_true_branch else false_branch

                    if not selected_branch:
                        raise ValueError(f"分支节点 {node_id} 未找到有效的目标分支")

                    next_nodes = [selected_branch]
                    logger.info(
                        f"分支节点 {node_id} 选择了分支: {selected_branch} (结果: {is_true_branch})"
                    )
                except Exception as e:
                    logger.error(f"分支节点结果处理失败: {str(e)}")
                    raise ValueError(f"分支节点 {node_id} 结果处理失败: {str(e)}")
            else:
                next_nodes = self.get_next_nodes(node_id)
            yield WorkflowEvent(
                event_type=WorkflowEventType.NODE_OUTPUT,
                node_id=node_id,
                data=result.content,
                metadata={"state": "output_generated"},
            )

            yield WorkflowEvent(
                event_type=WorkflowEventType.NODE_END,
                node_id=node_id,
                data=None,
                metadata={"state": "completed"},
            )

            next_level_total = len(next_nodes)
            next_level_index = (
                (level_index or 0) + 1 if level_index is not None else None
            )

            # 如果是分割节点，并行执行后续节点
            if node.config["type"] == "splitterNode":
                # 创建队列来接收事件
                queue = asyncio.Queue()
                tasks = []
                parallel_visited = set()  # 用于追踪并行执行完成的节点
                logger.info(f"分割节点 {node_id} 的配置: {node.config}")
                logger.info(f"================================================")
                logger.info(f"================================================")
                logger.info(f"================================================")
                logger.info(f"传入下一节点的数据: {context}")
                logger.info(f"================================================")
                logger.info(f"================================================")
                logger.info(f"================================================")

                logger.info(f"================================================")
                logger.info(f"================================================")
                logger.info(f"================================================")

                # 创建事件收集任务
                async def collect_events(node_id, context, visited):
                    try:
                        async for event in self._execute_node_stream(
                            node_id,
                            context,
                            visited.copy(),
                            state_machine,
                            level_index=next_level_index,
                            total_nodes=next_level_total,
                        ):
                            await queue.put((node_id, event))
                        parallel_visited.add(node_id)  # 标记节点执行完成
                    except Exception as e:
                        logger.error(f"并行节点 {node_id} 执行错误: {str(e)}")
                        await queue.put(
                            (
                                node_id,
                                WorkflowEvent(
                                    event_type=WorkflowEventType.NODE_ERROR,
                                    node_id=node_id,
                                    data=str(e),
                                    metadata={"error_type": type(e).__name__},
                                ),
                            )
                        )

                # 启动所有并行任务
                for next_node_id in next_nodes:
                    task = asyncio.create_task(
                        collect_events(next_node_id, context, visited)
                    )
                    tasks.append(task)

                # 等待所有任务完成，同时实时yield事件
                done_count = 0
                total_tasks = len(tasks)

                while done_count < total_tasks:
                    try:
                        node_id, event = await queue.get()
                        yield event
                        if event.event_type in [
                            WorkflowEventType.NODE_END,
                            WorkflowEventType.NODE_ERROR,
                        ]:
                            done_count += 1
                    except Exception as e:
                        logger.error(f"处理并行事件时出错: {str(e)}")
                        break

                # 等待所有任务完成
                await asyncio.gather(*tasks, return_exceptions=True)

                # 更新visited集合，加入所有已完成的并行节点
                visited.update(parallel_visited)

                # 处理聚合节点
                aggregator_nodes = []
                for edge in self.edges:
                    if (
                        edge["from"] in parallel_visited
                        and self.nodes[edge["to"]].config["type"] == "aggregatorNode"
                    ):
                        aggregator_nodes.append(edge["to"])

                # 执行聚合节点
                for aggregator_node in set(aggregator_nodes):  # 使用set去重
                    deps = {
                        edge["from"]
                        for edge in self.edges
                        if edge["to"] == aggregator_node
                    }
                    if deps.issubset(visited):
                        logger.info(f"执行聚合节点: {aggregator_node}")
                        async for event in self._execute_node_stream(
                            aggregator_node,
                            context,
                            visited,
                            state_machine,
                            level_index=next_level_index,
                            total_nodes=next_level_total,
                        ):
                            yield event

            else:
                # 修改后续节点的执行逻辑
                for next_node_id in next_nodes:
                    if next_node_id in visited:
                        continue

                    if self.nodes[next_node_id].config["type"] == "aggregatorNode":
                        deps = {
                            edge["from"]
                            for edge in self.edges
                            if edge["to"] == next_node_id
                        }
                        if not deps.issubset(visited):
                            continue

                    async for event in self._execute_node_stream(
                        next_node_id,
                        context,
                        visited,
                        state_machine,
                        level_index=next_level_index,
                        total_nodes=next_level_total,
                    ):
                        yield event

        except Exception as e:
            logger.error(f"节点 {node_id} 执行错误: {str(e)}", exc_info=True)
            state_machine.update_node_state(
                node_id, StreamState.NODE_ERROR, error=str(e)
            )
            yield WorkflowEvent(
                event_type=WorkflowEventType.NODE_ERROR,
                node_id=node_id,
                data=str(e),
                metadata={
                    "error_type": type(e).__name__,
                    "state": StreamState.NODE_ERROR.value,
                },
            )
            raise

    async def _execute_parallel_node(self, *args, **kwargs):
        """执行并行节点并收集所有事件"""
        events = []
        async for event in self._execute_node_stream(*args, **kwargs):
            events.append(event)
        return events

    def get_next_nodes(self, node_id: str) -> List[str]:
        """获取指定节点的所有后续节点"""
        next_nodes = []
        node = self.nodes[node_id]

        # 如果是分支节点，返回所有可能的分支
        if node.config["type"] == "branchNode":
            true_branch = node.config.get("true_branch")
            false_branch = node.config.get("false_branch")
            if true_branch:
                next_nodes.append(true_branch)
            if false_branch:
                next_nodes.append(false_branch)
        else:
            # 常规节点处理
            for edge in self.edges:
                if edge["from"] == node_id:
                    next_nodes.append(edge["to"])

        logger.info(f"节点 {node_id} 的后续节点: {next_nodes}")
        return next_nodes

    async def _gather_events(self, generator):
        """收集异步生成器的所有事件"""
        events = []
        async for event in generator:
            events.append(event)
        return events
