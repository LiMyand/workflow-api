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
from contextlib import asynccontextmanager
import traceback


class WorkflowManager:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.sync_events = {}
        self.node_results = {}  # 存储节点执行结果
        self.node_dependencies = {}  # 存储节点依赖关系
        logger.info("初始化新的工作流管理器")

    @asynccontextmanager
    async def level_synchronizer(self, level_index: int, total_nodes: int):
        """
        层级同步器，确保同层节点同步执行
        """
        # 为每一层创建开始和结束事件
        if level_index not in self.sync_events:
            self.sync_events[level_index] = {
                "start": asyncio.Event(),
                "end": asyncio.Event(),
                "counter": 0,
                "total": total_nodes,
            }

        try:
            # 等待所有节点就绪
            self.sync_events[level_index]["counter"] += 1
            if (
                self.sync_events[level_index]["counter"]
                == self.sync_events[level_index]["total"]
            ):
                self.sync_events[level_index]["start"].set()

            # 等待开始信号
            await self.sync_events[level_index]["start"].wait()

            yield

            # 节点执行完成后
            self.sync_events[level_index]["counter"] -= 1
            if self.sync_events[level_index]["counter"] == 0:
                self.sync_events[level_index]["end"].set()
        finally:
            # 清理事件
            if self.sync_events[level_index]["counter"] == 0:
                del self.sync_events[level_index]

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

        # 创建对应的 agent
        if node_type == "startNode":
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

    def _all_dependencies_visited(self, node_id: str, visited: set) -> bool:
        """检查节点的所有前置依赖是否都已访问"""
        dependencies = {edge["from"] for edge in self.edges if edge["to"] == node_id}
        return all(dep in visited for dep in dependencies)

    def _build_dependencies(self):
        """构建节点依赖关系"""
        self.node_dependencies = {}
        for edge in self.edges:
            if edge["to"] not in self.node_dependencies:
                self.node_dependencies[edge["to"]] = set()
            self.node_dependencies[edge["to"]].add(edge["from"])

    async def _execute_level_node(
        self, node_id: str, context: dict, level_index: int, total_nodes: int
    ):
        """执行层级节点，包含同步和依赖检查"""
        async with self.level_synchronizer(level_index, total_nodes):
            logger.info(f"节点 {node_id} 准备就绪，等待同层节点")

            # 检查依赖是否都已完成
            if node_id in self.node_dependencies:
                deps = self.node_dependencies[node_id]
                while not all(dep in self.node_results for dep in deps):
                    logger.info(f"节点 {node_id} 等待依赖完成: {deps}")
                    await asyncio.sleep(0.1)  # 短暂等待后重试

            # 准备输入数据
            input_data = self._prepare_node_input(node_id, context)

            # 执行节点
            logger.info(f"开始执行节点 {node_id}")
            result = await self._execute_node(node_id, input_data)
            logger.info(f"节点 {node_id} 执行完成")

            # 存储结果
            self.node_results[node_id] = result
            return result

    async def execute_workflow(
        self, start_node_id: str, initial_context: dict = None
    ) -> dict:
        context = initial_context or {}
        execution_levels = self._get_execution_levels(start_node_id)

        for level_index, level_nodes in enumerate(execution_levels):
            # 创建该层所有节点的任务
            tasks = []
            for node_id in level_nodes:
                task = self._execute_level_node(
                    node_id, context, level_index, len(level_nodes)
                )
                tasks.append((node_id, task))

            # 并行执行该层的所有任务
            results = await asyncio.gather(*[task for _, task in tasks])

            # 更新上下文
            for (node_id, _), result in zip(tasks, results):
                context[node_id] = result

        return context

    def _get_parent_nodes(self, node_id: str) -> List[str]:
        """获取节点的所有父节点"""
        return [edge["from"] for edge in self.edges if edge["to"] == node_id]

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

    def _topological_sort(self, start_node_id: str) -> list:
        """
        拓扑排序获取执行顺序
        """
        visited = set()
        order = []

        def dfs(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            # 先将当前节点加入顺序列表
            order.append(node_id)
            # 然后再访问后续节点
            if hasattr(self.nodes[node_id], "next_nodes"):
                for next_node in self.nodes[node_id].next_nodes:
                    # 从字典中获取节点ID
                    next_node_id = (
                        next_node["node_id"]
                        if isinstance(next_node, dict)
                        else next_node
                    )
                    dfs(next_node_id)

        dfs(start_node_id)
        return order

    def _prepare_node_input(self, node_id: str, context: dict) -> List[WorkflowData]:
        """准备节点的输入数据，包含依赖检查"""
        # 特殊处理 start_node
        if node_id == "start_node":
            if "start_node" in context:
                return [context["start_node"]]
            return [WorkflowData(content="", metadata={})]

        # 获取所有输入边
        incoming_edges = [edge for edge in self.edges if edge["to"] == node_id]
        input_data = []

        # 收集所有输入数据
        for edge in incoming_edges:
            from_node_id = edge["from"]
            if from_node_id not in context:
                logger.error(f"节点 {node_id} 的依赖节点 {from_node_id} 数据未找到")
                raise ValueError(f"依赖节点 {from_node_id} 的数据未找到")

            source_data = context[from_node_id]
            if not isinstance(source_data, WorkflowData):
                logger.warning(
                    f"节点 {from_node_id} 的输出不是 WorkflowData 类型，进行转换"
                )
                source_data = WorkflowData(content=str(source_data))

            # 应用数据映射
            if "params_mapping" in edge:
                source_data = self._map_data(source_data, edge["params_mapping"])

            input_data.append(source_data)
            logger.info(
                f"节点 {node_id} 收到来自 {from_node_id} 的输入数据: {source_data.content[:100]}..."
            )

        if not input_data:
            logger.error(f"节点 {node_id} 没有任何输入数据")
            raise ValueError(f"节点 {node_id} 没有输入数据")

        return input_data

    def _map_data(self, source_data: WorkflowData, mapping: dict) -> WorkflowData:
        """根据参数映射转换数据"""
        if not mapping:
            return source_data

        mapped_data = {}
        for target_key, source_key in mapping.items():
            value = None

            # 处理字符串类型的 source_key
            if isinstance(source_key, str):
                if "." in source_key:
                    # 处理点号分隔的路径
                    parts = source_key.split(".")
                    current = source_data
                    for part in parts:
                        if isinstance(current, dict):
                            current = current.get(part)
                        elif hasattr(current, part):
                            current = getattr(current, part)
                        else:
                            current = None
                            break
                    value = current
                else:
                    # 直接属性访问
                    value = getattr(source_data, source_key, None)

            # 处理列表类型的 source_key
            elif isinstance(source_key, list):
                current = source_data
                for part in source_key:
                    if isinstance(current, dict):
                        current = current.get(part)
                    elif hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        current = None
                        break
                value = current

            if value is not None:
                mapped_data[target_key] = value

        return WorkflowData(
            content=mapped_data.get("content", source_data.content),
            metadata=source_data.metadata,
            prompt=source_data.prompt,
            additional_params=mapped_data,
        )

    async def execute_node_with_retry(
        self, node: WorkflowNode, input_data: List[WorkflowData], max_retries: int = 3
    ):
        retries = 0
        while retries < max_retries:
            try:
                return await node.agent.execute(
                    inputs=input_data,
                    prompt_template=node.prompt_template,
                    **node.config,
                )
            except Exception as e:
                retries += 1
                if retries == max_retries:
                    raise Exception(
                        f"节点 {node.id} 执行失败，已重试 {max_retries} 次: {str(e)}"
                    )
                await asyncio.sleep(1 * retries)

    def validate_workflow(self):
        """验证工作流的完整性和正确性"""
        # 检查是否存在循环依赖
        if self._has_cycle():
            raise ValueError("工作流中存在循环依赖")

        # 检查所有节点的参数映射是否有效
        for edge in self.edges:
            source_node = self.nodes[edge.source_id]
            target_node = self.nodes[edge.target_id]

            # 验证参数映射
            for target_key, source_key in edge.data_mapping.items():
                if not self._is_valid_mapping(
                    source_node, target_node, source_key, target_key
                ):
                    raise ValueError(f"无效的参数映射: {source_key} -> {target_key}")

    def register_node(self, node_id: str, node_type: str, **kwargs):
        """注册节点"""
        if node_type == "startNode":
            node_id = "start_node"  # 确保 start 节点的 ID 统一为 start_node

    async def _execute_sub_workflow_iteration(
        self, node: WorkflowNode, inputs: List[WorkflowData], context: dict
    ) -> WorkflowData:
        sub_workflow_config = node.config.get("sub_workflow", {})
        return await node.agent.execute(
            inputs=inputs,
            prompt_template=node.prompt_template,
            sub_workflow_config=sub_workflow_config,
            **node.config,
        )

    async def execute_workflow_stream(
        self, start_node_id: str, initial_context: Dict = None
    ):
        """以流式方式执行工作流"""
        state_machine = ParallelStreamStateMachine()

        try:
            # 初始化上下文
            context = initial_context or {}
            visited = set()

            # 开始执行工作流
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

            # 只有在所有依赖都满足时才标记为已访问
            if node_id in visited:
                logger.info(f"节点 {node_id} 已经执行过，跳过")
                return

            # 如果提供了层级信息，使用层级同步器
            if level_index is not None and total_nodes is not None:
                async with self.level_synchronizer(level_index, total_nodes):
                    logger.info(f"节点 {node_id} 等待同层节点就绪")
                    # 执行节点的主要逻辑
                    async for event in self._execute_node_stream_core(
                        node_id,
                        node,
                        context,
                        visited,
                        state_machine,
                        level_index=level_index,
                        total_nodes=total_nodes,
                    ):
                        yield event
            else:
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

        logger.info(f"开始流式执行节点: {node_id} (类型: {node.config['type']})")
        logger.info(f"当前已访问节点: {visited}")
        logger.info(f"当前上下文包含节点: {list(context.keys())}")

        # 更新节点状态为开始
        state_machine.update_node_state(node_id, StreamState.NODE_STARTED)
        yield WorkflowEvent(
            event_type=WorkflowEventType.NODE_START,
            node_id=node_id,
            metadata={
                "state": StreamState.NODE_STARTED.value,
                "execution_order": len(visited),
            },
        )

        yield WorkflowEvent(
            event_type=WorkflowEventType.NODE_RUNNING,
            node_id=node_id,
            data=None,
            metadata={
                "state": StreamState.NODE_RUNNING.value,
            },
        )

        # 准备节点输入数据
        input_data = []
        for edge in self.edges:
            if edge["to"] == node_id and edge["from"] in context:
                input_data.append(context[edge["from"]])

        result = await self._execute_node(node_id, input_data)
        context[node_id] = result
        yield WorkflowEvent(
            event_type=WorkflowEventType.NODE_END,
            node_id=node_id,
            data=result.content,
            metadata={"state": "completed"},
        )

        yield WorkflowEvent(
            event_type=WorkflowEventType.NODE_END,
            node_id=node_id,
            metadata={"state": "completed"},
        )

        next_nodes = self.get_next_nodes(node_id)
        logger.info(f"节点 {node_id} 执行完成，后续节点: {next_nodes}")

        # 获取下一层级的节点数量
        next_level_total = len(next_nodes)
        next_level_index = (level_index or 0) + 1 if level_index is not None else None

        for next_node_id in next_nodes:
            # 对于聚合节点，检查是否所有依赖都已完成
            if self.nodes[next_node_id].config["type"] == "aggregatorNode":
                deps = {
                    edge["from"] for edge in self.edges if edge["to"] == next_node_id
                }
                if not deps.issubset(visited):
                    continue

            logger.info(f"准备执行后续节点: {next_node_id}")
            async for event in self._execute_node_stream(
                next_node_id,
                context,
                visited,
                state_machine,
                level_index=next_level_index,
                total_nodes=next_level_total,
            ):
                yield event

    def _transition_to(
        self, state_machine: StreamStateMachine, new_state: StreamState
    ) -> bool:
        """
        尝试转换到新状态
        """
        # 添加错误状态的特殊处理
        if new_state in [StreamState.WORKFLOW_ERROR, StreamState.NODE_ERROR]:
            state_machine.current_state = new_state
            return True

        # 常规状态转换检查
        if new_state in state_machine.transitions[state_machine.current_state]:
            state_machine.current_state = new_state
            return True
        return False

    def _create_error_event(
        self,
        message: str,
        error_type: str = "StateTransitionError",
        details: dict = None,
    ) -> WorkflowEvent:
        """
        创建错误事件
        :param message: 错误信息
        :param error_type: 错误类型
        :param details: 错误详情
        """
        metadata = {"error_type": error_type, "details": details or {}}

        return WorkflowEvent(
            event_type=WorkflowEventType.NODE_ERROR,
            node_id="state_machine",
            data=message,
            metadata=metadata,
        )

    def _validate_node_dependencies(self, node_id: str, context: dict):
        """验证节点的所有依赖是否都已就绪"""
        incoming_edges = [edge for edge in self.edges if edge["to"] == node_id]
        missing_deps = []

        for edge in incoming_edges:
            from_node_id = edge["from"]
            if from_node_id not in context:
                missing_deps.append(from_node_id)

        if missing_deps:
            raise ValueError(f"节点 {node_id} 的依赖节点未就绪: {missing_deps}")

    async def _execute_parallel_nodes(self, nodes: List[str], context: dict) -> Dict:
        """并行执行多个节点"""
        tasks = []
        for node_id in nodes:
            input_data = self._prepare_node_input(node_id, context)
            task = self._execute_node(node_id, input_data)
            tasks.append((node_id, task))

        # 使用 asyncio.gather 并行执行所有任务
        results = await asyncio.gather(*[task for _, task in tasks])

        # 返回结果字典
        return {node_id: result for (node_id, _), result in zip(tasks, results)}

    async def _execute_parallel_branch(
        self,
        node_id: str,
        branch_data: dict,
        context: dict,
        state_machine: ParallelStreamStateMachine,
    ):
        """执行并行分支"""
        node = self.nodes[node_id]
        tasks = []

        # 获取分支路由信息
        routing = node.config.get("routing", {})

        # 为每个分支创建执行任务
        for segment_key, content in branch_data.items():
            if segment_key in routing:
                target_nodes = routing[segment_key]
                if isinstance(target_nodes, str):
                    target_nodes = [target_nodes]

                for target_node_id in target_nodes:
                    # 创建分支上下文
                    branch_context = {
                        node_id: WorkflowData(
                            content=content, metadata={"segment_key": segment_key}
                        )
                    }

                    # 创建分支执行任务
                    task = self._execute_node_stream(
                        target_node_id,
                        branch_context,
                        set(),  # 新的访问集合
                        state_machine,
                    )
                    tasks.append((target_node_id, task))

        # 使用 asyncio.as_completed 并行执行并获取事件流
        for future in asyncio.as_completed([task for _, task in tasks]):
            try:
                async for event in await future:
                    yield event
            except Exception as e:
                logger.error(f"分支执行错误: {str(e)}")
                raise

    async def _execute_branch_node(self, node_id: str, branch_context: dict):
        """执行单个分支节点"""
        node = self.nodes[node_id]
        input_data = self._prepare_node_input(node_id, branch_context)
        return await self._execute_node(node_id, input_data)

    async def _handle_node_stream(
        self, node_id: str, stream, state_machine: ParallelStreamStateMachine
    ):
        """处理节点的事件流"""
        try:
            async for event in stream:
                # 更新节点状态
                if event.event_type == WorkflowEventType.NODE_ERROR:
                    state_machine.update_node_state(
                        node_id, StreamState.NODE_ERROR, error=event.data
                    )
                    yield event
                    break
                elif event.event_type == WorkflowEventType.NODE_END:
                    state_machine.update_node_state(node_id, StreamState.NODE_COMPLETED)
                    # 如果事件包含结果数据，保存它
                    if isinstance(event.data, dict) and "content" in event.data:
                        # 将结果数据添加到事件中
                        event.metadata["result"] = WorkflowData(
                            content=event.data["content"],
                            metadata=event.data.get("metadata", {}),
                            prompt=event.data.get("prompt"),
                            additional_params=event.data.get("additional_params", {}),
                        )
                    yield event
                else:
                    # 传递其他事件
                    yield event

        except Exception as e:
            logger.error(f"处理节点 {node_id} 的事件流时出错: {str(e)}")
            state_machine.update_node_state(
                node_id, StreamState.NODE_ERROR, error=str(e)
            )
            # 生成错误事件
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

    def get_next_nodes(self, node_id: str) -> List[str]:
        """
        获取指定节点的所有后续节点
        :param node_id: 当前节点ID
        :return: 后续节点ID列表
        """
        next_nodes = []
        for edge in self.edges:
            if edge["from"] == node_id:
                next_nodes.append(edge["to"])

        logger.info(f"节点 {node_id} 的后续节点: {next_nodes}")
        return next_nodes

    async def _execute_workflow_level(
        self, level_nodes: List[str], context: dict
    ) -> Dict:
        """执行单个层级的所有节点"""
        # 检查是否有并行执行的节点
        parallel_nodes = [
            node_id
            for node_id in level_nodes
            if self.nodes[node_id].metadata.get("parallel_execution")
        ]

        if parallel_nodes:
            # 并行执行节点
            parallel_results = await self._execute_parallel_nodes(
                parallel_nodes, context
            )
            context.update(parallel_results)

        # 执行其他节点
        sequential_nodes = [
            node_id for node_id in level_nodes if node_id not in parallel_nodes
        ]
        for node_id in sequential_nodes:
            result = await self._execute_node(
                node_id, self._prepare_node_input(node_id, context)
            )
            context[node_id] = result

        return context
