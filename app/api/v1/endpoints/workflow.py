from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from typing import Dict, Any, List, Optional, Union
from ....core.dto import WorkflowData
from ....core.manager import WorkflowManager
from ....agents.llm_agents import LLMAgentFactory
from ....core.workflow import WorkflowNode, WorkflowEdge
from ....core.config import settings
from datetime import datetime
from ....core.logger import logger
from ....agents.base import StartAgent, EndAgent
from ....agents.delay_agent import DelayAgent
from ....agents.aggregator_agent import AggregatorAgent
from ....agents.loop_agent import LoopAgent
from fastapi.responses import StreamingResponse
import json
import asyncio
import traceback

router = APIRouter()


class Position(BaseModel):
    x: float
    y: float


class NodeData(BaseModel):
    model_name: Optional[str] = None
    prompt_template: Optional[str] = None
    config: Dict = {
        "num_segments": 3,
        "optimize_content": False,
        "routing": {},
    }


class Node(BaseModel):
    id: str
    type: str = "llmNode"
    position: Position
    data: NodeData

    @field_validator("type")
    @classmethod
    def validate_node_type(cls, v):
        allowed_types = [
            "startNode",
            "endNode",
            "llmNode",
            "branchNode",
            "iterationNode",
            "parallelNode",
            "aggregatorNode",
            "delayNode",
            "loopNode",
        ]
        if v not in allowed_types:
            raise ValueError(f"不支持的节点类型: {v}. 支持的类型: {allowed_types}")
        return v

    @field_validator("data")
    @classmethod
    def validate_node_data(cls, v, values):
        # 获取节点类型
        node_type = values.data.get("type", "llmNode")

        # 不需要验证模型名称的节点类型列表
        exempt_types = [
            "startNode",
            "endNode",
            "delayNode",  # 延迟节点不需要模型
            "aggregatorNode",  # 聚合节点不需要模型
        ]

        # 对于特殊节点类型，不需要验证模型名称
        if node_type in exempt_types:
            return v

        # 对其他节点类型进行模型验证
        if not v.model_name:
            raise ValueError("非特殊节点必须指定模型名称")

        return v


class EdgeData(BaseModel):
    params_mapping: Dict[str, Union[str, List[str]]] = {}


class Edge(BaseModel):
    id: str
    source: str
    target: str
    type: str = "smoothstep"
    data: EdgeData


class ReactFlowData(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


class WorkflowRequest(BaseModel):
    input_text: str
    reactflow: Dict[str, Any]


@router.post("/execute")
async def execute_workflow(request: WorkflowRequest):
    logger.info("=== 开始执行工作流 ===")

    try:
        workflow = WorkflowManager()

        # 添加开始节点
        start_node = WorkflowNode(
            id="start_node",
            agent=StartAgent(),
            prompt_template=None,
            config={"type": "startNode"},
        )
        workflow.add_node(start_node)

        # 添加结束节点
        end_node = WorkflowNode(
            id="end_node",
            agent=EndAgent(),
            prompt_template=None,
            config={"type": "endNode"},
        )
        workflow.add_node(end_node)

        # 添加其他节点
        first_process_node_id = None
        for node in request.reactflow.nodes:
            if node.type not in ["startNode", "endNode"]:
                logger.info(
                    f"创建节点 [{node.id}] - 类型: {node.type}, 模型: {node.data.model_name}"
                )

                # 根据节点类型创建不同的 agent
                if node.type == "delayNode":
                    agent = DelayAgent()
                elif node.type == "aggregatorNode":
                    agent = AggregatorAgent(
                        node.data.config.get("aggregation_type", "concat")
                    )
                elif node.type == "loopNode":
                    agent = LoopAgent()
                elif node.type == "branchNode":
                    agent = BranchAgent(
                        evaluation_mode=node.data.config.get(
                            "evaluation_mode", "direct"
                        )
                    )
                    workflow_node = WorkflowNode(
                        id=node.id,
                        agent=agent,
                        prompt_template=node.data.prompt_template,
                        config={
                            "type": "branchNode",
                            "condition": node.data.config.get("condition", {}),
                            "true_branch": node.data.config.get("true_branch"),
                            "false_branch": node.data.config.get("false_branch"),
                            "evaluation_mode": node.data.config.get(
                                "evaluation_mode", "direct"
                            ),
                        },
                    )
                    workflow.add_node(workflow_node)
                else:
                    # 其他需要 LLM 的节点类型
                    agent = LLMAgentFactory.create_agent(node.data.model_name)

                workflow_node = WorkflowNode(
                    id=node.id,
                    agent=agent,
                    prompt_template=node.data.prompt_template,
                    config=node.data.config,
                )
                workflow.add_node(workflow_node)
                logger.info(f"节点 [{node.id}] 的提示模板: {node.data.prompt_template}")

                # 记录第一个处理节点的ID
                if first_process_node_id is None:
                    first_process_node_id = node.id

        # 连接开始节点到第一个处理节点
        if first_process_node_id:
            workflow.add_edge(
                from_node_id="start_node", to_node_id=first_process_node_id
            )

        # 添加其他边
        last_node_id = None
        for edge in request.reactflow.edges:
            logger.info(f"添加边: {edge.source} -> {edge.target}")
            workflow.add_edge(
                from_node_id=edge.source,
                to_node_id=edge.target,
                params_mapping=edge.data.params_mapping,
            )
            last_node_id = edge.target

        # 连接最后一个节点到结束节点
        if last_node_id:
            workflow.add_edge(from_node_id=last_node_id, to_node_id="end_node")

        # 正确设置初始上下文
        initial_context = {
            "start_node": WorkflowData(
                content=request.input_text,
                metadata={"is_start_node": True},
                prompt=None,
            )
        }

        results = await workflow.execute_workflow("start_node", initial_context)

        logger.info("=== 工作流执行完成 ===")
        for node_id, data in results.items():
            logger.info(f"节点 [{node_id}] 输出:")
            logger.info(f"  内容: {data.content}")
            logger.info(f"  元数据: {data.metadata}")

        return {
            "workflow_id": f"wf_{datetime.now().timestamp()}",
            "results": {
                node_id: {"content": data.content, "metadata": data.metadata}
                for node_id, data in results.items()
            },
        }

    except Exception as e:
        logger.error(f"工作流执行错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute/stream")
async def execute_workflow_stream(request: WorkflowRequest):
    try:
        workflow = WorkflowManager()

        # 从 reactflow 配置中构建工作流
        for node in request.reactflow["nodes"]:
            workflow.add_node(
                node_id=node["id"],
                node_type=node["type"],
                config={
                    **node["data"].get("config", {}),
                    "evaluation_prompt": node["data"].get("evaluation_prompt"),
                },
                prompt_template=node["data"].get("prompt_template"),
                model_name=node["data"].get("model_name"),
            )

        for edge in request.reactflow["edges"]:
            workflow.add_edge(
                from_node_id=edge["source"],
                to_node_id=edge["target"],
                params_mapping=edge["data"].get("params_mapping", {}),
            )

        # 设置初始上下文
        initial_context = {
            "start_node": WorkflowData(
                content=request.input_text,
                metadata={"is_start_node": True},
                prompt=None,
            )
        }

        logger.info(f"初始上下文: {initial_context}")

        async def event_generator():
            try:
                async for event in workflow.execute_workflow_stream(
                    "start_node", initial_context
                ):
                    event_data = {
                        "type": event.event_type.value,
                        "node_id": event.node_id,
                        "data": event.data,
                        "metadata": event.metadata,
                        "timestamp": event.timestamp,
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    await asyncio.sleep(0.01)

            except Exception as e:
                error_event = {
                    "type": "workflow_error",
                    "error": str(e),
                    "timestamp": datetime.now().timestamp(),
                    "details": {
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                    },
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"工作流执行错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
