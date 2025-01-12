from typing import List, Dict, Any
from ..core.workflow import WorkflowData
from .base import BaseAgent
from ..agents.llm_agents import LLMAgentFactory
import logging

logger = logging.getLogger(__name__)


class LoopAgent(BaseAgent):
    def __init__(self):
        self.llm_agent = None
        self.evaluation_agent = None

    async def execute(
        self, inputs: List[WorkflowData], prompt_template: str = None, **kwargs
    ) -> WorkflowData:
        if not prompt_template:
            raise ValueError("循环节点需要提供生成提示词模板")

        evaluation_prompt = kwargs.get("evaluation_prompt")
        if not evaluation_prompt:
            raise ValueError("循环节点需要提供评估提示词模板")

        # 初始化 LLM agents
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        if not self.llm_agent:
            self.llm_agent = LLMAgentFactory.create_agent(model_name)
        if not self.evaluation_agent:
            self.evaluation_agent = LLMAgentFactory.create_agent(model_name)

        max_iterations = kwargs.get("max_iterations", 3)
        current_iteration = 0
        current_content = inputs[0].content if inputs else ""

        while current_iteration < max_iterations:

            generation_inputs = [
                WorkflowData(
                    content=current_content,
                    metadata={"iteration": current_iteration},
                    prompt=prompt_template,
                )
            ]

            generation_result = await self.llm_agent.execute(
                inputs=generation_inputs,
                prompt_template=prompt_template,
                temperature=kwargs.get("generation_temperature", 0.7),
            )
            current_content = generation_result.content

            evaluation_inputs = [
                WorkflowData(
                    content=current_content,
                    metadata={"iteration": current_iteration},
                    prompt=evaluation_prompt,
                )
            ]
            evaluation_result = await self.evaluation_agent.execute(
                inputs=evaluation_inputs,
                prompt_template=evaluation_prompt,
                temperature=kwargs.get("evaluation_temperature", 0.1),
            )

            try:
                result = evaluation_result.content.strip().lower()
                if result == "false":
                    break
                elif result == "true":
                    current_iteration += 1
                    continue
                else:
                    logger.warning(f"评估结果格式不符合预期: {result}")
                    break
            except Exception as e:
                logger.error(f"评估结果处理失败: {str(e)}")
                break

        return WorkflowData(
            content=current_content,
            metadata={
                "iterations_completed": current_iteration,
                "max_iterations": max_iterations,
                "final_evaluation": result if "result" in locals() else None,
            },
            prompt=f"Loop completed after {current_iteration} iterations",
        )
