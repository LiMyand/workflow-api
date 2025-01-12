from typing import List
from .base import BaseAgent
from ..core.dto import WorkflowData
from ..agents.llm_agents import LLMAgentFactory
import json
from ..core.iteration import SubWorkflow, IterationContext


class IterationAgent(BaseAgent):
    def __init__(self, iteration_type: str = "custom"):
        super().__init__()
        self.iteration_type = iteration_type
        self.sub_workflow = None
        self.iteration_context = IterationContext()

    async def execute(
        self, inputs: List[WorkflowData], prompt_template: str = None, **kwargs
    ) -> WorkflowData:
        # 保持原有的执行逻辑
        if self.iteration_type == "custom":
            return await self._execute_original(inputs, prompt_template, **kwargs)

        # 新的子工作流执行逻辑
        elif self.iteration_type == "sub_workflow":
            return await self._execute_sub_workflow(inputs, **kwargs)

    async def _execute_original(
        self, inputs: List[WorkflowData], prompt_template: str = None, **kwargs
    ) -> WorkflowData:
        if not inputs or not inputs[0].content:
            raise ValueError("迭代节点需要输入数据")

        # 获取迭代配置
        step = kwargs.get("current_step", 0)
        iterations = kwargs.get("iterations", [])  # 这里应该是一个迭代配置列表
        model_name = kwargs.get("model_name", "gpt-3.5-turbo")
        iteration_count = kwargs.get("iteration_count", 0)
        max_iterations = kwargs.get("max_iterations", 5)

        # 第一次迭代时初始化状态
        if step == 0 and iteration_count == 0:
            self.iteration_state = {
                "original_content": inputs[0].content,
                "intermediate_results": [],
                "final_result": None,
                "current_iteration": 0,
            }
            self.llm_agent = LLMAgentFactory.create_agent(model_name)

        content = inputs[0].content
        metadata = {
            "iteration_count": iteration_count,
            "max_iterations": max_iterations,
        }

        # 如果还没有达到最大迭代次数，执行当前步骤
        if iteration_count < max_iterations:
            # 获取当前迭代的配置
            current_iteration = iterations[iteration_count % len(iterations)]
            result = await self._execute_iteration_step(
                content=content,
                iteration_config=current_iteration,
                step=step,
                iteration_count=iteration_count,
            )
            metadata["needs_more_iteration"] = True
        else:
            result = WorkflowData(
                content=content,
                metadata={"iteration_complete": True},
                prompt="迭代完成",
            )
            metadata["needs_more_iteration"] = False

        return WorkflowData(
            content=result.content,
            metadata={
                **result.metadata,
                **metadata,
            },
            prompt=result.prompt,
        )

    async def _execute_iteration_step(
        self,
        content: str,
        iteration_config: dict,
        step: int,
        iteration_count: int,
    ) -> WorkflowData:
        """执行单个迭代步骤"""
        prompt_template = iteration_config.get("prompt_template")
        if not prompt_template:
            raise ValueError(f"步骤 {step + 1} 未提供 prompt_template")

        # 组合提示词，支持更多变量
        formatted_prompt = prompt_template.format(
            content=content,
            iteration_count=iteration_count + 1,
            previous_results=self.iteration_state["intermediate_results"],
            original_content=self.iteration_state["original_content"],
            **iteration_config.get("prompt_variables", {}),  # 支持额外的提示词变量
        )

        llm_result = await self.llm_agent.execute(
            inputs=[WorkflowData(content=content)],
            prompt_template=formatted_prompt,
            **iteration_config.get("llm_config", {}),  # 每次迭代可以有不同的LLM配置
        )

        self.iteration_state["intermediate_results"].append(llm_result.content)
        return llm_result

    async def _execute_sub_workflow(
        self, inputs: List[WorkflowData], **kwargs
    ) -> WorkflowData:
        if not self.sub_workflow:
            self.sub_workflow = SubWorkflow(kwargs.get("sub_workflow_config", {}))

        while not self._should_stop():
            # 准备迭代上下文
            iteration_context = self._prepare_iteration_context(inputs)

            # 执行子工作流
            sub_workflow_result = await self.sub_workflow.execute(iteration_context)

            # 处理迭代结果
            self.iteration_context.add_iteration_result(sub_workflow_result)

            # 检查终止条件
            if self._check_termination_conditions(sub_workflow_result):
                break

        return self._prepare_final_result()
