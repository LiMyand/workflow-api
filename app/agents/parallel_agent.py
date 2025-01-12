from typing import List
import asyncio
from .base import BaseAgent
from ..core.dto import WorkflowData
from ..agents.llm_agents import LLMAgentFactory


class ParallelAgent(BaseAgent):
    def __init__(self, parallel_config: dict = None):
        self.parallel_config = parallel_config or {}

    async def execute(
        self, inputs: List[WorkflowData], prompt_template: str = None, **kwargs
    ) -> WorkflowData:
        if not inputs:
            raise ValueError("并行节点需要输入数据")

        parallel_tasks = kwargs.get("parallel_tasks", [])

        # 支持批量处理
        batch_size = kwargs.get("batch_size", None)
        if batch_size:
            batches = [
                parallel_tasks[i : i + batch_size]
                for i in range(0, len(parallel_tasks), batch_size)
            ]
            all_results = []
            for batch in batches:
                tasks = [self._execute_task(task, inputs) for task in batch]
                batch_results = await asyncio.gather(*tasks)
                all_results.extend(batch_results)
            results = all_results
        else:
            tasks = [self._execute_task(task, inputs) for task in parallel_tasks]
            results = await asyncio.gather(*tasks)

        # 支持多种结果聚合方式
        aggregation_mode = kwargs.get("aggregation_mode", "concat")
        if aggregation_mode == "concat":
            combined_content = "\n".join([r.content for r in results])
        elif aggregation_mode == "list":
            combined_content = [r.content for r in results]
        elif aggregation_mode == "dict":
            combined_content = {f"result_{i}": r.content for i, r in enumerate(results)}

        combined_metadata = {f"task_{i}": r.metadata for i, r in enumerate(results)}

        return WorkflowData(
            content=combined_content,
            metadata={
                "parallel_tasks_count": len(results),
                "task_results": combined_metadata,
            },
            prompt="Parallel execution results",
        )

    async def _execute_task(self, task_config, inputs):
        model_name = task_config.get("model_name", "gpt-3.5-turbo")
        task_prompt = task_config.get("prompt_template")
        agent = LLMAgentFactory.create_agent(model_name)

        return await agent.execute(
            inputs=inputs,
            prompt_template=task_prompt,
            **task_config.get("config", {}),
        )
