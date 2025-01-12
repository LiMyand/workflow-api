from abc import ABC, abstractmethod
from typing import List, Optional
from ..core.dto import WorkflowData


class BaseAgent(ABC):
    @abstractmethod
    async def execute(
        self, inputs: List[WorkflowData], prompt_template: str = None, **kwargs
    ) -> WorkflowData:
        pass


class StartAgent(BaseAgent):
    async def execute(self, inputs=None, **kwargs):
        # 如果有输入数据，直接返回第一个输入
        if inputs and len(inputs) > 0:
            return inputs[0]

        # 如果没有输入数据，返回空数据
        return WorkflowData(content="", metadata={"node_type": "start"})


class EndAgent(BaseAgent):
    async def execute(
        self, inputs: List[WorkflowData], prompt_template: str = None, **kwargs
    ) -> WorkflowData:
        if not inputs:
            return WorkflowData(
                content="工作流执行完成",
                metadata={"workflow_status": "completed", "is_end_node": True},
            )
        return WorkflowData(
            content=inputs[0].content,
            metadata={
                **inputs[0].metadata,
                "workflow_status": "completed",
                "is_end_node": True,
            },
            prompt="工作流执行完成",
        )
