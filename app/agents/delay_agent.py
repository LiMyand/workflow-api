from typing import List
import asyncio
from .base import BaseAgent
from ..core.dto import WorkflowData


class DelayAgent(BaseAgent):
    async def execute(
        self, inputs: List[WorkflowData], prompt_template: str = None, **kwargs
    ) -> WorkflowData:
        delay_seconds = kwargs.get("delay_seconds", 1)
        await asyncio.sleep(delay_seconds)

        return WorkflowData(
            content=inputs[0].content if inputs else None,
            metadata={"delay_seconds": delay_seconds, "type": "delay"},
            prompt=f"Delayed for {delay_seconds} seconds",
        )
