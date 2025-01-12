from typing import List
from .base import BaseAgent
from ..core.dto import WorkflowData


class AggregatorAgent(BaseAgent):
    def __init__(self, aggregation_type: str = "concat"):
        self.aggregation_type = aggregation_type

    async def execute(
        self, inputs: List[WorkflowData], prompt_template: str = None, **kwargs
    ) -> WorkflowData:
        if not inputs:
            raise ValueError("聚合节点需要输入数据")

        # 获取聚合类型，支持从 kwargs 覆盖初始化时的设置
        aggregation_type = kwargs.get("aggregation_type", self.aggregation_type)

        if aggregation_type == "concat":
            result = "\n".join([data.content for data in inputs])
        elif aggregation_type == "list":
            result = [data.content for data in inputs]
        elif aggregation_type == "join":
            separator = kwargs.get("separator", " ")
            result = separator.join([str(data.content) for data in inputs])
        elif aggregation_type == "dict":
            result = {f"result_{i}": data.content for i, data in enumerate(inputs)}
        elif aggregation_type == "custom":
            # 自定义聚合逻辑
            custom_format = kwargs.get("custom_format", "{content}")
            result = "\n".join(
                [
                    custom_format.format(content=data.content, index=i, **data.metadata)
                    for i, data in enumerate(inputs)
                ]
            )
        else:
            raise ValueError(f"不支持的聚合类型: {aggregation_type}")

        return WorkflowData(
            content=result,
            metadata={
                "aggregation_type": aggregation_type,
                "input_count": len(inputs),
                "aggregation_params": kwargs,
            },
            prompt=f"Aggregated using {aggregation_type}",
        )
