from typing import List, Dict, Any
from .base import BaseAgent
from ..core.dto import WorkflowData
from ..agents.llm_agents import LLMAgentFactory
import json
import logging

logger = logging.getLogger(__name__)


class SplitterAgent(BaseAgent):
    def __init__(self, split_config: Dict = None):
        self.split_config = split_config or {}
        self.llm_agent = None

    async def execute(
        self, inputs: List[WorkflowData], prompt_template: str = None, **kwargs
    ) -> WorkflowData:
        if not inputs or not inputs[0].content:
            raise ValueError("分段节点需要输入数据")

        logger.info("开始执行分段处理")
        logger.info(f"输入内容长度: {len(inputs[0].content)}")

        # 获取分段配置
        num_segments = kwargs.get("num_segments", 3)
        optimize_content = kwargs.get("optimize_content", False)
        model_name = kwargs.get("model_name", "gpt-3.5-turbo")
        routing = kwargs.get("routing", {})

        logger.info(
            f"分段配置: 段数={num_segments}, 优化内容={optimize_content}, 模型={model_name}"
        )

        # 初始化 LLM agent
        if not self.llm_agent:
            logger.info(f"初始化 LLM agent: {model_name}")
            self.llm_agent = LLMAgentFactory.create_agent(model_name)

        content = inputs[0].content

        # 构建分段提示词
        logger.info("构建分段提示词")
        split_prompt = f"""请将以下内容分成{num_segments}个段落。

        要求：
        1. 必须返回标准的 JSON 格式
        2. 每个段落使用 title1, title2, title3...（直到title{num_segments}）作为键名
        3. 不要添加任何其他解释性文字
        4. 确保返回的是可解析的 JSON 字符串

        输入内容:
        {content}

        {("请对每个部分进行适当的优化和改写。" if optimize_content else "请保持原始内容，只进行分段。")}

        示例返回格式：
        {{
            "title1": "第一段内容",
            "title2": "第二段内容",
            "title3": "第三段内容"
        }}

        请严格按照上述 JSON 格式返回结果。"""

        try:
            logger.info("开始调用 LLM 进行分段")
            split_result = await self.llm_agent.execute(
                inputs=[WorkflowData(content=split_prompt)],
                temperature=0.3,
            )

            # 尝试清理和解析 JSON
            logger.info("开始解析 LLM 返回结果")
            content = split_result.content.strip()

            # JSON 清理和解析过程
            if content.startswith("```") and content.endswith("```"):
                logger.info("清理代码块标记")
                content = content[3:-3].strip()
            if content.startswith("json") or content.startswith("JSON"):
                logger.info("清理 JSON 标记")
                content = content[4:].strip()

            try:
                logger.info("尝试解析 JSON")
                segments = json.loads(content)
                logger.info(f"成功解析 JSON，获得 {len(segments)} 个段落")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败: {str(e)}，使用备用分段方法")
                text_segments = content.split("\n\n")[:num_segments]
                segments = {
                    f"title{i+1}": segment.strip()
                    for i, segment in enumerate(text_segments)
                }
                logger.info(f"使用备用方法分段完成，获得 {len(segments)} 个段落")

            # 验证分段数量
            if len(segments) != num_segments:
                missing_segments = num_segments - len(segments)
                logger.warning(f"段落数量不足，补充 {missing_segments} 个空段落")
                for i in range(len(segments) + 1, num_segments + 1):
                    segments[f"title{i}"] = f"第{i}部分内容"

            # 修改路由映射逻辑
            segment_routing = {}
            for i in range(num_segments):
                title_key = f"title{i+1}"
                # 确保每个段落都有对应的目标节点
                target_nodes = routing.get(str(i), [])
                if not target_nodes:
                    logger.warning(f"段落 {title_key} 没有指定目标节点")
                    continue

                # 确保目标节点是列表格式
                if isinstance(target_nodes, str):
                    target_nodes = [target_nodes]
                elif not isinstance(target_nodes, list):
                    target_nodes = list(target_nodes)

                segment_routing[title_key] = target_nodes
                logger.info(f"段落 {title_key} 将路由到节点: {target_nodes}")

            # 修改 segment_contents 的构建方式
            segment_contents = {}
            for title_key, content in segments.items():
                # 获取该段落对应的目标节点列表
                target_nodes = segment_routing.get(title_key, [])

                # 为每个目标节点创建只包含对应段落的内容
                for target_node in target_nodes:
                    segment_contents[target_node] = {
                        "content": content,  # 只包含当前段落的内容
                        "source_key": title_key,
                        "targets": [target_node],
                    }

            result = WorkflowData(
                content=segments,
                metadata={
                    "num_segments": num_segments,
                    "segment_routing": segment_routing,
                    "parallel_execution": True,
                    "segment_contents": segment_contents,  # 更新后的 segment_contents
                    "optimized": optimize_content,
                    "model_used": model_name,
                    "split_config": kwargs,
                },
                prompt=split_prompt,
            )

            return result

        except Exception as e:
            logger.error(f"分段处理失败: {str(e)}", exc_info=True)
            raise ValueError(f"分段处理失败: {str(e)}")
