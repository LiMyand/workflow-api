from typing import List
from .base import BaseAgent
from ..core.dto import WorkflowData
from ..agents.llm_agents import LLMAgentFactory
import re
import logging

logger = logging.getLogger("agents.branch")


class BranchAgent(BaseAgent):
    def __init__(self, evaluation_mode: str = "direct"):
        self.evaluation_mode = evaluation_mode  # "direct" 或 "llm"
        self.llm_agent = None  # 延迟初始化，在需要时才创建

    async def execute(
        self, inputs: List[WorkflowData], prompt_template: str = None, **kwargs
    ) -> WorkflowData:
        logger.debug("=" * 50)
        logger.debug("分支节点执行开始")

        if not inputs or not inputs[0].content:
            raise ValueError("分支节点需要输入数据")

        print("================")
        print(inputs)
        print("================")

        # 记录输入数据
        logger.info(f"分支节点接收到的输入: {inputs[0].content}")
        logger.info(f"分支节点的配置: {kwargs}")

        # 获取条件配置
        condition = kwargs.get("condition", {})
        true_branch = kwargs.get("true_branch")
        false_branch = kwargs.get("false_branch")

        # 记录分支配置
        logger.info(f"分支条件配置: {condition}")
        logger.info(f"True分支: {true_branch}")
        logger.info(f"False分支: {false_branch}")

        # 根据评估模式判断条件
        if self.evaluation_mode == "direct":
            result = self._evaluate_condition(inputs[0].content, condition)
            logger.info(f"Direct模式评估结果: {result}")
        else:  # llm 模式
            result = await self._evaluate_with_llm(inputs[0].content, condition)
            logger.info(f"LLM模式评估结果: {result}")

        logger.info(f"分支节点最终结果: {result}")

        # 根据结果选择分支
        selected_branch = true_branch if result else false_branch

        return WorkflowData(
            content=str(result).lower(),  # 确保返回标准化的布尔值字符串
            metadata={
                "evaluation_mode": self.evaluation_mode,
                "condition": condition,
                "condition_result": result,
                "selected_branch": selected_branch,
                "true_branch": true_branch,
                "false_branch": false_branch,
                "input_content": inputs[0].content,  # 添加输入内容到元数据
            },
            prompt=f"Branch evaluation result: {result}",
        )

    def _evaluate_condition(self, content: str, condition: dict) -> bool:
        condition_type = condition.get("type", "contains")
        value = condition.get("value", "")

        if condition_type == "contains":
            return value.lower() in str(content).lower()
        elif condition_type == "regex":
            return bool(re.search(value, str(content)))
        elif condition_type == "equals":
            return str(content).lower() == str(value).lower()
        elif condition_type == "greater_than":
            return float(content) > float(value)
        elif condition_type == "less_than":
            return float(content) < float(value)
        return False

    async def _evaluate_with_llm(self, content: str, condition: dict) -> bool:
        prompt = condition.get("prompt", "")
        if not prompt:
            raise ValueError("LLM 模式需要提供 prompt")

        # 记录 LLM 评估的输入
        logger.info(f"LLM评估的输入内容: {content}")
        logger.info(f"LLM评估的条件: {prompt}")

        # 修改提示词，要求明确返回 true 或 false
        formatted_prompt = f"""请根据以下内容进行判断，只返回 true 或 false：

            输入内容：
            {content}

            判断条件：
            {prompt}

            请只返回 true 或 false，不要包含其他内容。"""

        logger.info(f"发送给LLM的完整提示词: {formatted_prompt}")

        if not self.llm_agent:
            self.llm_agent = LLMAgentFactory.create_agent(
                condition.get("model_name", "gpt-3.5-turbo")
            )

        result = await self.llm_agent.execute(
            inputs=[WorkflowData(content=formatted_prompt)],
            temperature=0.1,
        )

        # 记录 LLM 返回的原始结果
        logger.info(f"LLM返回的原始结果: {result.content}")

        # 清理和标准化结果
        result_text = result.content.strip().lower()
        if result_text not in ["true", "false"]:
            logger.warning(f"LLM 返回了非标准结果: {result_text}，将默认返回 false")
            return False

        return result_text == "true"
