from typing import List, Optional
from openai import AsyncOpenAI
from .base import BaseAgent
from ..core.dto import WorkflowData
from ..core.config import settings
from ..core.logger import logger
import httpx
import asyncio
import logging
import json

logger = logging.getLogger("workflow")


class LLMAgentFactory:
    @staticmethod
    def create_agent(model_name: str) -> "BaseLLMAgent":
        if model_name not in settings.AVAILABLE_MODELS:
            raise ValueError(f"不支持的模型: {model_name}")

        model_config = settings.AVAILABLE_MODELS[model_name]
        provider = model_config["provider"]

        if provider == "openai":
            return OpenAIAgent(model_name)
        else:
            raise ValueError(f"不支持的提供商: {provider}")


class BaseLLMAgent(BaseAgent):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_config = settings.AVAILABLE_MODELS[model_name]

    def validate_temperature(self, temperature: float):
        min_temp, max_temp = self.model_config["temperature_range"]
        if not min_temp <= temperature <= max_temp:
            raise ValueError(f"temperature 必须在 {min_temp} 和 {max_temp} 之间")


class OpenAIAgent(BaseLLMAgent):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            timeout=60.0,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                transport=httpx.AsyncHTTPTransport(retries=3),
            ),
        )

    def _build_prompt(
        self, inputs: List[WorkflowData], prompt_template: str = None
    ) -> str:
        if not inputs:
            raise ValueError("输入数据不能为空")

        if not prompt_template:
            return inputs[0].content

        try:
            return prompt_template.format(input=inputs[0].content)
        except Exception as e:
            raise ValueError(f"构建提示词失败: {str(e)}")

    async def execute(
        self,
        inputs: List[WorkflowData],
        prompt_template: str = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> WorkflowData:
        logger.info("=" * 50)
        logger.info(f"OpenAI 调用开始，模型: {self.model_name}")

        # 打印输入数据
        for idx, input_data in enumerate(inputs):
            logger.info(
                f"输入数据 ============================================================================ {idx + 1}:"
            )
            logger.info(
                f"================================================================================================================================================"
            )
            logger.info(
                f"================================================================================================================================================"
            )
            logger.info(
                f"================================================================================================================================================"
            )
            logger.info(
                f"================================================================================================================================================"
            )
            logger.info(
                f"================================================================================================================================================"
            )
            logger.info(
                f"================================================================================================================================================"
            )
            logger.info(
                f"================================================================================================================================================"
            )
            logger.info(f"  内容: {input_data.content}")
            logger.info(
                f"================================================================================================================================================"
            )
            logger.info(
                f"================================================================================================================================================"
            )
            logger.info(
                f"================================================================================================================================================"
            )
            logger.info(f"  元数据: {input_data.metadata}")

        self.validate_temperature(temperature)
        prompt = self._build_prompt(inputs, prompt_template)

        logger.info(f"构建的最终提示词: {prompt}")
        logger.info(f"温度设置: {temperature}")
        logger.info(f"其他参数: {json.dumps(kwargs, ensure_ascii=False, indent=2)}")

        api_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": kwargs.get("max_tokens", 2000),
            "top_p": kwargs.get("top_p", 1.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
        }

        api_params = {k: v for k, v in api_params.items() if v is not None}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(**api_params)
                result = response.choices[0].message.content

                logger.info(
                    f"OpenAI 调用成功，消耗 tokens: {response.usage.total_tokens}"
                )

                return WorkflowData(
                    content=result,
                    metadata={
                        "model": self.model_name,
                        "prompt": prompt,
                        **{k: v for k, v in kwargs.items() if k in api_params},
                    },
                    prompt=prompt,
                )
            except Exception as e:
                logger.warning(
                    f"API 调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt == max_retries - 1:
                    logger.error(f"LLM API 调用失败: {str(e)}")
                    raise
                await asyncio.sleep(2**attempt)
