"""
LLM适配器模块
统一的Anthropic Claude API调用接口
支持流式/非流式、工具调用、多轮对话等功能
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# 任务类型到max_tokens的映射
# 根据任务复杂度和预期输出长度动态设置
TASK_TOKEN_LIMITS = {
    # Phase 5 写作任务（需要大量tokens）
    'section-writer': 8192,
    'section-iterative-optimizer': 8192,
    'abstract-generator': 2048,
    'abstract-iterative-optimizer': 2048,
    'memo-letter-writer': 4096,
    'fact-checker': 4096,
    
    # Phase 3 建模任务（需要生成完整代码和公式）
    'model-builder': 8192,
    'model-solver': 8192,
    'model-selector': 4096,
    'model-justification-generator': 6144,
    'hybrid-model-designer': 6144,
    'physics-informed-nn': 8192,
    'neural-operators': 8192,
    'kan-networks': 8192,
    'transformer-forecasting': 8192,
    'reinforcement-learning': 8192,
    'causal-inference': 8192,
    
    # Phase 2 分析任务
    'problem-decomposer': 6144,
    'sub-problem-analyzer': 4096,
    'assumption-generator': 4096,
    'variable-definer': 4096,
    
    # Phase 4 验证任务
    'sensitivity-analyzer': 6144,
    'model-validator': 4096,
    'result-interpreter': 4096,
    
    # Phase 6 可视化任务
    'chart-generator': 6144,
    'figure-narrative-generator': 4096,
    'infographic-generator': 4096,
    
    # Phase 8 质量检查
    'consistency-checker': 4096,
    'hallucination-detector': 4096,
    
    # Phase 9 优化任务
    'final-polisher': 6144,
    'academic-english-optimizer': 6144,
    
    # 默认值
    'default': 4096
}


def get_max_tokens_for_skill(skill_name: str) -> int:
    """
    根据skill名称获取适当的max_tokens值
    
    Args:
        skill_name: 技能名称
        
    Returns:
        推荐的max_tokens值
    """
    return TASK_TOKEN_LIMITS.get(skill_name, TASK_TOKEN_LIMITS['default'])


class LLMAdapter:
    """
    Anthropic Claude API 适配器

    提供统一的LLM调用接口，支持多种调用模式
    """

    # 默认模型配置
    DEFAULT_MODELS = {
        "smart": "claude-3-5-sonnet-20241022",      # 高质量任务
        "fast": "claude-3-5-haiku-20241022",        # 快速任务
        "legacy": "claude-3-opus-20240229",         # 兼容旧版
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "smart",
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        """
        初始化LLM适配器

        Args:
            api_key: Anthropic API密钥 (None时从环境变量读取)
            model: 模型名称 或 预设名称 (smart/fast/legacy)
            max_retries: 最大重试次数
            base_delay: 基础延迟时间(秒)
        """
        import anthropic
        import os

        # 获取API密钥
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set, LLM calls will fail")

        # 解析模型名称
        if model in self.DEFAULT_MODELS:
            self.model = self.DEFAULT_MODELS[model]
        else:
            self.model = model

        # 配置重试
        self.max_retries = max_retries
        self.base_delay = base_delay

        # 创建客户端
        if self.api_key:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("LLM client not initialized due to missing API key")

        # 统计信息
        self.stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_errors": 0,
            "calls_by_model": {}
        }

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        response_format: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        同步完成 (单次调用)

        Args:
            prompt: 用户提示词
            max_tokens: 最大生成token数
            temperature: 温度参数 (0-1)
            response_format: 响应格式 ("json" 或 "text")
            system_prompt: 系统提示词
            stop_sequences: 停止序列
            top_p: top-p采样参数

        Returns:
            LLM响应文本
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized. Set ANTHROPIC_API_KEY.")

        # 构建消息
        messages = [{"role": "user", "content": prompt}]

        # 如果需要JSON格式，在提示词中说明
        if response_format == "json":
            messages[0]["content"] += "\n\nRespond in JSON format only."

        # 执行调用
        for attempt in range(self.max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                    system=system_prompt,
                    stop_sequences=stop_sequences,
                    top_p=top_p
                )

                # 更新统计
                self._update_stats(response)

                # 返回文本内容
                return response.content[0].text

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt == self.max_retries - 1:
                    self.stats["total_errors"] += 1
                    raise

                # 指数退避
                await asyncio.sleep(self.base_delay * (2 ** attempt))

        return ""  # 不会到达这里

    async def stream_complete(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        流式完成

        Args:
            prompt: 用户提示词
            max_tokens: 最大生成token数
            temperature: 温度参数
            system_prompt: 系统提示词

        Yields:
            生成的文本片段
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized. Set ANTHROPIC_API_KEY.")

        messages = [{"role": "user", "content": prompt}]

        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                system=system_prompt
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.exception(f"LLM stream failed: {e}")
            raise

    async def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        带工具调用的完成

        Args:
            prompt: 用户提示词
            tools: 工具定义列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            system_prompt: 系统提示词

        Returns:
            包含响应内容和工具调用结果的字典
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized. Set ANTHROPIC_API_KEY.")

        messages = [{"role": "user", "content": prompt}]

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            system=system_prompt,
            tools=tools
        )

        result = {
            "content": [],
            "tool_use": None
        }

        for block in response.content:
            if block.type == "text":
                result["content"].append(block.text)
            elif block.type == "tool_use":
                result["tool_use"] = {
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                }

        return result

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        多轮对话

        Args:
            messages: 消息历史 [{"role": "user/assistant", "content": "..."}]
            max_tokens: 最大生成token数
            temperature: 温度参数
            system_prompt: 系统提示词

        Returns:
            LLM响应文本
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized. Set ANTHROPIC_API_KEY.")

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            system=system_prompt
        )

        self._update_stats(response)
        return response.content[0].text

    async def evaluate(
        self,
        text: str,
        criteria: Dict[str, Any],
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        使用LLM评估文本质量

        Args:
            text: 待评估的文本
            criteria: 评估标准
            return_details: 是否返回详细评估

        Returns:
            评分 (0-1) 或 详细评估结果
        """
        prompt = f"""Evaluate the following text based on the criteria below.

Criteria:
{json.dumps(criteria, indent=2)}

Text to evaluate:
{text}

Provide:
1. A score from 0.0 to 1.0
2. Brief feedback (strengths and weaknesses)

Return your response in JSON format:
{{
  "score": 0.0,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggestions": ["..."]
}}
"""

        try:
            response = await self.complete(prompt, response_format="json")
            result = json.loads(response)

            if return_details:
                return result
            return result.get("score", 0.0)

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.0 if not return_details else {"score": 0.0, "error": str(e)}

    async def batch_complete(
        self,
        prompts: List[str],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        concurrency: int = 5
    ) -> List[str]:
        """
        批量完成 (并行处理)

        Args:
            prompts: 提示词列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            concurrency: 并发数

        Returns:
            响应列表
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def complete_one(prompt: str) -> str:
            async with semaphore:
                return await self.complete(prompt, max_tokens, temperature)

        tasks = [complete_one(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def _update_stats(self, response):
        """更新调用统计"""
        self.stats["total_calls"] += 1

        # 统计token使用
        if hasattr(response, "usage"):
            usage = response.usage
            self.stats["total_tokens"] += usage.input_tokens + usage.output_tokens

        # 按模型统计
        model = response.model if hasattr(response, "model") else self.model
        self.stats["calls_by_model"][model] = self.stats["calls_by_model"].get(model, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """获取调用统计"""
        return {
            **self.stats,
            "model": self.model,
            "has_api_key": self.api_key is not None
        }

    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_errors": 0,
            "calls_by_model": {}
        }


class MockLLMAdapter(LLMAdapter):
    """
    模拟LLM适配器 (用于测试)
    """

    def __init__(self, model: str = "mock"):
        """初始化模拟适配器"""
        self.model = model
        self.api_key = "mock"
        self.client = None
        self.stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_errors": 0,
            "calls_by_model": {}
        }
        self.responses = {}

    def set_mock_response(self, prompt_pattern: str, response: str):
        """设置模拟响应"""
        self.responses[prompt_pattern] = response

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        response_format: Optional[str] = None,
        **kwargs
    ) -> str:
        """模拟完成"""
        self.stats["total_calls"] += 1

        # 检查是否有预设响应
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return response

        # 默认响应
        if response_format == "json":
            return json.dumps({"result": "mock response", "score": 0.85})
        return "This is a mock response for testing purposes."

    async def stream_complete(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """模拟流式完成"""
        response = await self.complete(prompt)
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.01)


def create_llm_adapter(
    api_key: Optional[str] = None,
    model: str = "smart",
    mock: bool = False
) -> Union[LLMAdapter, MockLLMAdapter]:
    """
    创建LLM适配器实例

    Args:
        api_key: API密钥
        model: 模型名称
        mock: 是否使用模拟适配器

    Returns:
        LLM适配器实例
    """
    if mock:
        return MockLLMAdapter(model)
    return LLMAdapter(api_key=api_key, model=model)


# 预定义的系统提示词
SYSTEM_PROMPTS = {
    "academic_writing": """You are an expert academic writer specializing in MCM/ICM mathematical modeling papers.
Your writing should be:
- Clear and concise
- Mathematically precise
- Well-structured with proper transitions
- Free of clichés and redundancy
- Quantitative and data-driven when possible""",

    "abstract_writer": """You are an expert at writing O-award level MCM/ICM abstracts.
An excellent abstract:
- Opens with a compelling hook (avoid clichés)
- States the problem clearly
- Describes the methodology
- Highlights innovations
- Presents quantitative results
- Concludes with impact
- Is 300-500 words""",

    "analyst": """You are a mathematical modeling analyst.
Break down problems systematically:
1. Identify key components
2. Make reasonable assumptions
3. Define variables clearly
4. Select appropriate models
5. Validate results""",

    "critic": """You are a rigorous paper critic.
Evaluate papers on:
- Mathematical soundness
- Clarity of exposition
- Quality of visualizations
- Strength of conclusions
- Proper attribution
Be thorough but constructive."""
}


def get_system_prompt(name: str) -> str:
    """获取预定义的系统提示词"""
    return SYSTEM_PROMPTS.get(name, "")
