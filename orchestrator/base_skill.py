"""
技能基类模块
定义所有技能的抽象基类、元数据结构和结果类型
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import logging
import re

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """技能执行模式"""
    LLM = "llm"          # 基于LLM调用
    SCRIPT = "script"    # 基于Python脚本执行
    API = "api"          # 基于外部API调用
    HYBRID = "hybrid"    # 混合模式


@dataclass
class SkillMetadata:
    """技能元数据"""
    name: str                           # 技能名称 (唯一标识符)
    phase: int                          # 所属阶段 (1-10)
    execution_mode: Union[ExecutionMode, str]  # 执行模式
    depends_on: List[str] = field(default_factory=list)  # 依赖的其他技能
    outputs: List[str] = field(default_factory=list)      # 输出的状态键
    timeout: int = 300                 # 超时时间(秒)
    fallback: Optional[str] = None     # 失败时的备选技能
    description: str = ""              # 技能描述
    category: str = "general"          # 技能类别

    def __post_init__(self):
        """后处理：将字符串转换为枚举"""
        if isinstance(self.execution_mode, str):
            self.execution_mode = ExecutionMode(self.execution_mode)


@dataclass
class SkillResult:
    """技能执行结果"""
    success: bool                       # 是否成功
    data: Dict[str, Any] = field(default_factory=dict)  # 输出数据
    errors: List[str] = field(default_factory=list)     # 错误信息
    execution_time: float = 0.0         # 执行时间(秒)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "data": self.data,
            "errors": self.errors,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_error(cls, error: Exception, skill_name: str = "") -> 'SkillResult':
        """从异常创建失败结果"""
        return cls(
            success=False,
            errors=[f"{skill_name}: {str(error)}"],
            metadata={"error_type": type(error).__name__}
        )

    @classmethod
    def from_success(cls, data: Dict[str, Any], execution_time: float = 0.0) -> 'SkillResult':
        """创建成功结果"""
        return cls(
            success=True,
            data=data,
            execution_time=execution_time
        )


class BaseSkill(ABC):
    """
    所有技能的抽象基类

    技能是系统中的最小执行单元，每个技能负责一个特定的任务。
    子类必须实现 execute 方法来定义具体的执行逻辑。
    """

    def __init__(self, metadata: SkillMetadata, llm_adapter=None):
        """
        初始化技能

        Args:
            metadata: 技能元数据
            llm_adapter: LLM适配器 (可选)
        """
        self.metadata = metadata
        self.llm = llm_adapter
        self.name = metadata.name
        self.phase = metadata.phase

        logger.debug(f"Initialized skill: {self.name} (Phase {self.phase})")

    @abstractmethod
    async def execute(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SkillResult:
        """
        执行技能的核心方法

        Args:
            state: 全局状态字典
            context: 执行上下文 (包含当前阶段、配置等)

        Returns:
            SkillResult: 执行结果
        """
        pass

    def validate_inputs(self, state: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        验证输入状态是否满足执行条件

        Args:
            state: 全局状态字典

        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []

        # 检查依赖是否满足
        for dep in self.metadata.depends_on:
            if dep not in state:
                errors.append(f"Missing dependency: {dep}")

            # 检查依赖的执行结果
            elif isinstance(state[dep], dict):
                dep_result = state[dep]
                if not dep_result.get('success', True):
                    errors.append(f"Dependency {dep} failed")

        return len(errors) == 0, errors

    def extract_outputs(self, result: SkillResult) -> Dict[str, Any]:
        """
        从执行结果中提取输出

        Args:
            result: 执行结果

        Returns:
            输出字典
        """
        outputs = {}
        for output_key in self.metadata.outputs:
            if output_key in result.data:
                outputs[output_key] = result.data[output_key]
        return outputs

    def get_state_value(self, state: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        从状态中获取值，支持嵌套路径

        Args:
            state: 状态字典
            key: 键名 (支持点号分隔的嵌套路径)
            default: 默认值

        Returns:
            状态值
        """
        keys = key.split('.')
        value = state
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def format_prompt(self, template: str, state: Dict[str, Any]) -> str:
        """
        格式化提示词模板

        Args:
            template: 提示词模板 (支持 {variable} 语法)
            state: 状态字典

        Returns:
            格式化后的提示词
        """
        # 简单的变量替换
        result = template

        # 查找所有 {variable} 占位符
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, template)

        for var in matches:
            value = self.get_state_value(state, var, f"{{{var}}}")
            result = result.replace(f"{{{var}}}", str(value))

        return result


class LLMSkill(BaseSkill):
    """
    基于LLM的技能

    使用提示词模板调用LLM完成特定任务
    """

    def __init__(
        self,
        metadata: SkillMetadata,
        prompt_template: str,
        llm_adapter,
        response_format: str = "text"
    ):
        """
        初始化LLM技能

        Args:
            metadata: 技能元数据
            prompt_template: 提示词模板
            llm_adapter: LLM适配器
            response_format: 响应格式 ("text" 或 "json")
        """
        super().__init__(metadata, llm_adapter)
        self.prompt_template = prompt_template
        self.response_format = response_format

    async def execute(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SkillResult:
        """执行LLM技能"""
        import time
        start_time = time.time()

        # 验证输入
        valid, errors = self.validate_inputs(state)
        if not valid:
            return SkillResult(
                success=False,
                errors=errors,
                metadata={"validation_errors": errors}
            )

        try:
            # 构建提示词
            prompt = self.format_prompt(self.prompt_template, state)

            # 调用LLM
            if self.llm is None:
                raise RuntimeError("LLM adapter not configured")

            response = await self.llm.complete(
                prompt=prompt,
                max_tokens=context.get('max_tokens', 4096),
                temperature=context.get('temperature', 0.7),
                response_format=self.response_format
            )

            # 解析响应
            parsed_data = self._parse_response(response)

            execution_time = time.time() - start_time
            return SkillResult(
                success=True,
                data=parsed_data,
                execution_time=execution_time,
                metadata={
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    "model": getattr(self.llm, 'model', 'unknown')
                }
            )

        except Exception as e:
            logger.exception(f"LLM skill {self.name} failed")
            return SkillResult.from_error(e, self.name)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        if self.response_format == "json":
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # 尝试提取JSON代码块
                match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
                return {"raw_response": response}
        return {"response": response}


class ScriptSkill(BaseSkill):
    """
    基于Python脚本的技能

    动态加载并执行Python模块中的函数
    """

    def __init__(self, metadata: SkillMetadata, skills_dir):
        """
        初始化脚本技能

        Args:
            metadata: 技能元数据
            skills_dir: 技能目录路径
        """
        super().__init__(metadata, llm_adapter=None)
        self.skills_dir = skills_dir
        self._module = None

    def _load_module(self):
        """懒加载模块"""
        if self._module is None:
            import importlib.util
            import sys

            # 构建脚本路径
            script_path = (
                self.skills_dir /
                f"phase{self.metadata.phase}-{self._get_phase_name()}" /
                self.metadata.name /
                "scripts" /
                f"{self.metadata.name.replace('-', '_')}.py"
            )

            if not script_path.exists():
                # 尝试主入口文件
                script_path = script_path.parent / "main.py"

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            # 动态加载
            spec = importlib.util.spec_from_file_location(
                f"skill_{self.metadata.name}",
                script_path
            )
            if spec and spec.loader:
                self._module = importlib.util.module_from_spec(spec)
                sys.modules[f"skill_{self.metadata.name}"] = self._module
                spec.loader.exec_module(self._module)

    def _get_phase_name(self) -> str:
        """获取阶段名称"""
        phase_names = {
            1: "input", 2: "analysis", 3: "modeling", 4: "validation",
            5: "writing", 6: "visualization", 7: "integration",
            8: "quality", 9: "optimization", 10: "submission"
        }
        return phase_names.get(self.phase, "unknown")

    async def execute(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SkillResult:
        """执行脚本技能"""
        import time
        start_time = time.time()

        # 验证输入
        valid, errors = self.validate_inputs(state)
        if not valid:
            return SkillResult(
                success=False,
                errors=errors,
                metadata={"validation_errors": errors}
            )

        try:
            self._load_module()

            # 查找执行函数
            handler_name = self.metadata.name.replace('-', '_')
            handler = getattr(self._module, handler_name, None)
            if handler is None:
                handler = getattr(self._module, 'main', None)
            if handler is None:
                handler = getattr(self._module, 'execute', None)

            if handler is None:
                raise AttributeError(
                    f"No handler function found in {self._module.__name__}. "
                    f"Expected: {handler_name}, main, or execute"
                )

            # 执行 (支持同步和异步)
            import asyncio
            if asyncio.iscoroutinefunction(handler):
                result = await handler(state, context)
            else:
                result = handler(state, context)

            execution_time = time.time() - start_time
            return SkillResult(
                success=True,
                data=result if isinstance(result, dict) else {"output": result},
                execution_time=execution_time
            )

        except Exception as e:
            logger.exception(f"Script skill {self.name} failed")
            return SkillResult.from_error(e, self.name)


class APISkill(BaseSkill):
    """
    基于外部API的技能

    调用外部API获取数据或执行操作
    """

    def __init__(
        self,
        metadata: SkillMetadata,
        api_config: Dict[str, Any]
    ):
        """
        初始化API技能

        Args:
            metadata: 技能元数据
            api_config: API配置 (url, method, headers等)
        """
        super().__init__(metadata, llm_adapter=None)
        self.api_config = api_config

    async def execute(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SkillResult:
        """执行API技能"""
        import time
        import httpx

        start_time = time.time()

        # 验证输入
        valid, errors = self.validate_inputs(state)
        if not valid:
            return SkillResult(
                success=False,
                errors=errors,
                metadata={"validation_errors": errors}
            )

        try:
            # 构建请求
            url = self._format_url(state)
            method = self.api_config.get('method', 'GET').upper()
            headers = self.api_config.get('headers', {})
            params = self._build_params(state)
            data = self._build_body(state)

            # 发送请求
            async with httpx.AsyncClient(timeout=self.metadata.timeout) as client:
                if method == 'GET':
                    response = await client.get(url, headers=headers, params=params)
                elif method == 'POST':
                    response = await client.post(url, headers=headers, json=data)
                else:
                    raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            result = response.json()

            execution_time = time.time() - start_time
            return SkillResult(
                success=True,
                data=result,
                execution_time=execution_time,
                metadata={
                    "url": url,
                    "status_code": response.status_code
                }
            )

        except httpx.HTTPError as e:
            return SkillResult.from_error(e, self.name)
        except Exception as e:
            logger.exception(f"API skill {self.name} failed")
            return SkillResult.from_error(e, self.name)

    def _format_url(self, state: Dict[str, Any]) -> str:
        """格式化URL"""
        url = self.api_config['url']
        for key, value in state.items():
            url = url.replace(f"{{{key}}}", str(value))
        return url

    def _build_params(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """构建查询参数"""
        params = {}
        for param in self.api_config.get('params', []):
            if param in state:
                params[param] = state[param]
        return params

    def _build_body(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """构建请求体"""
        body = {}
        for field in self.api_config.get('body', []):
            if field in state:
                body[field] = state[field]
        return body


class HybridSkill(BaseSkill):
    """
    混合技能

    结合LLM和脚本执行的复杂技能
    """

    def __init__(
        self,
        metadata: SkillMetadata,
        prompt_template: Optional[str] = None,
        script_path: Optional[str] = None,
        llm_adapter=None,
        skills_dir=None
    ):
        """
        初始化混合技能

        Args:
            metadata: 技能元数据
            prompt_template: 提示词模板 (可选)
            script_path: 脚本路径 (可选)
            llm_adapter: LLM适配器
            skills_dir: 技能目录
        """
        super().__init__(metadata, llm_adapter)
        self.prompt_template = prompt_template
        self.script_path = script_path
        self.skills_dir = skills_dir

        # 创建子技能
        self._llm_skill = None
        self._script_skill = None

        if prompt_template and llm_adapter:
            self._llm_skill = LLMSkill(metadata, prompt_template, llm_adapter)
        if script_path and skills_dir:
            self._script_skill = ScriptSkill(metadata, skills_dir)

    async def execute(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SkillResult:
        """执行混合技能"""
        import time
        start_time = time.time()

        # 验证输入
        valid, errors = self.validate_inputs(state)
        if not valid:
            return SkillResult(
                success=False,
                errors=errors,
                metadata={"validation_errors": errors}
            )

        results = {}

        try:
            # 先执行脚本 (如果存在)
            if self._script_skill:
                script_result = await self._script_skill.execute(state, context)
                if not script_result.success:
                    return script_result
                results.update(script_result.data)
                state.update(script_result.data)

            # 再执行LLM (如果存在)
            if self._llm_skill:
                llm_result = await self._llm_skill.execute(state, context)
                if not llm_result.success:
                    return llm_result
                results.update(llm_result.data)

            execution_time = time.time() - start_time
            return SkillResult(
                success=True,
                data=results,
                execution_time=execution_time,
                metadata={"components_executed": list(results.keys())}
            )

        except Exception as e:
            logger.exception(f"Hybrid skill {self.name} failed")
            return SkillResult.from_error(e, self.name)


def create_skill(
    metadata: SkillMetadata,
    llm_adapter=None,
    skills_dir=None,
    prompt_template: Optional[str] = None,
    api_config: Optional[Dict] = None
) -> BaseSkill:
    """
    根据元数据创建技能实例

    Args:
        metadata: 技能元数据
        llm_adapter: LLM适配器
        skills_dir: 技能目录
        prompt_template: 提示词模板 (LLM技能)
        api_config: API配置 (API技能)

    Returns:
        技能实例
    """
    mode = metadata.execution_mode

    if isinstance(mode, str):
        mode = ExecutionMode(mode)

    if mode == ExecutionMode.LLM:
        if prompt_template is None:
            raise ValueError("prompt_template required for LLM skills")
        return LLMSkill(metadata, prompt_template, llm_adapter)

    elif mode == ExecutionMode.SCRIPT:
        if skills_dir is None:
            raise ValueError("skills_dir required for script skills")
        return ScriptSkill(metadata, skills_dir)

    elif mode == ExecutionMode.API:
        if api_config is None:
            raise ValueError("api_config required for API skills")
        return APISkill(metadata, api_config)

    elif mode == ExecutionMode.HYBRID:
        return HybridSkill(
            metadata,
            prompt_template=prompt_template,
            script_path=str(skills_dir) if skills_dir else None,
            llm_adapter=llm_adapter,
            skills_dir=skills_dir
        )

    else:
        raise ValueError(f"Unknown execution mode: {mode}")
