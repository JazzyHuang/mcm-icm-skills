"""
技能执行器模块
提供统一的技能执行接口，支持超时控制、错误处理、并行执行
"""

import asyncio
import logging
import time
from copy import deepcopy
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from .base_skill import SkillResult, SkillMetadata
from .skill_registry import SkillRegistry
from .llm_adapter import LLMAdapter, get_max_tokens_for_skill
from .knowledge_injector import inject_knowledge

logger = logging.getLogger(__name__)


class SkillExecutionError(Exception):
    """技能执行错误"""
    def __init__(self, skill: str, message: str, cause: Optional[Exception] = None):
        self.skill = skill
        self.message = message
        self.cause = cause
        super().__init__(f"Skill '{skill}' failed: {message}")


class SkillTimeoutError(SkillExecutionError):
    """技能超时错误"""
    def __init__(self, skill: str, timeout: float):
        super().__init__(skill, f"Timeout after {timeout}s")


class SkillExecutor:
    """
    技能执行器

    负责:
    - 执行单个技能
    - 并行执行多个技能
    - 超时控制
    - 错误处理和重试
    - 执行统计
    """

    def __init__(
        self,
        registry: SkillRegistry,
        default_timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化技能执行器

        Args:
            registry: 技能注册表
            default_timeout: 默认超时时间(秒)
            max_retries: 最大重试次数
            retry_delay: 重试延迟(秒)
        """
        self.registry = registry
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 执行统计
        self.stats = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "retried": 0,
            "timeout": 0,
            "total_time": 0.0,
            "by_skill": {}
        }

    async def execute(
        self,
        skill_name: str,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> SkillResult:
        """
        执行单个技能

        Args:
            skill_name: 技能名称
            state: 全局状态
            context: 执行上下文
            timeout: 超时时间 (None使用默认)

        Returns:
            技能执行结果
        """
        start_time = time.time()
        context = context or {}
        timeout = timeout or self.default_timeout
        
        # 动态设置max_tokens（如果context中未指定）
        if 'max_tokens' not in context:
            context['max_tokens'] = get_max_tokens_for_skill(skill_name)
            logger.debug(f"Skill '{skill_name}' using max_tokens={context['max_tokens']}")
        
        # 注入知识库内容（如果context中未指定）
        if 'knowledge_injection' not in context:
            problem_type = state.get('problem_type')
            context['knowledge_injection'] = inject_knowledge(
                skill_name, 
                problem_type=problem_type,
                format_text=True
            )
            logger.debug(f"Skill '{skill_name}' knowledge injected for problem_type={problem_type}")

        # 获取技能
        try:
            skill = self.registry.get_skill(skill_name)
        except ValueError as e:
            logger.error(f"Skill not found: {skill_name}")
            return SkillResult.from_error(e, skill_name)

        # 获取技能特定的超时
        skill_timeout = getattr(skill.metadata, 'timeout', timeout)
        actual_timeout = min(timeout, skill_timeout)

        # 尝试执行 (带重试)
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = await self._execute_with_timeout(
                    skill,
                    state,
                    context,
                    actual_timeout
                )

                # 更新统计
                execution_time = time.time() - start_time
                self._update_stats(skill_name, execution_time, result.success, attempt > 0)

                return result

            except asyncio.TimeoutError:
                last_error = SkillTimeoutError(skill_name, actual_timeout)
                logger.warning(f"Skill {skill_name} timed out (attempt {attempt + 1})")

            except Exception as e:
                last_error = e
                logger.warning(f"Skill {skill_name} failed (attempt {attempt + 1}): {e}")

            # 重试逻辑
            if attempt < self.max_retries - 1:
                self.stats["retried"] += 1
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        # 所有重试都失败
        execution_time = time.time() - start_time
        self._update_stats(skill_name, execution_time, False, True)

        return SkillResult(
            success=False,
            errors=[str(last_error) if last_error else "Unknown error"],
            execution_time=execution_time,
            metadata={"attempts": self.max_retries}
        )

    async def _execute_with_timeout(
        self,
        skill,
        state: Dict[str, Any],
        context: Dict[str, Any],
        timeout: float
    ) -> SkillResult:
        """带超时执行技能"""
        # 使用状态的深拷贝，避免并发修改
        local_state = deepcopy(state)

        result = await asyncio.wait_for(
            skill.execute(local_state, context),
            timeout=timeout
        )

        return result

    async def execute_all(
        self,
        skill_names: List[str],
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        concurrency: int = 5
    ) -> Dict[str, SkillResult]:
        """
        并行执行多个技能

        Args:
            skill_names: 技能名称列表
            state: 全局状态
            context: 执行上下文
            timeout: 每个技能的超时时间
            concurrency: 最大并发数

        Returns:
            技能名称到结果的映射
        """
        if not skill_names:
            return {}

        # 创建信号量限制并发
        semaphore = asyncio.Semaphore(concurrency)

        async def execute_one(name: str) -> Tuple[str, SkillResult]:
            async with semaphore:
                result = await self.execute(name, state, context, timeout)
                return name, result

        # 并行执行所有技能
        tasks = [execute_one(name) for name in skill_names]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        return dict(results)

    async def execute_sequence(
        self,
        skill_names: List[str],
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        stop_on_error: bool = True
    ) -> Dict[str, SkillResult]:
        """
        顺序执行多个技能

        Args:
            skill_names: 技能名称列表 (按执行顺序)
            state: 全局状态
            context: 执行上下文
            stop_on_error: 遇到错误是否停止

        Returns:
            技能名称到结果的映射
        """
        results = {}
        context = context or {}

        for skill_name in skill_names:
            result = await self.execute(skill_name, state, context)
            results[skill_name] = result

            # 更新状态，使后续技能可访问
            if result.success:
                state.update(result.data)

            elif stop_on_error:
                logger.error(f"Stopping sequence due to error in {skill_name}")
                break

        return results

    async def execute_parallel_groups(
        self,
        groups: List[List[str]],
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, SkillResult]:
        """
        按组执行技能 (组内并行，组间顺序)

        Args:
            groups: 技能组列表 [[skill1, skill2], [skill3], ...]
            state: 全局状态
            context: 执行上下文

        Returns:
            所有技能的执行结果
        """
        all_results = {}
        context = context or {}

        for group in groups:
            # 并行执行组内技能
            group_results = await self.execute_all(group, state, context)

            # 更新结果和状态
            all_results.update(group_results)

            # 只有成功的技能才更新状态
            for skill_name, result in group_results.items():
                if result.success:
                    state.update(result.data)

            # 检查是否有严重错误
            failed = [name for name, result in group_results.items() if not result.success]
            if failed:
                logger.warning(f"Some skills in group failed: {failed}")

        return all_results

    async def execute_stream(
        self,
        skill_name: str,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        流式执行技能 (用于支持流式输出的技能)

        Args:
            skill_name: 技能名称
            state: 全局状态
            context: 执行上下文

        Yields:
            (事件类型, 数据) 元组
        """
        context = context or {}
        skill = self.registry.get_skill(skill_name)

        # 检查技能是否支持流式执行
        if not hasattr(skill, 'stream_execute'):
            # 不支持，回退到普通执行
            result = await self.execute(skill_name, state, context)
            yield ("result", result)
            return

        # 流式执行
        async for event, data in skill.stream_execute(state, context):
            yield event, data

    def _update_stats(
        self,
        skill_name: str,
        execution_time: float,
        success: bool,
        retried: bool
    ):
        """更新执行统计"""
        self.stats["total_executions"] += 1
        self.stats["total_time"] += execution_time

        if success:
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1

        if retried:
            self.stats["retried"] += 1

        # 按技能统计
        if skill_name not in self.stats["by_skill"]:
            self.stats["by_skill"][skill_name] = {
                "count": 0,
                "success": 0,
                "failed": 0,
                "total_time": 0.0
            }

        skill_stats = self.stats["by_skill"][skill_name]
        skill_stats["count"] += 1
        skill_stats["total_time"] += execution_time

        if success:
            skill_stats["success"] += 1
        else:
            skill_stats["failed"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        stats = self.stats.copy()
        if stats["total_executions"] > 0:
            stats["average_time"] = stats["total_time"] / stats["total_executions"]
            stats["success_rate"] = stats["successful"] / stats["total_executions"]
        return stats

    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "retried": 0,
            "timeout": 0,
            "total_time": 0.0,
            "by_skill": {}
        }

    def __repr__(self) -> str:
        return f"SkillExecutor(skills={len(self.registry)}, timeout={self.default_timeout})"


class ProgressTrackingExecutor(SkillExecutor):
    """
    支持进度追踪的技能执行器

    执行技能时调用进度回调，便于实时显示进度
    """

    def __init__(
        self,
        registry: SkillRegistry,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
        **kwargs
    ):
        """
        初始化进度追踪执行器

        Args:
            registry: 技能注册表
            progress_callback: 进度回调 (skill_name, status, progress)
            **kwargs: 其他参数传递给父类
        """
        super().__init__(registry, **kwargs)
        self.progress_callback = progress_callback
        self._current_phase = 0
        self._total_phases = 10
        self._phase_progress = {}  # phase -> (completed, total)

    def set_phase_count(self, total: int):
        """设置总阶段数"""
        self._total_phases = total

    def set_current_phase(self, phase: int):
        """设置当前阶段"""
        self._current_phase = phase

    def update_phase_progress(self, phase: int, completed: int, total: int):
        """更新阶段进度"""
        self._phase_progress[phase] = (completed, total)

    async def execute(
        self,
        skill_name: str,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> SkillResult:
        """执行技能并报告进度"""
        if self.progress_callback:
            self.progress_callback(skill_name, "started", 0.0)

        result = await super().execute(skill_name, state, context, timeout)

        if self.progress_callback:
            status = "success" if result.success else "failed"
            self.progress_callback(skill_name, status, 1.0)

        return result

    async def execute_parallel_groups(
        self,
        groups: List[List[str]],
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, SkillResult]:
        """按组执行并报告进度"""
        all_results = {}
        context = context or {}

        total_skills = sum(len(g) for g in groups)
        completed = 0

        for group in groups:
            # 报告组开始
            if self.progress_callback:
                group_name = ", ".join(group)
                self.progress_callback(group_name, "started", completed / total_skills)

            # 执行组
            group_results = await self.execute_all(group, state, context)
            all_results.update(group_results)

            # 更新状态
            for skill_name, result in group_results.items():
                if result.success:
                    state.update(result.data)
                completed += 1

                # 报告单个技能完成
                if self.progress_callback:
                    status = "success" if result.success else "failed"
                    self.progress_callback(skill_name, status, completed / total_skills)

        return all_results

    def get_overall_progress(self) -> float:
        """获取总体进度 (0-1)"""
        if not self._phase_progress:
            return 0.0

        total_completed = 0
        total_skills = 0

        for phase, (completed, total) in self._phase_progress.items():
            total_completed += completed
            total_skills += total

        if total_skills == 0:
            return 0.0

        return total_completed / total_skills


def create_skill_executor(
    registry: SkillRegistry,
    default_timeout: float = 300.0,
    progress_tracking: bool = False
) -> SkillExecutor:
    """
    创建技能执行器实例

    Args:
        registry: 技能注册表
        default_timeout: 默认超时时间
        progress_tracking: 是否启用进度追踪

    Returns:
        技能执行器实例
    """
    if progress_tracking:
        return ProgressTrackingExecutor(registry, default_timeout=default_timeout)
    return SkillExecutor(registry, default_timeout=default_timeout)
