"""
质量门禁执行器模块
执行质量门禁检查，管理回退逻辑
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .quality_checkers import (
    BaseChecker,
    CheckResult,
    CHECKER_REGISTRY,
    create_checker
)
from .llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """门禁检查结果"""
    phase: int                           # 阶段编号
    passed: bool                         # 是否通过
    checks: List[CheckResult] = field(default_factory=list)  # 检查结果列表
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    fallback_phase: Optional[int] = None  # 回退目标阶段
    retry_count: int = 0                 # 重试次数

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "phase": self.phase,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
            "fallback_phase": self.fallback_phase,
            "retry_count": self.retry_count
        }

    def get_failed_checks(self) -> List[CheckResult]:
        """获取失败的检查项"""
        return [c for c in self.checks if not c.passed]

    def get_critical_failures(self) -> List[CheckResult]:
        """获取关键失败项"""
        return [c for c in self.checks if not c.passed and c.details.get('priority') == 'critical']


class QualityGateExecutor:
    """
    质量门禁执行器

    负责:
    - 执行阶段门禁检查
    - 管理检查器实例
    - 确定回退策略
    - 生成检查报告
    """

    # 阶段回退映射
    FALLBACK_MAP = {
        5: 4,   # 写作阶段失败 -> 回退到验证阶段
        6: 5,   # 可视化失败 -> 回退到写作阶段
        8: 5,   # 质量检查失败 -> 回退到写作阶段
        9: 8,   # 优化失败 -> 回退到质量检查
    }

    def __init__(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        gate_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化门禁执行器

        Args:
            llm_adapter: LLM适配器
            gate_config: 门禁配置
        """
        self.llm = llm_adapter
        self.gate_config = gate_config or {}
        self.checkers: Dict[str, BaseChecker] = {}
        self.history: List[GateResult] = []

        # 初始化检查器
        self._init_checkers()

    def _init_checkers(self):
        """初始化所有检查器"""
        for name, checker_class in CHECKER_REGISTRY.items():
            # 某些检查器需要LLM
            if name in ['abstract_score', 'lat_score']:
                self.checkers[name] = checker_class(self.llm)
            else:
                self.checkers[name] = checker_class()

    async def check_phase_gates(
        self,
        phase: int,
        state: Dict[str, Any],
        gate_config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """
        检查阶段门禁

        Args:
            phase: 阶段编号
            state: 全局状态
            gate_config: 门禁配置 (可选，覆盖默认)

        Returns:
            门禁检查结果
        """
        config = gate_config or self.gate_config
        phase_key = f'phase{phase}_'

        # 查找该阶段的门禁配置
        phase_gates = None
        for key, value in config.items():
            if key.startswith(phase_key) and isinstance(value, dict) and 'gates' in value:
                phase_gates = value['gates']
                break

        # 如果没有配置，使用默认门禁
        if phase_gates is None:
            phase_gates = self._get_default_gates(phase)

        # 执行检查
        results = []
        all_passed = True
        critical_failures = []

        for gate_name, gate_criteria in phase_gates.items():
            gate_result = await self._check_single_gate(
                gate_name, gate_criteria, state
            )
            results.extend(gate_result)

            # 检查关键失败
            for check in gate_result:
                if not check.passed and gate_criteria.get('priority') in ['critical', 'high']:
                    all_passed = False
                    if gate_criteria.get('priority') == 'critical':
                        critical_failures.append(check)

        # 创建门禁结果
        gate_result = GateResult(
            phase=phase,
            passed=all_passed or len(critical_failures) == 0,
            checks=results
        )

        # 确定回退策略
        if not gate_result.passed:
            fallback_phase = self._determine_fallback(phase, critical_failures)
            gate_result.fallback_phase = fallback_phase

        # 记录历史
        self.history.append(gate_result)

        return gate_result

    async def _check_single_gate(
        self,
        gate_name: str,
        gate_criteria: Dict[str, Any],
        state: Dict[str, Any]
    ) -> List[CheckResult]:
        """检查单个门禁"""
        results = []
        criteria_list = gate_criteria.get('criteria', [])

        for criterion in criteria_list:
            checker_name = criterion.get('name')
            checker = self.checkers.get(checker_name)

            if checker is None:
                logger.warning(f"Checker not found: {checker_name}")
                continue

            try:
                result = await checker.check(state, criterion)
                # 添加优先级信息
                result.details['priority'] = gate_criteria.get('priority', 'medium')
                results.append(result)
            except Exception as e:
                logger.exception(f"Check failed: {checker_name}")
                results.append(CheckResult(
                    name=checker_name,
                    passed=False,
                    actual_value="error",
                    threshold=criterion.get('min', criterion.get('max')),
                    message=f"Check error: {str(e)}"
                ))

        return results

    def _get_default_gates(self, phase: int) -> Dict[str, Any]:
        """获取阶段默认门禁"""
        # 基于阶段定义关键门禁
        if phase == 5:  # 写作阶段
            return {
                "abstract_quality": {
                    "priority": "critical",
                    "criteria": [
                        {"name": "abstract_score", "min": 0.85},
                        {"name": "hook_quality", "min": 0.80},
                        {"name": "quantification_density", "min": 0.75}
                    ]
                }
            }
        elif phase == 8:  # 质量检查阶段
            return {
                "language_quality": {
                    "priority": "high",
                    "criteria": [
                        {"name": "lat_score", "min": 7.5},
                        {"name": "chinglish_score", "max": 0.20}
                    ]
                },
                "consistency": {
                    "priority": "medium",
                    "criteria": [
                        {"name": "consistency_score", "min": 0.90}
                    ]
                },
                "hallucination": {
                    "priority": "critical",
                    "criteria": [
                        {"name": "hallucination_count", "max": 0}
                    ]
                }
            }
        else:
            return {}  # 其他阶段无默认门禁

    def _determine_fallback(
        self,
        phase: int,
        critical_failures: List[CheckResult]
    ) -> Optional[int]:
        """确定回退目标阶段"""
        # 检查失败类型确定回退策略
        for failure in critical_failures:
            if failure.name in ['abstract_score', 'hook_quality']:
                return 4  # 回退到验证阶段获取更多内容
            elif failure.name in ['chinglish_score', 'lat_score']:
                return 5  # 回退到写作阶段重写
            elif failure.name == 'consistency_score':
                return 6  # 回退到可视化阶段
            elif failure.name == 'hallucination_count':
                return 1  # 回退到输入阶段重新验证引用

        # 使用预定义的回退映射
        return self.FALLBACK_MAP.get(phase)

    async def check_all_gates(
        self,
        state: Dict[str, Any],
        phases: Optional[List[int]] = None
    ) -> Dict[int, GateResult]:
        """
        检查多个阶段的门禁

        Args:
            state: 全局状态
            phases: 要检查的阶段列表 (None检查所有)

        Returns:
            阶段到门禁结果的映射
        """
        phases = phases or list(range(1, 11))
        results = {}

        for phase in phases:
            result = await self.check_phase_gates(phase, state)
            results[phase] = result

        return results

    def generate_report(self) -> Dict[str, Any]:
        """生成门禁检查报告"""
        total_phases = len(self.history)
        passed_phases = sum(1 for r in self.history if r.passed)

        # 统计检查项
        total_checks = sum(len(r.checks) for r in self.history)
        passed_checks = sum(
            sum(1 for c in r.checks if c.passed)
            for r in self.history
        )

        # 收集失败项
        all_failures = []
        for result in self.history:
            for check in result.get_failed_checks():
                all_failures.append({
                    "phase": result.phase,
                    "check": check.name,
                    "message": check.message
                })

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_phases": total_phases,
                "passed_phases": passed_phases,
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "success_rate": passed_checks / total_checks if total_checks > 0 else 1.0
            },
            "failures": all_failures,
            "history": [r.to_dict() for r in self.history]
        }

    def get_failed_phase_gates(self) -> List[int]:
        """获取失败的阶段列表"""
        return [r.phase for r in self.history if not r.passed]

    def clear_history(self):
        """清除检查历史"""
        self.history.clear()

    def register_checker(self, name: str, checker: BaseChecker):
        """注册自定义检查器"""
        self.checkers[name] = checker

    def load_config(self, config: Dict[str, Any]):
        """加载门禁配置"""
        self.gate_config = config

    def get_phase_gate_config(self, phase: int) -> Optional[Dict[str, Any]]:
        """获取指定阶段的门禁配置"""
        phase_key = f'phase{phase}_'
        for key, value in self.gate_config.items():
            if key.startswith(phase_key):
                return value
        return None

    def __repr__(self) -> str:
        return f"QualityGateExecutor(checkers={len(self.checkers)}, history={len(self.history)})"


class AdaptiveGateExecutor(QualityGateExecutor):
    """
    自适应门禁执行器

    根据历史表现动态调整门禁标准
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_retry_counts: Dict[int, int] = {}

    async def check_phase_gates(
        self,
        phase: int,
        state: Dict[str, Any],
        gate_config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """检查阶段门禁 (自适应)"""
        # 检查重试次数
        retry_count = self.phase_retry_counts.get(phase, 0)

        # 如果重试多次，可以考虑放宽标准
        if retry_count >= 2:
            gate_config = self._relax_standards(gate_config or self.gate_config, phase)

        result = await super().check_phase_gates(phase, state, gate_config)

        # 更新重试计数
        if not result.passed:
            self.phase_retry_counts[phase] = retry_count + 1
        else:
            # 成功后清除计数
            self.phase_retry_counts.pop(phase, 0)

        result.retry_count = self.phase_retry_counts.get(phase, 0)
        return result

    def _relax_standards(
        self,
        config: Dict[str, Any],
        phase: int
    ) -> Dict[str, Any]:
        """放宽门禁标准"""
        relaxed = {}
        for key, value in config.items():
            if key.startswith(f'phase{phase}_'):
                relaxed_value = value.copy()
                if 'gates' in relaxed_value:
                    for gate_name, gate_criteria in relaxed_value['gates'].items():
                        if 'criteria' in gate_criteria:
                            for criterion in gate_criteria['criteria']:
                                # 降低10%的标准
                                if 'min' in criterion:
                                    criterion['min'] *= 0.9
                                if 'max' in criterion:
                                    criterion['max'] *= 1.1
                relaxed[key] = relaxed_value
            else:
                relaxed[key] = value
        return relaxed


def create_gate_executor(
    llm_adapter: Optional[LLMAdapter] = None,
    adaptive: bool = False
) -> QualityGateExecutor:
    """
    创建门禁执行器实例

    Args:
        llm_adapter: LLM适配器
        adaptive: 是否使用自适应模式

    Returns:
        门禁执行器实例
    """
    if adaptive:
        return AdaptiveGateExecutor(llm_adapter)
    return QualityGateExecutor(llm_adapter)
