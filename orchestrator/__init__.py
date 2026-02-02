"""
MCM/ICM Orchestrator Package
主编排器和相关组件
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .mcm_orchestrator import MCMOrchestrator, run_mcm_pipeline, PHASE_SKILLS
from .error_recovery import (
    ErrorRecoveryManager,
    RecoverableError,
    CriticalError,
    ErrorCategory,
    RecoveryStrategy
)
from .state_manager import StateManager
from .checkpoint_manager import CheckpointManager

# 新模块: 技能系统
from .base_skill import (
    BaseSkill,
    LLMSkill,
    ScriptSkill,
    APISkill,
    HybridSkill,
    SkillMetadata,
    SkillResult,
    ExecutionMode,
    create_skill
)
from .llm_adapter import LLMAdapter, MockLLMAdapter, create_llm_adapter
from .skill_registry import SkillRegistry, create_skill_registry
from .skill_executor import SkillExecutor, create_skill_executor
from .dependency_resolver import DependencyResolver, create_dependency_resolver
from .quality_checkers import (
    BaseChecker,
    CheckResult,
    create_checker
)
from .gate_executor import QualityGateExecutor, create_gate_executor, GateResult


@dataclass
class ExecutionResult:
    """
    标准化执行结果
    用于统一所有技能和阶段的返回值结构
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    phase: Optional[int] = None
    skill: Optional[str] = None
    checkpoint: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'phase': self.phase,
            'skill': self.skill,
            'checkpoint': self.checkpoint,
            'warnings': self.warnings,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """从字典创建"""
        return cls(
            success=data.get('success', False),
            data=data.get('data'),
            error=data.get('error'),
            phase=data.get('phase'),
            skill=data.get('skill'),
            checkpoint=data.get('checkpoint'),
            warnings=data.get('warnings', []),
            metadata=data.get('metadata', {})
        )


__all__ = [
    # 核心编排器
    'MCMOrchestrator',
    'run_mcm_pipeline',
    'PHASE_SKILLS',
    # 错误恢复
    'ErrorRecoveryManager',
    'RecoverableError',
    'CriticalError',
    'ErrorCategory',
    'RecoveryStrategy',
    # 状态管理
    'StateManager',
    'CheckpointManager',
    # 执行结果
    'ExecutionResult',
    # 技能基类
    'BaseSkill',
    'LLMSkill',
    'ScriptSkill',
    'APISkill',
    'HybridSkill',
    'SkillMetadata',
    'SkillResult',
    'ExecutionMode',
    'create_skill',
    # LLM适配器
    'LLMAdapter',
    'MockLLMAdapter',
    'create_llm_adapter',
    # 技能注册表
    'SkillRegistry',
    'create_skill_registry',
    # 技能执行器
    'SkillExecutor',
    'create_skill_executor',
    # 依赖解析器
    'DependencyResolver',
    'create_dependency_resolver',
    # 质量检查
    'BaseChecker',
    'CheckResult',
    'create_checker',
    # 门禁执行器
    'QualityGateExecutor',
    'create_gate_executor',
    'GateResult',
]
