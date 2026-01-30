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
    'MCMOrchestrator',
    'run_mcm_pipeline',
    'PHASE_SKILLS',
    'ErrorRecoveryManager',
    'RecoverableError',
    'CriticalError',
    'ErrorCategory',
    'RecoveryStrategy',
    'StateManager',
    'CheckpointManager',
    'ExecutionResult',
]
