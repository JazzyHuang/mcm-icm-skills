"""
错误恢复管理器
处理MCM/ICM流水线中的各类错误并执行恢复策略
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """错误分类"""
    EXECUTION = "execution"      # 执行级错误: API失败、连接超时
    SEMANTIC = "semantic"        # 语义级错误: 幻觉输出、错误查询
    STATE = "state"              # 状态级错误: 智能体信念与实际状态不一致
    CRITICAL = "critical"        # 严重错误: 需要人工干预


class RecoveryStrategy(Enum):
    """恢复策略"""
    RETRY = "retry"              # 重试
    FALLBACK = "fallback"        # 使用备选方案
    SKIP = "skip"                # 跳过
    HUMAN = "human"              # 人工干预
    ABORT = "abort"              # 终止


@dataclass
class ErrorConfig:
    """错误配置"""
    strategy: RecoveryStrategy
    max_retries: int = 3
    backoff_type: str = "exponential"  # exponential, linear, constant
    fallback_skill: Optional[str] = None
    message: Optional[str] = None


# 错误类型到恢复策略的映射
ERROR_RECOVERY_MAP = {
    # 可重试错误
    'NetworkError': ErrorConfig(
        strategy=RecoveryStrategy.RETRY,
        max_retries=3,
        backoff_type='exponential'
    ),
    'APIRateLimitError': ErrorConfig(
        strategy=RecoveryStrategy.RETRY,
        max_retries=5,
        backoff_type='exponential'
    ),
    'TimeoutError': ErrorConfig(
        strategy=RecoveryStrategy.RETRY,
        max_retries=2,
        backoff_type='linear'
    ),
    'ConnectionError': ErrorConfig(
        strategy=RecoveryStrategy.RETRY,
        max_retries=3,
        backoff_type='exponential'
    ),
    
    # 可降级错误
    'SolverInfeasible': ErrorConfig(
        strategy=RecoveryStrategy.FALLBACK,
        fallback_skill='heuristic_solver',
        message='优化问题无解，尝试启发式算法'
    ),
    'DataSourceUnavailable': ErrorConfig(
        strategy=RecoveryStrategy.FALLBACK,
        fallback_skill='alternative_data_source',
        message='主数据源不可用，切换备选数据源'
    ),
    'ModelConvergenceError': ErrorConfig(
        strategy=RecoveryStrategy.FALLBACK,
        fallback_skill='simplified_model',
        message='模型不收敛，使用简化模型'
    ),
    'LatexCompilationError': ErrorConfig(
        strategy=RecoveryStrategy.FALLBACK,
        fallback_skill='latex_error_fixer',
        message='LaTeX编译失败，尝试自动修复'
    ),
    
    # 可跳过错误
    'OptionalSkillError': ErrorConfig(
        strategy=RecoveryStrategy.SKIP,
        message='可选技能失败，跳过继续执行'
    ),
    
    # 需人工干预
    'ProblemParseError': ErrorConfig(
        strategy=RecoveryStrategy.HUMAN,
        message='无法解析题目，请手动提供结构化输入'
    ),
    'CriticalValidationError': ErrorConfig(
        strategy=RecoveryStrategy.HUMAN,
        message='发现严重质量问题，请检查'
    ),
    'HallucinationDetected': ErrorConfig(
        strategy=RecoveryStrategy.HUMAN,
        message='检测到内容幻觉，请人工审核'
    ),
}


# 幂等性分类
IDEMPOTENT_OPERATIONS = [
    'parse_problem',
    'search_literature',
    'analyze_sensitivity',
    'generate_chart',
    'compile_latex',
    'check_grammar',
    'validate_format',
]

NON_IDEMPOTENT_OPERATIONS = [
    'fetch_external_data',
    'call_paid_api',
    'write_output_file',
    'send_notification',
]


class RecoverableError(Exception):
    """可恢复的错误"""
    def __init__(self, message: str, error_type: str = 'GenericError', context: Optional[Dict] = None):
        super().__init__(message)
        self.error_type = error_type
        self.context = context or {}


class CriticalError(Exception):
    """严重错误，需要人工干预"""
    def __init__(self, message: str, error_type: str = 'CriticalError', context: Optional[Dict] = None):
        super().__init__(message)
        self.error_type = error_type
        self.context = context or {}


class ErrorRecoveryManager:
    """错误恢复管理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化错误恢复管理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.error_history = []
        self.recovery_attempts = {}
        
    def classify_error(self, error: Exception) -> ErrorCategory:
        """
        分类错误
        
        Args:
            error: 异常对象
            
        Returns:
            错误分类
        """
        error_type = type(error).__name__
        
        # 检查是否在映射表中
        if error_type in ERROR_RECOVERY_MAP:
            config = ERROR_RECOVERY_MAP[error_type]
            if config.strategy == RecoveryStrategy.HUMAN:
                return ErrorCategory.CRITICAL
            elif config.strategy in [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]:
                return ErrorCategory.EXECUTION
                
        # 根据错误特征分类
        error_msg = str(error).lower()
        
        if any(keyword in error_msg for keyword in ['network', 'connection', 'timeout', 'rate limit']):
            return ErrorCategory.EXECUTION
        elif any(keyword in error_msg for keyword in ['hallucination', 'invalid', 'incorrect']):
            return ErrorCategory.SEMANTIC
        elif any(keyword in error_msg for keyword in ['state', 'inconsistent', 'mismatch']):
            return ErrorCategory.STATE
        else:
            return ErrorCategory.CRITICAL
            
    def get_recovery_config(self, error: Exception) -> ErrorConfig:
        """
        获取错误的恢复配置
        
        Args:
            error: 异常对象
            
        Returns:
            恢复配置
        """
        error_type = type(error).__name__
        
        if error_type in ERROR_RECOVERY_MAP:
            return ERROR_RECOVERY_MAP[error_type]
            
        # 默认配置
        return ErrorConfig(
            strategy=RecoveryStrategy.RETRY,
            max_retries=2,
            backoff_type='exponential'
        )
        
    def is_idempotent(self, operation: str) -> bool:
        """
        检查操作是否幂等
        
        Args:
            operation: 操作名称
            
        Returns:
            是否幂等
        """
        return operation in IDEMPOTENT_OPERATIONS
        
    def calculate_backoff(self, attempt: int, backoff_type: str, base_delay: float = 2.0) -> float:
        """
        计算退避延迟
        
        Args:
            attempt: 当前尝试次数
            backoff_type: 退避类型
            base_delay: 基础延迟
            
        Returns:
            延迟秒数
        """
        if backoff_type == 'exponential':
            return base_delay ** attempt
        elif backoff_type == 'linear':
            return base_delay * attempt
        else:  # constant
            return base_delay
            
    async def handle_and_retry(
        self, 
        error: Exception, 
        state: Dict, 
        orchestrator: Any
    ) -> Dict:
        """
        处理错误并尝试恢复
        
        Args:
            error: 异常对象
            state: 当前状态
            orchestrator: 编排器实例
            
        Returns:
            恢复结果
        """
        error_type = type(error).__name__
        config = self.get_recovery_config(error)
        
        # 记录错误
        self.error_history.append({
            'error_type': error_type,
            'message': str(error),
            'phase': orchestrator.current_phase,
            'timestamp': state.get('timestamp')
        })
        
        # 根据策略处理
        if config.strategy == RecoveryStrategy.RETRY:
            return await self._handle_retry(error, state, orchestrator, config)
        elif config.strategy == RecoveryStrategy.FALLBACK:
            return await self._handle_fallback(error, state, orchestrator, config)
        elif config.strategy == RecoveryStrategy.SKIP:
            return self._handle_skip(error, state, config)
        elif config.strategy == RecoveryStrategy.HUMAN:
            return self.request_human_intervention(error, state)
        else:
            return self._handle_abort(error, state)
            
    async def _handle_retry(
        self, 
        error: Exception, 
        state: Dict, 
        orchestrator: Any,
        config: ErrorConfig
    ) -> Dict:
        """处理重试策略"""
        error_key = f"{orchestrator.current_phase}_{type(error).__name__}"
        
        # 获取当前重试次数
        current_attempts = self.recovery_attempts.get(error_key, 0)
        
        if current_attempts < config.max_retries:
            self.recovery_attempts[error_key] = current_attempts + 1
            delay = self.calculate_backoff(current_attempts, config.backoff_type)
            
            logger.info(f"Retrying after {delay}s (attempt {current_attempts + 1}/{config.max_retries})")
            await asyncio.sleep(delay)
            
            # 从当前阶段重新执行
            return await orchestrator.execute_pipeline(
                state.get('problem_input', {}),
                resume_from=orchestrator.current_phase
            )
        else:
            # 重试次数用尽
            return self.request_human_intervention(
                error, 
                state,
                message=f"重试{config.max_retries}次后仍然失败"
            )
            
    async def _handle_fallback(
        self, 
        error: Exception, 
        state: Dict, 
        orchestrator: Any,
        config: ErrorConfig
    ) -> Dict:
        """处理降级策略"""
        logger.info(f"Using fallback: {config.fallback_skill}")
        
        if config.message:
            logger.info(config.message)
            
        # 标记使用了降级方案
        state['fallback_used'] = state.get('fallback_used', [])
        state['fallback_used'].append({
            'original_error': str(error),
            'fallback_skill': config.fallback_skill,
            'phase': orchestrator.current_phase
        })
        
        # 继续执行
        return await orchestrator.execute_pipeline(
            state.get('problem_input', {}),
            resume_from=orchestrator.current_phase
        )
        
    def _handle_skip(self, error: Exception, state: Dict, config: ErrorConfig) -> Dict:
        """处理跳过策略"""
        logger.warning(f"Skipping due to error: {error}")
        
        if config.message:
            logger.info(config.message)
            
        state['skipped_skills'] = state.get('skipped_skills', [])
        state['skipped_skills'].append(str(error))
        
        return {
            'success': True,
            'warning': f"Some skills were skipped: {error}",
            'state': state
        }
        
    def _handle_abort(self, error: Exception, state: Dict) -> Dict:
        """处理终止策略"""
        logger.error(f"Aborting pipeline due to: {error}")
        
        return {
            'success': False,
            'error': str(error),
            'state': state,
            'error_history': self.error_history
        }
        
    def request_human_intervention(
        self, 
        error: Exception, 
        state: Dict,
        message: Optional[str] = None
    ) -> Dict:
        """
        请求人工干预
        
        Args:
            error: 异常对象
            state: 当前状态
            message: 附加消息
            
        Returns:
            包含干预请求信息的字典
        """
        config = self.get_recovery_config(error)
        intervention_message = message or config.message or "需要人工干预"
        
        logger.error(f"Human intervention required: {intervention_message}")
        
        return {
            'success': False,
            'requires_human_intervention': True,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'intervention_message': intervention_message,
            'state': state,
            'error_history': self.error_history,
            'suggestions': self._get_intervention_suggestions(error)
        }
        
    def _get_intervention_suggestions(self, error: Exception) -> list:
        """获取人工干预建议"""
        error_type = type(error).__name__
        
        suggestions = {
            'ProblemParseError': [
                "检查题目文本格式是否正确",
                "手动提取题目关键信息",
                "确认题目类型(A/B/C/D/E/F)"
            ],
            'CriticalValidationError': [
                "检查模型输出是否合理",
                "验证数据源是否可靠",
                "审核生成内容的准确性"
            ],
            'HallucinationDetected': [
                "核实所有引用的真实性",
                "检查数值计算是否正确",
                "验证方法描述的准确性"
            ],
        }
        
        return suggestions.get(error_type, [
            "检查错误日志获取详细信息",
            "从最近的检查点恢复执行",
            "手动修复问题后重新运行"
        ])
