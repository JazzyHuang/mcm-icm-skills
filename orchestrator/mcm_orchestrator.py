"""
MCM/ICM 主编排器
基于AgentOrchestra架构的层级式多智能体编排器
增强版: 包含质量门禁和阶段回退逻辑，支持真正的技能执行
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .error_recovery import ErrorRecoveryManager, RecoverableError, CriticalError
from .state_manager import StateManager
from .checkpoint_manager import CheckpointManager

# 新导入: 技能系统组件
from .base_skill import SkillResult
from .llm_adapter import LLMAdapter, create_llm_adapter
from .skill_registry import SkillRegistry, create_skill_registry
from .skill_executor import SkillExecutor, create_skill_executor
from .dependency_resolver import DependencyResolver, create_dependency_resolver
from .gate_executor import QualityGateExecutor, create_gate_executor, GateResult
from .planning_agent import PlanningAgent, create_planning_agent

logger = logging.getLogger(__name__)


# 阶段技能映射 (增强版 v2 - 包含高级算法和深度搜索)
PHASE_SKILLS = {
    1: ['problem-parser', 'problem-type-classifier', 'problem-reference-extractor',
        'data-collector', 'deep-reference-searcher', 'literature-searcher',
        'ai-deep-search-guide',  # 新增: 确保深度搜索和引用多样性
        'citation-validator', 'citation-diversity-validator'],
    2: ['problem-decomposer', 'sub-problem-analyzer', 'assumption-generator',
        'variable-definer', 'constraint-identifier'],
    3: [
        # 基础建模流程
        'model-selector', 'model-justification-generator', 'hybrid-model-designer',
        'model-builder', 'model-solver', 'code-verifier',
        # 高级算法 (新增: 前沿创新方法)
        'physics-informed-nn',      # PINN物理信息神经网络
        'neural-operators',         # FNO/DeepONet神经算子
        'transformer-forecasting',  # Transformer时间序列预测
        'reinforcement-learning',   # 强化学习
        'kan-networks',             # KAN网络 (2025 ICLR前沿方法)
        'causal-inference'          # 因果推断
    ],
    4: ['sensitivity-analyzer', 'uncertainty-quantifier', 'model-validator',
        'error-analyzer', 'limitation-analyzer', 'strengths-weaknesses', 
        'ethical-analyzer', 'model-explainer'],
    5: ['section-writer', 'section-iterative-optimizer',  # 新增: 章节迭代优化
        'fact-checker', 'abstract-first-impression',
        'abstract-generator', 'abstract-iterative-optimizer', 'memo-letter-writer'],
    6: ['chart-generator', 'figure-narrative-generator', 'publication-scaler',
        'table-formatter', 'figure-validator', 
        'infographic-generator'],  # 新增: 信息图生成器
    7: ['latex-compiler', 'compilation-error-handler', 'citation-manager',
        'format-checker', 'anonymization-checker'],
    8: ['quality-reviewer', 'hallucination-detector', 'grammar-checker',
        'chinglish-detector', 'consistency-checker', 'global-consistency-checker'],
    9: ['final-polisher', 'academic-english-optimizer', 'submission-preparer'],
    10: ['pre-submission-validator', 'submission-checklist']
}

# 可并行执行的技能组
PARALLEL_SKILLS = {
    1: [['problem-parser', 'problem-type-classifier'], 
        ['data-collector', 'deep-reference-searcher', 'literature-searcher']],
    4: [['sensitivity-analyzer', 'uncertainty-quantifier'], 
        ['error-analyzer', 'limitation-analyzer']],
    8: [['grammar-checker', 'chinglish-detector'], 
        ['consistency-checker', 'global-consistency-checker']],
}

# 备选技能映射
FALLBACK_SKILLS = {
    'model-solver': 'model-solver-fallback',
    'data-collector': 'data-collector-fallback',
    'latex-compiler': 'latex-compiler-fallback',
    'deep-reference-searcher': 'literature-searcher',
}

# 阶段回退映射
PHASE_FALLBACK_MAP = {
    # 当phase X的门禁失败时，回退到phase Y
    5: 4,  # 写作阶段失败 -> 回退到验证阶段
    6: 5,  # 可视化失败 -> 回退到写作阶段
    8: 5,  # 质量检查失败 -> 回退到写作阶段
    9: 8,  # 优化失败 -> 回退到质量检查
}


class QualityGateChecker:
    """质量门禁检查器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化质量门禁检查器
        
        Args:
            config_path: 质量门禁配置文件路径
        """
        self.gates_config = self._load_gates_config(config_path)
        self.check_results = {}
        
    def _load_gates_config(self, config_path: Optional[str] = None) -> Dict:
        """加载质量门禁配置"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'quality_gates.yaml'
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Quality gates config not found: {config_path}")
            return self._get_default_gates()
    
    def _get_default_gates(self) -> Dict:
        """返回默认门禁配置"""
        return {
            'global': {
                'enabled': True,
                'mode': 'strict',
                'failure_strategy': 'retry',
                'max_retries': 3
            }
        }
    
    async def check_phase_gates(self, phase: int, state: Dict, results: Dict) -> Tuple[bool, Dict]:
        """
        检查阶段门禁
        
        Args:
            phase: 阶段编号
            state: 当前状态
            results: 阶段执行结果
            
        Returns:
            (是否通过, 检查详情)
        """
        phase_key = f'phase{phase}_'
        gate_results = {
            'phase': phase,
            'passed': True,
            'gates_checked': [],
            'failures': [],
            'warnings': []
        }
        
        # 如果门禁未启用，直接通过
        if not self.gates_config.get('global', {}).get('enabled', True):
            return True, gate_results
        
        # 查找该阶段的门禁配置
        phase_gates = None
        for key, value in self.gates_config.items():
            if key.startswith(phase_key):
                phase_gates = value
                break
        
        if not phase_gates:
            logger.debug(f"No gates configured for phase {phase}")
            return True, gate_results
        
        # 检查每个门禁
        for gate_name, gate_config in phase_gates.get('gates', {}).items():
            gate_check = await self._check_single_gate(gate_name, gate_config, state, results)
            gate_results['gates_checked'].append(gate_check)
            
            if not gate_check['passed']:
                if gate_config.get('priority') == 'critical':
                    gate_results['passed'] = False
                    gate_results['failures'].append(gate_check)
                elif gate_config.get('priority') == 'high':
                    # 高优先级门禁失败也导致整体失败
                    gate_results['passed'] = False
                    gate_results['failures'].append(gate_check)
                else:
                    gate_results['warnings'].append(gate_check)
        
        self.check_results[phase] = gate_results
        return gate_results['passed'], gate_results
    
    async def _check_single_gate(
        self, 
        gate_name: str, 
        gate_config: Dict, 
        state: Dict, 
        results: Dict
    ) -> Dict:
        """检查单个门禁"""
        gate_check = {
            'name': gate_name,
            'description': gate_config.get('description', ''),
            'priority': gate_config.get('priority', 'medium'),
            'passed': True,
            'criteria_results': []
        }
        
        for criterion in gate_config.get('criteria', []):
            criterion_result = self._evaluate_criterion(criterion, state, results)
            gate_check['criteria_results'].append(criterion_result)
            
            if not criterion_result['passed']:
                if criterion.get('required', False):
                    gate_check['passed'] = False
        
        return gate_check
    
    def _evaluate_criterion(self, criterion: Dict, state: Dict, results: Dict) -> Dict:
        """评估单个标准"""
        criterion_result = {
            'name': criterion.get('name', 'unnamed'),
            'description': criterion.get('description', ''),
            'passed': True,
            'actual_value': None,
            'expected': None,
            'message': ''
        }
        
        # 获取实际值（从state或results中查找）
        # 使用 None 检查而非 or，避免 0 等假值被错误跳过
        value_key = criterion.get('name')
        actual_value = results.get(value_key)
        if actual_value is None:
            actual_value = state.get(value_key)
        criterion_result['actual_value'] = actual_value
        
        # 检查各种条件
        if 'required' in criterion and criterion['required']:
            if actual_value is None:
                criterion_result['passed'] = False
                criterion_result['message'] = f"Required value '{value_key}' not found"
                return criterion_result
        
        if 'min' in criterion:
            criterion_result['expected'] = f">= {criterion['min']}"
            if actual_value is not None and actual_value < criterion['min']:
                criterion_result['passed'] = False
                criterion_result['message'] = f"Value {actual_value} below minimum {criterion['min']}"
        
        if 'max' in criterion:
            criterion_result['expected'] = f"<= {criterion['max']}"
            if actual_value is not None and actual_value > criterion['max']:
                criterion_result['passed'] = False
                criterion_result['message'] = f"Value {actual_value} above maximum {criterion['max']}"
        
        if criterion_result['passed']:
            criterion_result['message'] = 'Passed'
        
        return criterion_result
    
    def get_fallback_phase(self, failed_phase: int) -> Optional[int]:
        """获取失败阶段应回退到的阶段"""
        return PHASE_FALLBACK_MAP.get(failed_phase)
    
    def generate_report(self) -> Dict:
        """生成质量门禁检查报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'passed',
            'phases_checked': len(self.check_results),
            'phases_passed': 0,
            'phases_failed': 0,
            'details': self.check_results
        }
        
        for phase, result in self.check_results.items():
            if result['passed']:
                report['phases_passed'] += 1
            else:
                report['phases_failed'] += 1
                report['overall_status'] = 'failed'
        
        return report


class MCMOrchestrator:
    """
    层级式多智能体编排器
    基于AgentOrchestra架构设计
    增强版: 包含质量门禁和阶段回退逻辑、并发安全、依赖注入、真正的技能执行
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        state_manager: Optional[StateManager] = None,
        error_handler: Optional[ErrorRecoveryManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        quality_checker: Optional[QualityGateChecker] = None,
        config: Optional[Dict] = None,
        # 新增参数
        llm_adapter: Optional[LLMAdapter] = None,
        skill_registry: Optional[SkillRegistry] = None,
        skill_executor: Optional[SkillExecutor] = None,
        dependency_resolver: Optional[DependencyResolver] = None,
        gate_executor: Optional[QualityGateExecutor] = None
    ):
        """
        初始化编排器（支持依赖注入）

        Args:
            config_path: 配置文件路径，默认使用 config/settings.yaml
            state_manager: 状态管理器实例（可选，用于依赖注入）
            error_handler: 错误处理器实例（可选，用于依赖注入）
            checkpoint_manager: 检查点管理器实例（可选，用于依赖注入）
            quality_checker: 质量检查器实例（可选，用于依赖注入）
            config: 配置字典（可选，优先于config_path）
            llm_adapter: LLM适配器（可选，用于依赖注入）
            skill_registry: 技能注册表（可选，用于依赖注入）
            skill_executor: 技能执行器（可选，用于依赖注入）
            dependency_resolver: 依赖解析器（可选，用于依赖注入）
            gate_executor: 门禁执行器（可选，用于依赖注入）
        """
        # 配置加载（支持直接传入配置字典）
        self.config = config if config is not None else self._load_config(config_path)

        # 确定根目录
        self.root_dir = Path(__file__).parent.parent

        # 依赖注入：如果未提供则创建默认实例
        self.state_manager = state_manager or StateManager()
        self.error_handler = error_handler or ErrorRecoveryManager(self.config)
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            checkpoint_dir=Path(self.config.get('output', {}).get('checkpoint_dir', 'output/checkpoints'))
        )
        self.quality_checker = quality_checker or QualityGateChecker()

        # === 新增: 初始化技能系统组件 ===
        # LLM适配器
        if llm_adapter is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            model = self.config.get('llm', {}).get('model', 'smart')
            mock_mode = self.config.get('llm', {}).get('mock', False)
            self.llm_adapter = create_llm_adapter(api_key=api_key, model=model, mock=mock_mode)
        else:
            self.llm_adapter = llm_adapter

        # 技能注册表
        if skill_registry is None:
            skills_dir = self.root_dir / 'skills'
            self.skill_registry = create_skill_registry(skills_dir, self.llm_adapter)
        else:
            self.skill_registry = skill_registry

        # 技能执行器
        if skill_executor is None:
            default_timeout = self.config.get('execution', {}).get('timeout_per_skill', 300)
            self.skill_executor = create_skill_executor(self.skill_registry, default_timeout=default_timeout)
        else:
            self.skill_executor = skill_executor

        # 依赖解析器
        if dependency_resolver is None:
            self.dependency_resolver = create_dependency_resolver(self.skill_registry)
        else:
            self.dependency_resolver = dependency_resolver

        # 门禁执行器
        if gate_executor is None:
            gate_config = self._load_gate_config()
            self.gate_executor = create_gate_executor(self.llm_adapter)
            self.gate_executor.load_config(gate_config)
        else:
            self.gate_executor = gate_executor

        # 规划Agent（新增）
        self.planning_agent = create_planning_agent(self.llm_adapter)
        
        # 执行状态
        self.current_phase = 0
        self.execution_log: List[Dict[str, Any]] = []
        self.fallback_count: Dict[int, int] = {}  # 记录每个阶段的回退次数
        self._state_lock = asyncio.Lock()  # 状态更新锁，防止并发竞态
        self._execution_id: Optional[str] = None  # 当前执行ID
        self.modeling_plan: Optional[Dict] = None  # 建模计划
        
        # 启动时验证skill注册
        self._validate_skill_registration()

        logger.info(f"MCMOrchestrator initialized with {len(self.skill_registry)} skills")

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """加载配置文件"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()

    def _load_gate_config(self) -> Dict:
        """加载质量门禁配置"""
        gate_config_path = Path(__file__).parent.parent / 'config' / 'quality_gates.yaml'

        try:
            with open(gate_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Gate config not found: {gate_config_path}")
            return {}
            
    def _get_default_config(self) -> Dict:
        """返回默认配置"""
        return {
            'execution': {
                'max_retries': 3,
                'retry_delay_base': 2,
                'timeout_per_skill': 300,
                'checkpoint_interval': 60,
                'parallel_skills': True,
                'lazy_loading': True,
                'quality_gates_enabled': True,
                'max_phase_fallbacks': 2,
            },
            'quality_thresholds': {
                'abstract_min_iterations': 12,
                'abstract_min_score': 0.85,
                'grammar_score_min': 7.5,
                'chinglish_score_max': 0.20,
                'hallucination_tolerance': 0.0,
                'consistency_score_min': 0.90,
                'citation_verification': True,
            },
            'output': {
                'checkpoint_dir': 'output/checkpoints',
                'papers_dir': 'output/papers',
                'figures_dir': 'output/figures',
                'quality_reports_dir': 'output/quality_reports',
            }
        }

    # ============ 启动验证和增强日志（新增） ============
    
    def _validate_skill_registration(self):
        """
        验证所有预期skills是否已注册
        
        在启动时检查PHASE_SKILLS中定义的所有技能是否都已在registry中注册。
        记录缺失的技能，但不阻止启动（允许部分功能运行）。
        """
        missing = {}
        total_expected = 0
        total_registered = 0
        
        for phase, skills in PHASE_SKILLS.items():
            missing_in_phase = []
            for skill in skills:
                total_expected += 1
                if self.skill_registry.has_skill(skill):
                    total_registered += 1
                else:
                    missing_in_phase.append(skill)
            
            if missing_in_phase:
                missing[phase] = missing_in_phase
                logger.warning(f"Phase {phase} missing skills: {missing_in_phase}")
        
        # 记录验证结果
        self._skill_validation_result = {
            'total_expected': total_expected,
            'total_registered': total_registered,
            'missing_by_phase': missing,
            'registration_rate': total_registered / total_expected if total_expected > 0 else 0
        }
        
        if missing:
            logger.warning(
                f"Skill registration validation: {total_registered}/{total_expected} skills registered "
                f"({self._skill_validation_result['registration_rate']:.1%})"
            )
            logger.warning(f"Missing skills summary: {missing}")
        else:
            logger.info(f"All {total_expected} expected skills are registered")
    
    def _log_phase_execution_details(
        self, 
        phase: int, 
        phase_results: Dict, 
        phase_duration: float,
        gate_results: Optional[Dict] = None
    ) -> Dict:
        """
        记录阶段执行的详细日志
        
        Args:
            phase: 阶段编号
            phase_results: 阶段执行结果
            phase_duration: 执行耗时（秒）
            gate_results: 门禁检查结果
            
        Returns:
            详细的执行日志记录
        """
        expected_skills = PHASE_SKILLS.get(phase, [])
        registered_skills = self.skill_registry.get_phase_skills(phase)
        executed_skills = list(phase_results.keys())
        
        # 计算各类技能
        successful_skills = [
            s for s, r in phase_results.items() 
            if isinstance(r, dict) and r.get('status') != 'failed' or not isinstance(r, dict)
        ]
        failed_skills = [
            s for s, r in phase_results.items() 
            if isinstance(r, dict) and r.get('status') == 'failed'
        ]
        skipped_skills = [s for s in expected_skills if s not in executed_skills]
        
        log_entry = {
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': round(phase_duration, 2),
            'skills': {
                'expected': expected_skills,
                'expected_count': len(expected_skills),
                'registered': registered_skills,
                'registered_count': len(registered_skills),
                'executed': executed_skills,
                'executed_count': len(executed_skills),
                'successful': successful_skills,
                'successful_count': len(successful_skills),
                'failed': failed_skills,
                'failed_count': len(failed_skills),
                'skipped': skipped_skills,
                'skipped_count': len(skipped_skills),
            },
            'execution_rate': len(executed_skills) / len(expected_skills) if expected_skills else 1.0,
            'success_rate': len(successful_skills) / len(executed_skills) if executed_skills else 0,
            'gate_results': gate_results,
            'status': 'success' if not failed_skills and (not gate_results or gate_results.get('passed', True)) else 'partial_failure'
        }
        
        # 记录详细日志
        logger.info(
            f"Phase {phase} execution summary: "
            f"{len(executed_skills)}/{len(expected_skills)} executed, "
            f"{len(successful_skills)} successful, "
            f"{len(failed_skills)} failed, "
            f"{len(skipped_skills)} skipped"
        )
        
        if failed_skills:
            logger.warning(f"Phase {phase} failed skills: {failed_skills}")
        
        if skipped_skills:
            logger.info(f"Phase {phase} skipped skills: {skipped_skills}")
        
        return log_entry
        
    async def execute_pipeline(
        self, 
        problem_input: Dict[str, Any],
        resume_from: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行完整流水线
        
        Args:
            problem_input: 问题输入
            resume_from: 从指定阶段恢复执行
            
        Returns:
            包含生成论文路径和执行日志的字典
        """
        start_time = datetime.now()
        logger.info(f"Starting MCM/ICM pipeline at {start_time}")
        
        # 初始化或恢复状态
        if resume_from:
            state = self.checkpoint_manager.load(f'phase{resume_from - 1}')
            if state is None:
                raise ValueError(f"Checkpoint for phase {resume_from - 1} not found")
            start_phase = resume_from
            logger.info(f"Resuming from phase {resume_from}")
        else:
            state = self.state_manager.initialize(problem_input)
            start_phase = 1
            
        try:
            phase = start_phase
            while phase <= 10:
                self.current_phase = phase
                logger.info(f"=== Starting Phase {phase} ===")
                
                phase_start = datetime.now()
                phase_results = await self._execute_phase(phase, state)
                phase_duration = (datetime.now() - phase_start).total_seconds()
                
                # 更新状态
                state.update(phase_results)
                
                # Phase 2完成后，执行规划Agent创建建模计划
                if phase == 2:
                    logger.info("Planning Agent: Creating modeling plan before Phase 3...")
                    problem_analysis = {
                        'sub_problems': state.get('sub_problems', []),
                        'assumptions': state.get('assumptions', []),
                        'variables': state.get('variables', {}),
                        'constraints': state.get('constraints', [])
                    }
                    self.modeling_plan = await self.planning_agent.create_modeling_plan(
                        problem_analysis, state
                    )
                    state['modeling_plan'] = self.modeling_plan
                    logger.info(f"Planning Agent: Plan created with skills: {self.modeling_plan.get('skills_to_execute', [])}")
                
                # Phase 5完成后，执行章节内容验证和自动扩展
                if phase == 5:
                    validation_report = self.state_manager.validate_all_sections()
                    logger.info(f"Section validation: {validation_report['sections_meeting_minimum']}/"
                                f"{validation_report['total_sections']} sections meet minimum, "
                                f"total {validation_report['total_word_count']} words")
                    
                    if validation_report['sections_needing_expansion']:
                        logger.info("Starting auto-expansion for short sections...")
                        state = await self._expand_short_sections(state)
                
                # 质量门禁检查
                if self.config.get('execution', {}).get('quality_gates_enabled', True):
                    gates_passed, gate_results = await self.quality_checker.check_phase_gates(
                        phase, state, phase_results
                    )
                    
                    if not gates_passed:
                        logger.warning(f"Phase {phase} failed quality gates")
                        
                        # 尝试回退
                        fallback_phase = await self._handle_gate_failure(phase, gate_results, state)
                        
                    if fallback_phase:
                        logger.info(f"Falling back to phase {fallback_phase}")
                        phase = fallback_phase
                        continue
                    else:
                        # 无法回退，记录失败并继续（或暂停）
                        logger.error(f"Phase {phase} failed and cannot fallback")
                        # 使用增强的日志记录
                        log_entry = self._log_phase_execution_details(
                            phase, phase_results, phase_duration, gate_results
                        )
                        log_entry['status'] = 'failed'
                        self.execution_log.append(log_entry)
                        
                        # 根据配置决定是继续还是停止
                        if self.config.get('execution', {}).get('stop_on_gate_failure', False):
                            return self._create_failure_response(phase, gate_results, state)
                        # 不停止时，跳过成功处理逻辑，继续下一阶段
                        phase += 1
                        continue
                
                # 门禁通过或未启用门禁
                state['completed_phases'].append(phase)
                
                # 保存检查点
                self.checkpoint_manager.save(state, f'phase{phase}')
                
                # 使用增强的日志记录
                log_entry = self._log_phase_execution_details(
                    phase, phase_results, phase_duration, 
                    gate_results if 'gate_results' in dir() else None
                )
                self.execution_log.append(log_entry)
                
                logger.info(f"Phase {phase} completed in {phase_duration:.2f}s")
                phase += 1
                
            # 生成质量报告
            quality_report = self.quality_checker.generate_report()
            self._save_quality_report(quality_report)
            
            # 获取最终输出
            final_output = self.state_manager.get_final_output(state)
            
            total_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Pipeline completed in {total_duration:.2f}s")
            
            return {
                'success': True,
                'output': final_output,
                'execution_log': self.execution_log,
                'quality_report': quality_report,
                'total_duration_seconds': total_duration
            }
            
        except RecoverableError as e:
            logger.warning(f"Recoverable error in phase {self.current_phase}: {e}")
            return await self.error_handler.handle_and_retry(e, state, self)
            
        except CriticalError as e:
            logger.error(f"Critical error in phase {self.current_phase}: {e}")
            return self.error_handler.request_human_intervention(e, state)
            
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return {
                'success': False,
                'error': str(e),
                'phase': self.current_phase,
                'execution_log': self.execution_log,
                'checkpoint': f'phase{self.current_phase - 1}' if self.current_phase > 1 else None
            }
    
    async def _handle_gate_failure(
        self, 
        phase: int, 
        gate_results: Dict, 
        state: Dict
    ) -> Optional[int]:
        """
        处理门禁失败
        
        Args:
            phase: 失败的阶段
            gate_results: 门禁检查结果
            state: 当前状态
            
        Returns:
            回退到的阶段，如果无法回退则返回None
        """
        max_fallbacks = self.config.get('execution', {}).get('max_phase_fallbacks', 2)
        
        # 检查是否已经回退过太多次
        if self.fallback_count.get(phase, 0) >= max_fallbacks:
            logger.warning(f"Phase {phase} has reached max fallback count ({max_fallbacks})")
            return None
        
        # 获取回退目标阶段
        fallback_phase = self.quality_checker.get_fallback_phase(phase)
        
        if fallback_phase is None:
            logger.warning(f"No fallback configured for phase {phase}")
            return None
        
        # 更新回退计数
        self.fallback_count[phase] = self.fallback_count.get(phase, 0) + 1
        
        # 清理需要重做的阶段状态
        for p in range(fallback_phase, phase + 1):
            if p in state.get('completed_phases', []):
                state['completed_phases'].remove(p)
        
        logger.info(f"Fallback from phase {phase} to phase {fallback_phase} "
                   f"(attempt {self.fallback_count[phase]}/{max_fallbacks})")
        
        return fallback_phase
    
    def _create_failure_response(
        self, 
        phase: int, 
        gate_results: Dict, 
        state: Dict
    ) -> Dict:
        """创建失败响应"""
        return {
            'success': False,
            'error': f'Phase {phase} failed quality gates',
            'phase': phase,
            'gate_results': gate_results,
            'execution_log': self.execution_log,
            'checkpoint': f'phase{phase - 1}' if phase > 1 else None,
            'state': state
        }
    
    def _save_quality_report(self, report: Dict):
        """保存质量报告"""
        report_dir = Path(self.config.get('output', {}).get('quality_reports_dir', 'output/quality_reports'))
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = report_dir / f'quality_report_{timestamp}.json'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Quality report saved to {report_path}")

    # ============ 动态高级算法选择（新增） ============
    
    # 高级算法映射：从model-selector推荐到对应skill
    ADVANCED_ALGORITHM_MAP = {
        'pinn': 'physics-informed-nn',
        'physics-informed neural network': 'physics-informed-nn',
        'neural operator': 'neural-operators',
        'fno': 'neural-operators',
        'deeponet': 'neural-operators',
        'transformer': 'transformer-forecasting',
        'tft': 'transformer-forecasting',
        'temporal fusion': 'transformer-forecasting',
        'reinforcement learning': 'reinforcement-learning',
        'rl': 'reinforcement-learning',
        'dqn': 'reinforcement-learning',
        'ppo': 'reinforcement-learning',
        'kan': 'kan-networks',
        'kolmogorov-arnold': 'kan-networks',
        'causal': 'causal-inference',
        'causal inference': 'causal-inference',
        'dml': 'causal-inference',
        'double machine learning': 'causal-inference',
    }
    
    async def _select_advanced_algorithms(self, state: Dict) -> List[str]:
        """
        根据model-selector的推荐和规划Agent的计划动态选择高级算法skills
        
        Args:
            state: 当前状态（包含model-selector的输出和modeling_plan）
            
        Returns:
            要执行的高级算法skill列表
        """
        selected_skills = []
        
        # 优先使用规划Agent的推荐
        modeling_plan = state.get('modeling_plan', {})
        plan_skills = modeling_plan.get('skills_to_execute', [])
        
        if plan_skills:
            for skill in plan_skills:
                if self.skill_registry.has_skill(skill) and skill not in selected_skills:
                    selected_skills.append(skill)
                    logger.info(f"Dynamic skill selection: adding '{skill}' from Planning Agent")
        
        # 获取model-selector的推荐
        model_selector_result = state.get('model-selector', {})
        recommendations = model_selector_result.get('recommendations', [])
        skills_to_trigger = model_selector_result.get('skills_to_trigger', [])
        
        # 直接使用model-selector推荐的skills
        if skills_to_trigger:
            for skill in skills_to_trigger:
                if skill in PHASE_SKILLS.get(3, []) and skill not in selected_skills:
                    selected_skills.append(skill)
                    logger.info(f"Dynamic skill selection: adding '{skill}' from model-selector recommendation")
        
        # 从recommendations中提取
        for rec in recommendations[:3]:  # 最多选择前3个推荐
            model_type = rec.get('model_type', '').lower()
            category = rec.get('category', '')
            
            # 只关注innovative类型的推荐
            if category != 'innovative' and len(selected_skills) >= 2:
                continue
            
            # 匹配到对应的skill
            for keyword, skill_name in self.ADVANCED_ALGORITHM_MAP.items():
                if keyword in model_type and skill_name not in selected_skills:
                    # 验证skill是否已注册
                    if self.skill_registry.has_skill(skill_name):
                        selected_skills.append(skill_name)
                        logger.info(f"Dynamic skill selection: adding '{skill_name}' based on recommendation '{model_type}'")
                        break
        
        # 根据问题类型添加默认高级算法
        problem_type = state.get('problem_type', state.get('problem-type-classifier', {}).get('problem_type', 'C'))
        
        if not selected_skills:
            # 根据问题类型选择默认的高级算法
            default_algorithms = {
                'A': ['physics-informed-nn', 'kan-networks'],
                'B': ['reinforcement-learning'],
                'C': ['transformer-forecasting', 'causal-inference'],
                'D': ['reinforcement-learning'],
                'E': ['causal-inference'],
                'F': ['causal-inference', 'reinforcement-learning'],
            }
            
            for skill in default_algorithms.get(problem_type, []):
                if self.skill_registry.has_skill(skill):
                    selected_skills.append(skill)
                    logger.info(f"Dynamic skill selection: adding default '{skill}' for problem type {problem_type}")
        
        logger.info(f"Selected advanced algorithms for Phase 3: {selected_skills}")
        return selected_skills
    
    # ============ 章节内容自动扩展（新增） ============
    
    async def _expand_short_sections(self, state: Dict) -> Dict:
        """
        自动扩展不达标的章节内容
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        # 获取需要扩展的章节
        sections_to_expand = self.state_manager.get_sections_needing_expansion()
        
        if not sections_to_expand:
            logger.info("All sections meet minimum word count requirements")
            return state
        
        logger.warning(f"Found {len(sections_to_expand)} sections needing expansion: "
                       f"{[s['section_name'] for s in sections_to_expand]}")
        
        for section_info in sections_to_expand:
            section_name = section_info['section_name']
            current_count = section_info['current_word_count']
            min_required = section_info['min_required']
            deficit = section_info['deficit']
            
            logger.info(f"Expanding section '{section_name}': "
                        f"{current_count}/{min_required} words (need {deficit} more)")
            
            # 获取当前内容
            section_data = state.get('sections', {}).get(section_name, {})
            current_content = section_data.get('content', '')
            
            if not current_content:
                logger.warning(f"Section '{section_name}' has no content to expand")
                continue
            
            try:
                # 扩展内容
                expanded_content = await self._expand_section_content(
                    section_name,
                    current_content,
                    target_additional_words=deficit + 100  # 略微超过目标
                )
                
                # 更新状态
                validation = self.state_manager.set_section(section_name, expanded_content)
                self.state_manager.mark_section_expanded(section_name)
                
                logger.info(f"Section '{section_name}' expanded: "
                            f"{current_count} -> {validation['word_count']} words")
                
            except Exception as e:
                logger.error(f"Failed to expand section '{section_name}': {e}")
        
        return self.state_manager.get_state()
    
    async def _expand_section_content(
        self, 
        section_name: str, 
        content: str, 
        target_additional_words: int
    ) -> str:
        """
        使用LLM扩展章节内容
        
        Args:
            section_name: 章节名称
            content: 当前内容
            target_additional_words: 目标增加字数
            
        Returns:
            扩展后的内容
        """
        expansion_prompt = f"""
You are an expert MCM/ICM paper writer. The following section needs to be expanded to meet 
the minimum word count requirement for an O-award level paper.

**Section Name**: {section_name}
**Current Word Count**: {len(content.split())}
**Additional Words Needed**: {target_additional_words}

**Current Content**:
{content}

**Expansion Requirements**:
1. Preserve all original content and key points
2. Add more detailed explanations and justifications
3. Include additional quantitative analysis where appropriate
4. Add connections to other parts of the paper
5. Expand on the methodology details
6. Add specific examples or case studies
7. Ensure academic writing style throughout

**Important Guidelines**:
- Every conclusion needs a "because..." explanation
- Add at least {target_additional_words // 100} new specific numbers
- Use depth markers: "indicates", "demonstrates", "reveals", "suggests"
- Avoid Chinglish expressions
- Maintain logical flow and coherence

Please output the COMPLETE expanded section content.
"""
        
        # 使用LLM扩展
        response = await self.llm_adapter.complete(
            prompt=expansion_prompt,
            max_tokens=8192,
            temperature=0.7
        )
        
        return response
    
    async def _get_phase3_skills_with_dynamic_selection(self, state: Dict) -> List[str]:
        """
        获取Phase 3的技能列表，包含动态选择的高级算法
        
        Args:
            state: 当前状态
            
        Returns:
            Phase 3要执行的完整技能列表
        """
        # 基础建模技能（必须执行）
        base_skills = [
            'model-selector',
            'model-justification-generator',
            'hybrid-model-designer',
            'model-builder',
            'model-solver',
            'code-verifier'
        ]
        
        # 动态选择的高级算法
        advanced_skills = await self._select_advanced_algorithms(state)
        
        # 合并列表，确保基础技能先执行
        all_skills = base_skills + [s for s in advanced_skills if s not in base_skills]
        
        return all_skills
            
    async def _execute_phase(self, phase: int, state: Dict) -> Dict:
        """
        执行单个阶段 - 支持真正的并行执行和动态技能选择

        使用依赖解析器确定技能分组，组内并行，组间顺序执行
        Phase 3特殊处理：根据model-selector结果动态选择高级算法

        Args:
            phase: 阶段编号 (1-10)
            state: 当前状态

        Returns:
            阶段执行结果
        """
        # 使用依赖解析器获取并行组
        parallel_groups = self.dependency_resolver.get_parallel_groups(phase)

        if not parallel_groups:
            # 没有分组，使用预定义的技能列表
            # Phase 3特殊处理：动态选择高级算法
            if phase == 3:
                skills = await self._get_phase3_skills_with_dynamic_selection(state)
                logger.info(f"Phase 3 skills (with dynamic selection): {skills}")
            else:
                skills = PHASE_SKILLS.get(phase, [])
            
            if skills:
                parallel_groups = [[s] for s in skills]

        results = {}
        enable_parallel = self.config.get('execution', {}).get('parallel_skills', True)

        for group in parallel_groups:
            if len(group) == 0:
                continue

            if len(group) == 1 or not enable_parallel:
                # 单个技能或禁用并行 -> 顺序执行
                for skill in group:
                    try:
                        result = await self._run_skill_with_retry(skill, state)
                        results[skill] = result
                        # 更新状态，使后续技能可访问
                        async with self._state_lock:
                            state.update({skill: result})
                    except Exception as e:
                        # 尝试备选方案
                        fallback = FALLBACK_SKILLS.get(skill)
                        if fallback:
                            logger.warning(f"Skill {skill} failed, trying fallback: {fallback}")
                            try:
                                results[skill] = await self._run_skill(fallback, state)
                            except Exception as fallback_error:
                                logger.error(f"Fallback {fallback} also failed: {fallback_error}")
                                results[skill] = {'status': 'failed', 'error': str(e)}
                        else:
                            raise
            else:
                # 并行执行一组技能
                group_results = await self._execute_parallel_safe(group, state)
                for skill, result in group_results.items():
                    results[skill] = result
                # 并行组执行完成后更新 state，确保后续组可访问这些结果
                async with self._state_lock:
                    for skill in group:
                        if skill in results:
                            state.update({skill: results[skill]})

        return results
    
    async def _execute_parallel_safe(self, skills: List[str], state: Dict) -> Dict[str, Any]:
        """
        并行执行多个技能（线程安全版本）
        
        每个技能使用state的深拷贝进行执行，完成后使用锁安全地合并结果
        
        Args:
            skills: 要并行执行的技能列表
            state: 当前状态
            
        Returns:
            技能执行结果字典
        """
        from copy import deepcopy
        
        results = {}
        
        async def run_skill_safe(skill: str):
            """安全执行单个技能"""
            try:
                # 为每个技能创建状态的深拷贝，避免并发修改
                async with self._state_lock:
                    local_state = deepcopy(state)
                
                result = await self._run_skill_with_retry(skill, local_state)
                return skill, result, None
            except Exception as e:
                return skill, None, e
        
        # 并行执行所有技能
        tasks = [run_skill_safe(skill) for skill in skills]
        completed = await asyncio.gather(*tasks, return_exceptions=False)
        
        # 收集结果
        for skill, result, error in completed:
            if error is not None:
                logger.error(f"Skill {skill} failed with error: {error}")
                results[skill] = {'status': 'failed', 'error': str(error)}
            else:
                results[skill] = result
        
        return results
        
    async def _execute_parallel(self, coroutines: List) -> List:
        """
        并行执行多个协程
        
        注意：此方法保留用于向后兼容，但建议使用_execute_parallel_safe
        """
        return await asyncio.gather(*coroutines, return_exceptions=True)
        
    async def _run_skill_with_retry(self, skill: str, state: Dict) -> Any:
        """
        带重试机制执行技能
        
        Args:
            skill: 技能名称
            state: 当前状态
            
        Returns:
            技能执行结果
        """
        max_retries = self.config.get('execution', {}).get('max_retries', 3)
        retry_delay_base = self.config.get('execution', {}).get('retry_delay_base', 2)
        
        for attempt in range(max_retries):
            try:
                return await self._run_skill(skill, state)
            except RecoverableError as e:
                if attempt < max_retries - 1:
                    delay = retry_delay_base ** attempt
                    logger.warning(f"Skill {skill} failed (attempt {attempt + 1}), retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    raise
                    
    async def _run_skill(self, skill: str, state: Dict) -> Any:
        """
        执行单个技能 - 真正实现

        使用技能执行器来真正执行技能，而不是返回模拟结果

        Args:
            skill: 技能名称
            state: 当前状态

        Returns:
            技能执行结果

        Raises:
            RecoverableError: 技能执行失败但可恢复
            CriticalError: 技能执行严重失败
        """
        logger.info(f"Executing skill: {skill}")

        try:
            # 使用技能执行器执行
            result = await self.skill_executor.execute(
                skill_name=skill,
                state=state,
                context={
                    'phase': self.current_phase,
                    'config': self.config,
                    'execution_id': self._execution_id
                }
            )

            # 检查执行结果
            if not result.success:
                errors = result.errors if result.errors else ["Unknown error"]
                error_msg = f"Skill {skill} failed: {', '.join(errors)}"

                # 根据错误类型决定是否可恢复
                if result.metadata.get('error_type') == 'timeout':
                    raise RecoverableError(error_msg, "SkillTimeout", {'skill': skill})
                else:
                    raise RecoverableError(error_msg, "SkillExecutionError", {'skill': skill})

            # 返回技能数据
            return result.data

        except ValueError as e:
            # 技能不存在
            if "not found" in str(e):
                logger.error(f"Skill not found: {skill}")
                raise CriticalError(f"Skill '{skill}' not registered in the system")
            raise

        except RecoverableError:
            # 重新抛出可恢复错误
            raise

        except Exception as e:
            # 捕获其他异常并包装
            logger.exception(f"Unexpected error executing skill {skill}")
            raise RecoverableError(f"Skill {skill} failed: {str(e)}", "UnexpectedError", {'skill': skill})
        
    def get_status(self) -> Dict:
        """获取当前执行状态"""
        return {
            'current_phase': self.current_phase,
            'execution_log': self.execution_log,
            'fallback_count': self.fallback_count,
            'quality_results': self.quality_checker.check_results,
            'config': self.config,
            # 新增: 技能系统状态
            'skills': {
                'total': len(self.skill_registry),
                'registered': self.skill_registry.get_skill_count(),
                'by_phase': self.skill_registry.list_skills_by_phase()
            },
            'skill_executor_stats': self.skill_executor.get_stats() if hasattr(self.skill_executor, 'get_stats') else {},
            'dependency_resolver_stats': self.dependency_resolver.get_statistics() if hasattr(self.dependency_resolver, 'get_statistics') else {},
        }


# 便捷函数
async def run_mcm_pipeline(
    problem_text: str,
    problem_type: str,
    team_control_number: str,
    data_files: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    resume_from: Optional[int] = None
) -> Dict:
    """
    运行MCM/ICM论文生成流水线的便捷函数
    
    Args:
        problem_text: 题目文本
        problem_type: 题型 (A/B/C/D/E/F)
        team_control_number: 团队控制号
        data_files: 附加数据文件列表
        config_path: 配置文件路径
        resume_from: 从指定阶段恢复
        
    Returns:
        执行结果
    """
    orchestrator = MCMOrchestrator(config_path)
    
    problem_input = {
        'problem_text': problem_text,
        'problem_type': problem_type,
        'team_control_number': team_control_number,
        'data_files': data_files or [],
        'timestamp': datetime.now().isoformat()
    }
    
    return await orchestrator.execute_pipeline(problem_input, resume_from)


if __name__ == '__main__':
    # 测试代码
    import asyncio
    
    async def test():
        result = await run_mcm_pipeline(
            problem_text="Test problem",
            problem_type="A",
            team_control_number="12345"
        )
        print(json.dumps(result, indent=2, default=str))
        
    asyncio.run(test())
