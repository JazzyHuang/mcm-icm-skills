"""
MCM/ICM 主编排器
基于AgentOrchestra架构的层级式多智能体编排器
增强版: 包含质量门禁和阶段回退逻辑
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .error_recovery import ErrorRecoveryManager, RecoverableError, CriticalError
from .state_manager import StateManager
from .checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


# 阶段技能映射 (增强版)
PHASE_SKILLS = {
    1: ['problem-parser', 'problem-type-classifier', 'problem-reference-extractor',
        'data-collector', 'deep-reference-searcher', 'literature-searcher', 
        'citation-validator', 'citation-diversity-validator'],
    2: ['problem-decomposer', 'sub-problem-analyzer', 'assumption-generator',
        'variable-definer', 'constraint-identifier'],
    3: ['model-selector', 'model-justification-generator', 'hybrid-model-designer', 
        'model-builder', 'model-solver', 'code-verifier'],
    4: ['sensitivity-analyzer', 'uncertainty-quantifier', 'model-validator',
        'error-analyzer', 'limitation-analyzer', 'strengths-weaknesses', 'ethical-analyzer'],
    5: ['section-writer', 'fact-checker', 'abstract-first-impression',
        'abstract-generator', 'abstract-iterative-optimizer', 'memo-letter-writer'],
    6: ['chart-generator', 'figure-narrative-generator', 'publication-scaler', 
        'table-formatter', 'figure-validator'],
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
        value_key = criterion.get('name')
        actual_value = results.get(value_key) or state.get(value_key)
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
    增强版: 包含质量门禁和阶段回退逻辑
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化编排器
        
        Args:
            config_path: 配置文件路径，默认使用 config/settings.yaml
        """
        self.config = self._load_config(config_path)
        self.state_manager = StateManager()
        self.error_handler = ErrorRecoveryManager(self.config)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(self.config.get('output', {}).get('checkpoint_dir', 'output/checkpoints'))
        )
        self.quality_checker = QualityGateChecker()
        self.current_phase = 0
        self.execution_log = []
        self.fallback_count = {}  # 记录每个阶段的回退次数
        
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
                            self.execution_log.append({
                                'phase': phase,
                                'duration_seconds': phase_duration,
                                'status': 'failed',
                                'gate_results': gate_results
                            })
                            # 根据配置决定是继续还是停止
                            if self.config.get('execution', {}).get('stop_on_gate_failure', False):
                                return self._create_failure_response(phase, gate_results, state)
                
                # 门禁通过或未启用门禁
                state['completed_phases'].append(phase)
                
                # 保存检查点
                self.checkpoint_manager.save(state, f'phase{phase}')
                
                # 记录执行日志
                self.execution_log.append({
                    'phase': phase,
                    'duration_seconds': phase_duration,
                    'skills_executed': list(phase_results.keys()),
                    'status': 'success'
                })
                
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
            
    async def _execute_phase(self, phase: int, state: Dict) -> Dict:
        """
        执行单个阶段
        
        Args:
            phase: 阶段编号 (1-10)
            state: 当前状态
            
        Returns:
            阶段执行结果
        """
        skills = PHASE_SKILLS.get(phase, [])
        results = {}
        
        # 检查是否有可并行执行的技能组
        if phase in PARALLEL_SKILLS and self.config.get('execution', {}).get('parallel_skills', True):
            skill_groups = PARALLEL_SKILLS[phase]
            for group in skill_groups:
                if len(group) > 1:
                    # 并行执行
                    group_results = await self._execute_parallel(
                        [self._run_skill_with_retry(skill, state) for skill in group]
                    )
                    for skill, result in zip(group, group_results):
                        if isinstance(result, Exception):
                            logger.error(f"Skill {skill} failed with error: {result}")
                            results[skill] = {'status': 'failed', 'error': str(result)}
                        else:
                            results[skill] = result
                else:
                    # 单个执行
                    for skill in group:
                        results[skill] = await self._run_skill_with_retry(skill, state)
                        state.update({skill: results[skill]})
        else:
            # 顺序执行
            for skill in skills:
                try:
                    result = await self._run_skill_with_retry(skill, state)
                    results[skill] = result
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
                        
        return results
        
    async def _execute_parallel(self, coroutines: List) -> List:
        """并行执行多个协程"""
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
        执行单个技能
        
        Args:
            skill: 技能名称
            state: 当前状态
            
        Returns:
            技能执行结果
        """
        logger.info(f"Executing skill: {skill}")
        
        # 这里是技能执行的占位实现
        # 在实际使用中，这会调用对应的技能处理逻辑
        # 技能可以是:
        # 1. 调用LLM完成特定任务
        # 2. 执行Python脚本
        # 3. 调用外部API
        
        # 模拟技能执行
        result = {
            'skill': skill,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    def get_status(self) -> Dict:
        """获取当前执行状态"""
        return {
            'current_phase': self.current_phase,
            'execution_log': self.execution_log,
            'fallback_count': self.fallback_count,
            'quality_results': self.quality_checker.check_results,
            'config': self.config
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
