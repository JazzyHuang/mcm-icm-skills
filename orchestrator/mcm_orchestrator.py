"""
MCM/ICM 主编排器
基于AgentOrchestra架构的层级式多智能体编排器
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .error_recovery import ErrorRecoveryManager, RecoverableError, CriticalError
from .state_manager import StateManager
from .checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


# 阶段技能映射
PHASE_SKILLS = {
    1: ['problem-parser', 'problem-type-classifier', 'data-collector', 
        'literature-searcher', 'citation-validator'],
    2: ['problem-decomposer', 'sub-problem-analyzer', 'assumption-generator',
        'variable-definer', 'constraint-identifier'],
    3: ['model-selector', 'hybrid-model-designer', 'model-builder',
        'model-solver', 'code-verifier'],
    4: ['sensitivity-analyzer', 'uncertainty-quantifier', 'model-validator',
        'strengths-weaknesses', 'ethical-analyzer'],
    5: ['section-writer', 'fact-checker', 'abstract-generator',
        'abstract-iterative-optimizer', 'memo-letter-writer'],
    6: ['chart-generator', 'publication-scaler', 'table-formatter',
        'figure-validator'],
    7: ['latex-compiler', 'compilation-error-handler', 'citation-manager',
        'format-checker', 'anonymization-checker'],
    8: ['quality-reviewer', 'hallucination-detector', 'grammar-checker',
        'consistency-checker'],
    9: ['final-polisher', 'academic-english-optimizer', 'submission-preparer'],
    10: ['pre-submission-validator', 'submission-checklist']
}

# 可并行执行的技能组
PARALLEL_SKILLS = {
    1: [['problem-parser'], ['data-collector', 'literature-searcher']],
}

# 备选技能映射
FALLBACK_SKILLS = {
    'model-solver': 'model-solver-fallback',
    'data-collector': 'data-collector-fallback',
    'latex-compiler': 'latex-compiler-fallback',
}


class MCMOrchestrator:
    """
    层级式多智能体编排器
    基于AgentOrchestra架构设计
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
        self.current_phase = 0
        self.execution_log = []
        
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
            },
            'quality_thresholds': {
                'abstract_min_iterations': 12,
                'grammar_score_min': 7.5,
                'hallucination_tolerance': 0.0,
                'citation_verification': True,
            },
            'output': {
                'checkpoint_dir': 'output/checkpoints',
                'papers_dir': 'output/papers',
                'figures_dir': 'output/figures',
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
            problem_input: 问题输入，包含:
                - problem_text: 题目文本
                - problem_type: 题型 (A/B/C/D/E/F)
                - data_files: 附加数据文件列表 (可选)
                - team_control_number: 团队控制号
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
            # 执行各阶段
            for phase in range(start_phase, 11):
                self.current_phase = phase
                logger.info(f"=== Starting Phase {phase} ===")
                
                phase_start = datetime.now()
                phase_results = await self._execute_phase(phase, state)
                phase_duration = (datetime.now() - phase_start).total_seconds()
                
                # 更新状态
                state.update(phase_results)
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
                
            # 获取最终输出
            final_output = self.state_manager.get_final_output(state)
            
            total_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Pipeline completed in {total_duration:.2f}s")
            
            return {
                'success': True,
                'output': final_output,
                'execution_log': self.execution_log,
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
                        results[skill] = await self._run_skill(fallback, state)
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
