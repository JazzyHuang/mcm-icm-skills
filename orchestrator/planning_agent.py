"""
规划Agent模块
基于CogWriter认知写作框架，在建模前进行任务分解和方法规划
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .knowledge_injector import get_knowledge_injector
from .llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)


class PlanningAgent:
    """
    规划Agent
    
    基于CogWriter认知写作框架，在Phase 3之前进行：
    1. 任务分解 - 将复杂问题分解为子任务
    2. 方法规划 - 选择最佳模型组合
    3. 创新策略 - 规划创新点
    4. 验证策略 - 规划验证方法
    """
    
    def __init__(self, llm_adapter: LLMAdapter):
        """
        初始化规划Agent
        
        Args:
            llm_adapter: LLM适配器
        """
        self.llm = llm_adapter
        self.knowledge_injector = get_knowledge_injector()
    
    async def create_modeling_plan(
        self, 
        problem_analysis: Dict[str, Any], 
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建建模计划
        
        Args:
            problem_analysis: 问题分析结果（来自Phase 2）
            state: 当前状态
            
        Returns:
            完整的建模计划
        """
        logger.info("Planning Agent: Creating modeling plan...")
        
        problem_type = state.get('problem_type', 'C')
        problem_text = state.get('problem_text', '')
        sub_problems = problem_analysis.get('sub_problems', [])
        assumptions = state.get('assumptions', [])
        variables = state.get('variables', {})
        
        # 获取知识库推荐
        model_recommendations = self.knowledge_injector.get_models_for_problem_type(problem_type)
        high_innovation_combos = self.knowledge_injector.get_high_innovation_combinations(problem_type)
        
        # 分析问题特征
        problem_features = await self._analyze_problem_features(
            problem_text, 
            sub_problems,
            problem_type
        )
        
        # 生成建模计划
        plan = await self._generate_plan(
            problem_type=problem_type,
            problem_features=problem_features,
            sub_problems=sub_problems,
            model_recommendations=model_recommendations,
            high_innovation_combos=high_innovation_combos,
            assumptions=assumptions,
            variables=variables
        )
        
        logger.info(f"Planning Agent: Created plan with {len(plan.get('skills_to_execute', []))} skills")
        
        return plan
    
    async def _analyze_problem_features(
        self, 
        problem_text: str,
        sub_problems: List[Dict],
        problem_type: str
    ) -> Dict[str, Any]:
        """
        分析问题特征，确定适用的方法类型
        
        Args:
            problem_text: 问题文本
            sub_problems: 子问题列表
            problem_type: 问题类型
            
        Returns:
            问题特征分析结果
        """
        analysis_prompt = f"""
Analyze the following MCM/ICM problem to identify its key features for model selection.

**Problem Type**: {problem_type}

**Problem Text**:
{problem_text[:2000]}  # Truncate to avoid token limits

**Sub-problems identified**:
{json.dumps(sub_problems[:5], indent=2)}

Please analyze and identify:

1. **Mathematical Characteristics**:
   - Does it involve PDEs/ODEs? (physical processes, diffusion, dynamics)
   - Does it involve time series? (forecasting, trends)
   - Does it involve optimization? (resource allocation, scheduling)
   - Does it involve networks/graphs? (routes, connections)
   - Does it involve uncertainty? (risk, probability)
   - Does it involve causality? (policy effects, interventions)
   - Does it involve multi-agent interactions? (game theory, competition)

2. **Data Characteristics**:
   - Is data provided or needs to be collected?
   - Is the data spatial, temporal, or both?
   - Is the data structured or unstructured?

3. **Output Requirements**:
   - Prediction/forecasting
   - Optimization solution
   - Policy recommendation
   - Risk assessment
   - Ranking/evaluation

Return your analysis in JSON format:
{{
  "mathematical_features": {{
    "has_pde_ode": boolean,
    "has_time_series": boolean,
    "has_optimization": boolean,
    "has_network_structure": boolean,
    "has_uncertainty": boolean,
    "has_causality": boolean,
    "has_multi_agent": boolean
  }},
  "data_features": {{
    "data_availability": "provided|partial|collection_needed",
    "data_type": ["spatial", "temporal", "tabular", "graph"],
    "data_scale": "small|medium|large"
  }},
  "output_type": ["prediction", "optimization", "policy", "risk", "ranking"],
  "complexity_score": 0.0-1.0,
  "innovation_opportunities": ["list of potential innovation points"]
}}
"""
        
        try:
            response = await self.llm.complete(
                prompt=analysis_prompt,
                max_tokens=2048,
                temperature=0.3,
                response_format="json"
            )
            
            # Parse JSON response
            result = json.loads(response)
            return result
            
        except Exception as e:
            logger.warning(f"Failed to analyze problem features: {e}")
            # Return default features based on problem type
            return self._get_default_features(problem_type)
    
    def _get_default_features(self, problem_type: str) -> Dict[str, Any]:
        """获取默认的问题特征（基于问题类型）"""
        defaults = {
            'A': {
                'mathematical_features': {
                    'has_pde_ode': True,
                    'has_time_series': False,
                    'has_optimization': True,
                    'has_network_structure': False,
                    'has_uncertainty': True,
                    'has_causality': False,
                    'has_multi_agent': False
                },
                'data_features': {'data_availability': 'partial', 'data_type': ['spatial'], 'data_scale': 'medium'},
                'output_type': ['prediction', 'optimization'],
                'complexity_score': 0.8,
                'innovation_opportunities': ['PINN for PDE solving', 'KAN for physics discovery']
            },
            'B': {
                'mathematical_features': {
                    'has_pde_ode': False,
                    'has_time_series': False,
                    'has_optimization': True,
                    'has_network_structure': True,
                    'has_uncertainty': False,
                    'has_causality': False,
                    'has_multi_agent': False
                },
                'data_features': {'data_availability': 'provided', 'data_type': ['graph'], 'data_scale': 'medium'},
                'output_type': ['optimization'],
                'complexity_score': 0.75,
                'innovation_opportunities': ['GNN for graph optimization', 'RL for sequential decisions']
            },
            'C': {
                'mathematical_features': {
                    'has_pde_ode': False,
                    'has_time_series': True,
                    'has_optimization': False,
                    'has_network_structure': False,
                    'has_uncertainty': True,
                    'has_causality': True,
                    'has_multi_agent': False
                },
                'data_features': {'data_availability': 'provided', 'data_type': ['temporal', 'tabular'], 'data_scale': 'large'},
                'output_type': ['prediction', 'policy'],
                'complexity_score': 0.7,
                'innovation_opportunities': ['Transformer forecasting', 'Causal inference for policy']
            },
            'D': {
                'mathematical_features': {
                    'has_pde_ode': False,
                    'has_time_series': True,
                    'has_optimization': True,
                    'has_network_structure': True,
                    'has_uncertainty': True,
                    'has_causality': False,
                    'has_multi_agent': False
                },
                'data_features': {'data_availability': 'partial', 'data_type': ['graph', 'temporal'], 'data_scale': 'large'},
                'output_type': ['optimization', 'prediction'],
                'complexity_score': 0.85,
                'innovation_opportunities': ['RL for scheduling', 'GNN for network flow']
            },
            'E': {
                'mathematical_features': {
                    'has_pde_ode': False,
                    'has_time_series': True,
                    'has_optimization': True,
                    'has_network_structure': False,
                    'has_uncertainty': True,
                    'has_causality': True,
                    'has_multi_agent': False
                },
                'data_features': {'data_availability': 'partial', 'data_type': ['temporal', 'tabular'], 'data_scale': 'medium'},
                'output_type': ['optimization', 'policy'],
                'complexity_score': 0.8,
                'innovation_opportunities': ['Multi-objective optimization', 'Causal policy evaluation']
            },
            'F': {
                'mathematical_features': {
                    'has_pde_ode': False,
                    'has_time_series': False,
                    'has_optimization': False,
                    'has_network_structure': True,
                    'has_uncertainty': True,
                    'has_causality': True,
                    'has_multi_agent': True
                },
                'data_features': {'data_availability': 'collection_needed', 'data_type': ['tabular'], 'data_scale': 'small'},
                'output_type': ['policy', 'ranking'],
                'complexity_score': 0.75,
                'innovation_opportunities': ['Game theory analysis', 'MARL for multi-agent']
            }
        }
        
        return defaults.get(problem_type, defaults['C'])
    
    async def _generate_plan(
        self,
        problem_type: str,
        problem_features: Dict,
        sub_problems: List[Dict],
        model_recommendations: Dict,
        high_innovation_combos: List[Dict],
        assumptions: List,
        variables: Dict
    ) -> Dict[str, Any]:
        """
        生成完整的建模计划
        """
        math_features = problem_features.get('mathematical_features', {})
        
        plan = {
            'problem_type': problem_type,
            'analysis_timestamp': None,
            'primary_models': [],
            'innovative_models': [],
            'skills_to_execute': [],
            'execution_order': [],
            'validation_strategy': [],
            'innovation_score_target': 0.85,
            'sub_problem_model_mapping': {},
            'high_innovation_combination': None
        }
        
        # 1. 选择主要模型
        plan['primary_models'] = model_recommendations.get('primary_methods', [])[:2]
        
        # 2. 选择创新模型
        plan['innovative_models'] = model_recommendations.get('innovative_methods', [])[:2]
        
        # 3. 确定要执行的高级算法Skills
        skills_to_execute = []
        
        # 基于问题特征选择skills
        if math_features.get('has_pde_ode'):
            skills_to_execute.extend(['physics-informed-nn', 'neural-operators'])
        
        if math_features.get('has_time_series'):
            skills_to_execute.append('transformer-forecasting')
        
        if math_features.get('has_optimization') and math_features.get('has_uncertainty'):
            skills_to_execute.append('reinforcement-learning')
        
        if math_features.get('has_causality'):
            skills_to_execute.append('causal-inference')
        
        if math_features.get('has_network_structure'):
            # GNN通常通过model-builder实现
            pass
        
        # KAN对于需要可解释性的问题
        if problem_type in ['A', 'C']:
            skills_to_execute.append('kan-networks')
        
        # 去重
        plan['skills_to_execute'] = list(set(skills_to_execute))
        
        # 4. 确定执行顺序
        plan['execution_order'] = [
            {'phase': 'model_selection', 'skills': ['model-selector']},
            {'phase': 'model_justification', 'skills': ['model-justification-generator', 'hybrid-model-designer']},
            {'phase': 'model_building', 'skills': ['model-builder'] + plan['skills_to_execute'][:2]},
            {'phase': 'model_solving', 'skills': ['model-solver', 'code-verifier']},
        ]
        
        # 5. 验证策略
        plan['validation_strategy'] = [
            {
                'method': 'sensitivity_analysis',
                'type': 'Sobol' if math_features.get('has_uncertainty') else 'Morris',
                'skill': 'sensitivity-analyzer'
            },
            {
                'method': 'cross_validation',
                'type': 'k-fold' if problem_features.get('data_features', {}).get('data_scale') == 'large' else 'leave-one-out',
                'skill': 'model-validator'
            },
            {
                'method': 'baseline_comparison',
                'type': 'vs_traditional',
                'skill': 'model-validator'
            }
        ]
        
        # 6. 选择高创新组合
        if high_innovation_combos:
            plan['high_innovation_combination'] = high_innovation_combos[0]
        
        # 7. 映射子问题到模型
        for i, sub_problem in enumerate(sub_problems[:5]):
            sub_name = sub_problem.get('name', f'sub_problem_{i+1}')
            sub_type = sub_problem.get('type', 'general')
            
            # 根据子问题类型选择方法
            if 'predict' in sub_type.lower():
                plan['sub_problem_model_mapping'][sub_name] = plan['innovative_models'][0] if plan['innovative_models'] else plan['primary_models'][0]
            elif 'optim' in sub_type.lower():
                plan['sub_problem_model_mapping'][sub_name] = 'optimization_framework'
            else:
                plan['sub_problem_model_mapping'][sub_name] = plan['primary_models'][0] if plan['primary_models'] else 'general_approach'
        
        return plan
    
    def get_recommended_skills(self, plan: Dict[str, Any]) -> List[str]:
        """
        从计划中获取推荐的skills列表
        
        Args:
            plan: 建模计划
            
        Returns:
            推荐的skill名称列表
        """
        skills = plan.get('skills_to_execute', [])
        
        # 添加验证相关的skills
        for validation in plan.get('validation_strategy', []):
            if 'skill' in validation:
                skills.append(validation['skill'])
        
        return list(set(skills))


def create_planning_agent(llm_adapter: LLMAdapter) -> PlanningAgent:
    """创建规划Agent实例"""
    return PlanningAgent(llm_adapter)
