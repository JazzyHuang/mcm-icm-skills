"""
结构分析器
分析美赛题目的结构并生成结构化输出
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def analyze_structure(parsed_data: Dict) -> Dict:
    """
    分析解析后的数据结构
    
    Args:
        parsed_data: 从pdf_parser或文本解析得到的数据
        
    Returns:
        结构化的问题描述
    """
    result = {
        'problem_type': parsed_data.get('problem_type'),
        'problem_title': parsed_data.get('problem_title'),
        'year': parsed_data.get('year'),
        'contest': determine_contest(parsed_data.get('problem_type')),
        'background': parsed_data.get('background', ''),
        'main_questions': parsed_data.get('main_questions', []),
        'sub_questions': extract_sub_questions(parsed_data),
        'provided_data': parsed_data.get('provided_data'),
        'constraints': parsed_data.get('constraints', []),
        'assumptions_needed': identify_assumption_areas(parsed_data),
        'model_types_suggested': suggest_model_types(parsed_data),
        'keywords': parsed_data.get('keywords', []),
        'complexity_estimate': estimate_complexity(parsed_data),
        'deliverables': extract_deliverables(parsed_data),
        'validation': {
            'has_type': parsed_data.get('problem_type') is not None,
            'has_title': parsed_data.get('problem_title') is not None,
            'has_questions': len(parsed_data.get('main_questions', [])) > 0,
            'completeness_score': calculate_completeness(parsed_data)
        }
    }
    
    return result


def determine_contest(problem_type: Optional[str]) -> str:
    """确定竞赛类型"""
    if problem_type in ['A', 'B', 'C']:
        return 'MCM'
    elif problem_type in ['D', 'E', 'F']:
        return 'ICM'
    return 'Unknown'


def extract_sub_questions(parsed_data: Dict) -> List[Dict]:
    """提取子问题的更细粒度分析"""
    main_questions = parsed_data.get('main_questions', [])
    sub_questions = []
    
    for q in main_questions:
        description = q.get('description', '')
        
        # 分析子问题的复杂度
        complexity = analyze_question_complexity(description)
        
        # 识别所需的方法类型
        methods = identify_required_methods(description)
        
        # 识别输出要求
        outputs = identify_outputs(description)
        
        sub_questions.append({
            'id': q.get('id'),
            'description': description,
            'type': q.get('type'),
            'complexity': complexity,
            'suggested_methods': methods,
            'expected_outputs': outputs
        })
        
    return sub_questions


def analyze_question_complexity(description: str) -> Dict:
    """分析问题复杂度"""
    desc_lower = description.lower()
    
    # 复杂度因素
    factors = {
        'multi_variable': any(w in desc_lower for w in ['multiple', 'various', 'different', 'several']),
        'time_dependent': any(w in desc_lower for w in ['time', 'temporal', 'dynamic', 'over time']),
        'optimization': any(w in desc_lower for w in ['optimize', 'maximize', 'minimize', 'optimal']),
        'uncertainty': any(w in desc_lower for w in ['uncertain', 'random', 'probabilistic', 'stochastic']),
        'large_scale': any(w in desc_lower for w in ['large', 'scale', 'global', 'nationwide']),
        'multi_objective': any(w in desc_lower for w in ['trade-off', 'balance', 'multiple objectives']),
    }
    
    # 计算复杂度分数
    complexity_score = sum(factors.values())
    
    if complexity_score <= 1:
        level = 'low'
    elif complexity_score <= 3:
        level = 'medium'
    else:
        level = 'high'
        
    return {
        'level': level,
        'score': complexity_score,
        'factors': factors
    }


def identify_required_methods(description: str) -> List[str]:
    """识别所需方法"""
    desc_lower = description.lower()
    methods = []
    
    method_keywords = {
        'optimization': ['optimize', 'maximize', 'minimize', 'optimal'],
        'prediction': ['predict', 'forecast', 'estimate', 'project'],
        'simulation': ['simulate', 'simulation', 'monte carlo'],
        'statistical_analysis': ['analyze', 'statistical', 'regression', 'correlation'],
        'differential_equations': ['rate', 'change', 'differential', 'continuous'],
        'graph_theory': ['network', 'graph', 'path', 'flow'],
        'machine_learning': ['classify', 'cluster', 'pattern', 'learn'],
        'sensitivity_analysis': ['sensitive', 'sensitivity', 'robust'],
        'cost_benefit': ['cost', 'benefit', 'economic', 'efficiency'],
    }
    
    for method, keywords in method_keywords.items():
        if any(kw in desc_lower for kw in keywords):
            methods.append(method)
            
    return methods


def identify_outputs(description: str) -> List[str]:
    """识别预期输出"""
    desc_lower = description.lower()
    outputs = []
    
    output_patterns = {
        'model': ['model', 'equation', 'formula'],
        'analysis': ['analysis', 'evaluate', 'assess'],
        'recommendation': ['recommend', 'suggest', 'propose'],
        'visualization': ['graph', 'chart', 'plot', 'visualize'],
        'numerical_result': ['calculate', 'compute', 'determine'],
        'memo': ['memo', 'letter', 'report'],
        'sensitivity': ['sensitivity', 'robust'],
    }
    
    for output_type, keywords in output_patterns.items():
        if any(kw in desc_lower for kw in keywords):
            outputs.append(output_type)
            
    return outputs


def identify_assumption_areas(parsed_data: Dict) -> List[Dict]:
    """识别需要假设的领域"""
    assumption_areas = []
    
    # 分析背景和问题
    text = parsed_data.get('background', '') + ' '
    for q in parsed_data.get('main_questions', []):
        text += q.get('description', '') + ' '
        
    text_lower = text.lower()
    
    # 常见需要假设的领域
    areas = {
        'data_quality': {
            'keywords': ['data', 'measurement', 'observation'],
            'description': '数据质量和完整性假设'
        },
        'system_behavior': {
            'keywords': ['system', 'process', 'mechanism'],
            'description': '系统行为假设'
        },
        'external_factors': {
            'keywords': ['external', 'environment', 'condition'],
            'description': '外部因素假设'
        },
        'simplification': {
            'keywords': ['complex', 'simplify', 'approximate'],
            'description': '模型简化假设'
        },
        'time_scope': {
            'keywords': ['time', 'period', 'duration'],
            'description': '时间范围假设'
        },
        'spatial_scope': {
            'keywords': ['location', 'region', 'area', 'spatial'],
            'description': '空间范围假设'
        },
    }
    
    for area_name, area_info in areas.items():
        if any(kw in text_lower for kw in area_info['keywords']):
            assumption_areas.append({
                'area': area_name,
                'description': area_info['description'],
                'importance': 'high' if area_name in ['data_quality', 'system_behavior'] else 'medium'
            })
            
    return assumption_areas


def suggest_model_types(parsed_data: Dict) -> List[Dict]:
    """建议模型类型"""
    problem_type = parsed_data.get('problem_type')
    
    # 基于题型的基础建议
    base_suggestions = {
        'A': [
            {'type': 'continuous', 'methods': ['ODE', 'PDE', '有限元'], 'confidence': 'high'},
            {'type': 'optimization', 'methods': ['非线性规划', '变分法'], 'confidence': 'medium'},
        ],
        'B': [
            {'type': 'discrete', 'methods': ['图论', '动态规划', '组合优化'], 'confidence': 'high'},
            {'type': 'algorithm', 'methods': ['启发式算法', '近似算法'], 'confidence': 'medium'},
        ],
        'C': [
            {'type': 'data_analysis', 'methods': ['机器学习', '统计分析'], 'confidence': 'high'},
            {'type': 'prediction', 'methods': ['时间序列', '回归分析'], 'confidence': 'high'},
        ],
        'D': [
            {'type': 'network', 'methods': ['网络流', '图算法'], 'confidence': 'high'},
            {'type': 'optimization', 'methods': ['整数规划', '调度算法'], 'confidence': 'high'},
        ],
        'E': [
            {'type': 'multi_objective', 'methods': ['Pareto优化', '权重法'], 'confidence': 'high'},
            {'type': 'system_dynamics', 'methods': ['系统动力学', '因果回路'], 'confidence': 'medium'},
        ],
        'F': [
            {'type': 'decision_making', 'methods': ['博弈论', '决策分析'], 'confidence': 'high'},
            {'type': 'simulation', 'methods': ['Agent模拟', '蒙特卡洛'], 'confidence': 'medium'},
        ],
    }
    
    return base_suggestions.get(problem_type, [])


def estimate_complexity(parsed_data: Dict) -> Dict:
    """估计整体复杂度"""
    # 计算各维度复杂度
    
    # 问题数量
    num_questions = len(parsed_data.get('main_questions', []))
    
    # 约束数量
    num_constraints = len(parsed_data.get('constraints', []))
    
    # 数据复杂度
    data = parsed_data.get('provided_data')
    data_complexity = 1 if data and data.get('files') else 0
    
    # 背景复杂度
    background_length = len(parsed_data.get('background', ''))
    background_complexity = min(background_length / 500, 3)
    
    # 综合评分
    total_score = (
        num_questions * 1.5 +
        num_constraints * 0.5 +
        data_complexity * 2 +
        background_complexity
    )
    
    if total_score <= 5:
        level = 'low'
        estimated_hours = '8-12'
    elif total_score <= 10:
        level = 'medium'
        estimated_hours = '12-20'
    else:
        level = 'high'
        estimated_hours = '20-30'
        
    return {
        'level': level,
        'score': round(total_score, 2),
        'estimated_hours': estimated_hours,
        'breakdown': {
            'questions': num_questions,
            'constraints': num_constraints,
            'data_complexity': data_complexity,
            'background_complexity': round(background_complexity, 2)
        }
    }


def extract_deliverables(parsed_data: Dict) -> List[Dict]:
    """提取交付物要求"""
    deliverables = []
    
    # 标准交付物
    standard = [
        {'name': 'Summary/Abstract', 'required': True, 'format': '1页'},
        {'name': 'Full Paper', 'required': True, 'format': '≤25页'},
    ]
    deliverables.extend(standard)
    
    # 检查是否需要memo
    text = ''
    for q in parsed_data.get('main_questions', []):
        text += q.get('description', '').lower()
        
    if 'memo' in text or 'letter' in text:
        deliverables.append({
            'name': 'Memo/Letter',
            'required': True,
            'format': '1-2页'
        })
        
    return deliverables


def calculate_completeness(parsed_data: Dict) -> float:
    """计算完整性分数"""
    checks = [
        parsed_data.get('problem_type') is not None,
        parsed_data.get('problem_title') is not None,
        len(parsed_data.get('background', '')) > 50,
        len(parsed_data.get('main_questions', [])) > 0,
        len(parsed_data.get('keywords', [])) > 3,
    ]
    
    return sum(checks) / len(checks)


def to_json(structured_data: Dict, indent: int = 2) -> str:
    """转换为JSON字符串"""
    return json.dumps(structured_data, indent=indent, ensure_ascii=False)


if __name__ == '__main__':
    # 测试代码
    from pdf_parser import parse_problem_text
    
    test_text = """
    2026 MCM Problem A: Optimizing Solar Panel Placement
    
    Background: Solar energy is becoming increasingly important as the world
    transitions to renewable energy sources. Efficient placement of solar panels
    can significantly impact energy production. Geographic location, seasonal
    variations, and local weather patterns all affect the optimal configuration.
    
    Your team should:
    1. Develop a mathematical model to determine optimal panel angles based on
       geographic location and seasonal variations.
    2. Analyze the impact of different panel configurations on energy output.
    3. Perform sensitivity analysis on your model parameters.
    4. Write a one-page memo to a solar company summarizing your recommendations.
    
    Data: Use the provided solar_data.csv file containing historical solar
    radiation measurements.
    
    Constraints:
    - Panels must be fixed at a single angle (no tracking systems)
    - Consider only continental US locations
    """
    
    parsed = parse_problem_text(test_text)
    structured = analyze_structure(parsed)
    
    print(to_json(structured))
