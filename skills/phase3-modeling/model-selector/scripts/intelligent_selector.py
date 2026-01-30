"""
Intelligent Model Selector
智能模型选择器

根据问题特征自动推荐模型，并计算创新性评分
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class IntelligentModelSelector:
    """智能模型选择器"""
    
    def __init__(self, model_database_path: str = None):
        """
        Args:
            model_database_path: model_database.json路径
        """
        self.model_database_path = model_database_path
        self.model_database = None
        self._load_database()
        
    def _load_database(self):
        """加载模型数据库"""
        if self.model_database_path and Path(self.model_database_path).exists():
            with open(self.model_database_path) as f:
                self.model_database = json.load(f)
        else:
            # 使用内置数据
            self.model_database = self._get_default_database()
            
    def _get_default_database(self) -> Dict:
        """获取默认模型数据库"""
        return {
            "problem_type_recommendations": {
                "A": {
                    "primary": ["ODE", "PDE", "FiniteElement", "Optimization"],
                    "innovative": ["PINN", "FNO", "DeepONet", "KAN"],
                    "visualization": ["3D surface", "contour", "streamplot"]
                },
                "B": {
                    "primary": ["GraphTheory", "DynamicProgramming", "MILP"],
                    "innovative": ["GNN", "DQN", "ACO"],
                    "visualization": ["network graph", "tree diagram"]
                },
                "C": {
                    "primary": ["MachineLearning", "TimeSeries", "Clustering"],
                    "innovative": ["Transformer", "CausalInference", "SHAP"],
                    "visualization": ["SHAP summary", "partial dependence"]
                },
                "D": {
                    "primary": ["NetworkFlow", "MILP", "Scheduling"],
                    "innovative": ["DQN", "PPO", "ACO"],
                    "visualization": ["Gantt chart", "network flow"]
                },
                "E": {
                    "primary": ["MultiObjective", "SystemDynamics"],
                    "innovative": ["NSGA-III", "MOEAD", "CausalForest"],
                    "visualization": ["Pareto front", "radar chart"]
                },
                "F": {
                    "primary": ["GameTheory", "ABM", "PolicyAnalysis"],
                    "innovative": ["MARL", "EvolutionaryGame"],
                    "visualization": ["game tree", "policy impact"]
                }
            },
            "innovation_scores": {
                "PINN": 0.95, "FNO": 0.96, "KAN": 0.98,
                "GNN": 0.95, "Transformer": 0.9, "MARL": 0.92,
                "CausalForest": 0.88, "NSGA-III": 0.92,
                "SHAP": 0.85, "LIME": 0.8,
                "DQN": 0.85, "PPO": 0.88,
                "ARIMA": 0.3, "LinearRegression": 0.2,
                "RandomForest": 0.5, "XGBoost": 0.6
            }
        }
        
    def analyze_problem(self, problem_text: str) -> Dict:
        """
        分析问题文本，提取关键特征
        
        Args:
            problem_text: 问题描述文本
            
        Returns:
            问题特征字典
        """
        features = {
            'problem_type': None,
            'keywords': [],
            'data_characteristics': {},
            'constraints': []
        }
        
        # 关键词识别
        keyword_mapping = {
            'continuous': ['微分', 'differential', 'PDE', 'ODE', '连续', 'continuous', '流体', 'fluid', '热', 'heat'],
            'discrete': ['离散', 'discrete', '图', 'graph', '网络', 'network', '路径', 'path', '调度', 'scheduling'],
            'prediction': ['预测', 'predict', 'forecast', '时间序列', 'time series'],
            'optimization': ['优化', 'optimize', 'minimize', 'maximize', '最优', 'optimal'],
            'policy': ['政策', 'policy', '可持续', 'sustainable', '决策', 'decision'],
            'game': ['博弈', 'game', '策略', 'strategy', '竞争', 'competition', '合作', 'cooperation']
        }
        
        problem_lower = problem_text.lower()
        
        for category, keywords in keyword_mapping.items():
            for kw in keywords:
                if kw.lower() in problem_lower:
                    features['keywords'].append(kw)
                    
        # 问题类型推断
        if any(k in features['keywords'] for k in ['微分', 'PDE', 'ODE', '流体', '热']):
            features['problem_type'] = 'A'
        elif any(k in features['keywords'] for k in ['图', '网络', '路径', '调度']):
            features['problem_type'] = 'B' if '调度' not in features['keywords'] else 'D'
        elif any(k in features['keywords'] for k in ['预测', '时间序列']):
            features['problem_type'] = 'C'
        elif any(k in features['keywords'] for k in ['可持续', '多目标']):
            features['problem_type'] = 'E'
        elif any(k in features['keywords'] for k in ['博弈', '策略', '政策']):
            features['problem_type'] = 'F'
        else:
            features['problem_type'] = 'C'  # 默认
            
        return features
        
    def recommend_models(self, problem_type: str, 
                         prefer_innovation: bool = True,
                         max_models: int = 5) -> List[Dict]:
        """
        根据问题类型推荐模型
        
        Args:
            problem_type: 'A', 'B', 'C', 'D', 'E', 'F'
            prefer_innovation: 是否优先创新性高的模型
            max_models: 最大推荐数量
            
        Returns:
            推荐模型列表，按创新性排序
        """
        recommendations = []
        
        if self.model_database and 'problem_type_recommendations' in self.model_database:
            recs = self.model_database['problem_type_recommendations'].get(problem_type, {})
        else:
            recs = self._get_default_database()['problem_type_recommendations'].get(problem_type, {})
            
        innovation_scores = (self.model_database.get('innovation_scores') 
                            if self.model_database else 
                            self._get_default_database()['innovation_scores'])
        
        # 收集所有推荐模型
        all_models = []
        
        for model in recs.get('innovative', []):
            score = innovation_scores.get(model, 0.8)
            all_models.append({
                'name': model,
                'category': 'innovative',
                'innovation_score': score,
                'recommendation_reason': f'Highly innovative for Type {problem_type} problems'
            })
            
        for model in recs.get('primary', []):
            score = innovation_scores.get(model, 0.5)
            all_models.append({
                'name': model,
                'category': 'primary',
                'innovation_score': score,
                'recommendation_reason': f'Fundamental method for Type {problem_type} problems'
            })
            
        # 排序
        if prefer_innovation:
            all_models.sort(key=lambda x: x['innovation_score'], reverse=True)
        
        return all_models[:max_models]
        
    def calculate_innovation_score(self, models_used: List[str],
                                   combinations: List[Tuple[str, str]] = None) -> Dict:
        """
        计算整体创新性评分
        
        Args:
            models_used: 使用的模型列表
            combinations: 模型组合 [(model1, model2), ...]
            
        Returns:
            创新性评分详情
        """
        scores = self._get_default_database()['innovation_scores']
        
        # 单模型分数
        model_scores = []
        for model in models_used:
            score = scores.get(model, 0.5)
            model_scores.append({'model': model, 'score': score})
            
        # 组合加分
        combination_bonus = 0
        high_innovation_combinations = [
            ('PINN', 'Traditional'),  # PINN + 传统方法
            ('Transformer', 'SHAP'),   # 时序预测 + 解释
            ('GNN', 'Optimization'),   # 图神经网络 + 优化
            ('Causal', 'ML'),          # 因果推断 + ML
        ]
        
        if combinations:
            for combo in combinations:
                for hi_combo in high_innovation_combinations:
                    if any(hi_combo[0] in c for c in combo) and any(hi_combo[1] in c for c in combo):
                        combination_bonus += 0.1
                        
        # 多样性加分
        categories = set()
        for model in models_used:
            if 'PINN' in model or 'FNO' in model:
                categories.add('physics_informed')
            elif 'Transformer' in model or 'LSTM' in model:
                categories.add('deep_learning')
            elif 'SHAP' in model or 'LIME' in model:
                categories.add('explainability')
            elif 'Causal' in model or 'DiD' in model:
                categories.add('causal')
                
        diversity_bonus = min(len(categories) * 0.05, 0.15)
        
        # 总分
        base_score = np.mean([s['score'] for s in model_scores]) if model_scores else 0.5
        total_score = min(base_score + combination_bonus + diversity_bonus, 1.0)
        
        return {
            'total_score': round(total_score, 3),
            'base_score': round(base_score, 3),
            'combination_bonus': round(combination_bonus, 3),
            'diversity_bonus': round(diversity_bonus, 3),
            'model_scores': model_scores,
            'categories_used': list(categories),
            'o_award_potential': 'High' if total_score >= 0.85 else 
                                'Medium' if total_score >= 0.7 else 'Low'
        }
        
    def get_visualization_recommendations(self, problem_type: str) -> List[str]:
        """
        获取可视化推荐
        
        Args:
            problem_type: 问题类型
            
        Returns:
            推荐的图表类型列表
        """
        recs = self._get_default_database()['problem_type_recommendations'].get(problem_type, {})
        return recs.get('visualization', ['line chart', 'bar chart'])
        
    def generate_model_combination(self, problem_type: str) -> Dict:
        """
        生成高创新性模型组合建议
        
        Args:
            problem_type: 问题类型
            
        Returns:
            模型组合建议
        """
        combinations = {
            'A': {
                'combination': ['PINN', 'FDM/FEM (验证)', 'SHAP'],
                'rationale': '用PINN求解PDE，传统方法验证，SHAP解释物理参数影响',
                'expected_score': 0.92
            },
            'B': {
                'combination': ['GNN', 'ACO', 'Traditional Graph Algorithm'],
                'rationale': '用GNN学习图结构，ACO优化路径，传统算法验证',
                'expected_score': 0.88
            },
            'C': {
                'combination': ['Transformer', 'SHAP', 'Conformal Prediction'],
                'rationale': '深度学习预测，SHAP解释，保形预测量化不确定性',
                'expected_score': 0.9
            },
            'D': {
                'combination': ['PPO/DQN', 'MILP', 'Discrete Event Simulation'],
                'rationale': '强化学习优化决策，MILP求解，仿真验证',
                'expected_score': 0.87
            },
            'E': {
                'combination': ['NSGA-III', 'System Dynamics', 'Causal Inference'],
                'rationale': '多目标优化，系统动力学建模，因果推断验证',
                'expected_score': 0.89
            },
            'F': {
                'combination': ['MARL', 'Evolutionary Game', 'ABM'],
                'rationale': '多智能体学习均衡策略，演化博弈分析，ABM仿真验证',
                'expected_score': 0.91
            }
        }
        
        return combinations.get(problem_type, {
            'combination': ['Machine Learning', 'Statistical Analysis'],
            'rationale': '通用方法',
            'expected_score': 0.7
        })


def select_models_for_problem(problem_text: str, 
                              prefer_innovation: bool = True) -> Dict:
    """
    便捷函数：为问题选择模型
    
    Args:
        problem_text: 问题描述
        prefer_innovation: 是否优先创新
        
    Returns:
        完整推荐结果
    """
    selector = IntelligentModelSelector()
    
    # 分析问题
    features = selector.analyze_problem(problem_text)
    problem_type = features['problem_type']
    
    # 推荐模型
    recommendations = selector.recommend_models(problem_type, prefer_innovation)
    
    # 生成组合
    combination = selector.generate_model_combination(problem_type)
    
    # 可视化推荐
    visualizations = selector.get_visualization_recommendations(problem_type)
    
    return {
        'problem_analysis': features,
        'recommended_models': recommendations,
        'suggested_combination': combination,
        'visualization_recommendations': visualizations
    }


if __name__ == '__main__':
    print("Testing Intelligent Model Selector...")
    
    selector = IntelligentModelSelector()
    
    # 测试问题分析
    test_problems = [
        "热传导方程求解，分析温度分布",
        "物流网络优化，最短路径问题",
        "销售数据预测，时间序列分析",
        "多目标可持续发展规划",
        "多方博弈策略分析"
    ]
    
    for problem in test_problems:
        features = selector.analyze_problem(problem)
        print(f"\nProblem: {problem}")
        print(f"  Type: {features['problem_type']}")
        
        recs = selector.recommend_models(features['problem_type'])
        print(f"  Top models: {[r['name'] for r in recs[:3]]}")
        
        combo = selector.generate_model_combination(features['problem_type'])
        print(f"  Suggested combo: {combo['combination']}")
        
    # 测试创新性评分
    score = selector.calculate_innovation_score(
        ['PINN', 'XGBoost', 'SHAP'],
        [('PINN', 'FDM')]
    )
    print(f"\nInnovation Score: {score['total_score']}")
    print(f"O-Award Potential: {score['o_award_potential']}")
    
    print("\nIntelligent Model Selector test completed!")
