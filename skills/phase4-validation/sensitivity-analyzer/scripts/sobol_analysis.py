"""
Sobol敏感性分析
全局敏感性分析的标准方法
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def run_sobol_analysis(
    model: Callable,
    param_names: List[str],
    param_bounds: List[Tuple[float, float]],
    n_samples: int = 2048,
    calc_second_order: bool = True
) -> Dict:
    """
    执行Sobol敏感性分析
    
    Args:
        model: 模型函数，接受参数数组返回标量
        param_names: 参数名称列表
        param_bounds: 参数边界列表 [(min, max), ...]
        n_samples: 样本数量
        calc_second_order: 是否计算二阶交互效应
        
    Returns:
        敏感性分析结果
    """
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol
    except ImportError:
        logger.error("SALib not installed. Run: pip install SALib")
        raise
        
    # 定义问题
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': param_bounds
    }
    
    logger.info(f"Generating Saltelli samples with N={n_samples}")
    
    # 生成样本
    samples = saltelli.sample(problem, n_samples, calc_second_order=calc_second_order)
    
    logger.info(f"Evaluating model for {len(samples)} samples")
    
    # 计算模型输出
    Y = np.zeros(len(samples))
    for i, sample in enumerate(samples):
        try:
            Y[i] = model(sample)
        except Exception as e:
            logger.warning(f"Model evaluation failed for sample {i}: {e}")
            Y[i] = np.nan
            
    # 移除NaN
    valid_mask = ~np.isnan(Y)
    if not all(valid_mask):
        logger.warning(f"Removed {sum(~valid_mask)} invalid samples")
        samples = samples[valid_mask]
        Y = Y[valid_mask]
        
    # Sobol分析
    logger.info("Running Sobol analysis")
    Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order)
    
    # 整理结果
    results = {
        'problem': problem,
        'n_samples': n_samples,
        'first_order': {
            name: {
                'S1': float(Si['S1'][i]),
                'S1_conf': float(Si['S1_conf'][i])
            }
            for i, name in enumerate(param_names)
        },
        'total_order': {
            name: {
                'ST': float(Si['ST'][i]),
                'ST_conf': float(Si['ST_conf'][i])
            }
            for i, name in enumerate(param_names)
        },
        'ranking': get_parameter_ranking(Si, param_names)
    }
    
    if calc_second_order:
        results['second_order'] = extract_second_order(Si, param_names)
        
    # 添加解释
    results['interpretation'] = interpret_results(results, param_names)
    
    return results


def get_parameter_ranking(Si: Dict, param_names: List[str]) -> List[Dict]:
    """按敏感性排序参数"""
    rankings = []
    for i, name in enumerate(param_names):
        rankings.append({
            'name': name,
            'S1': float(Si['S1'][i]),
            'ST': float(Si['ST'][i])
        })
        
    rankings.sort(key=lambda x: x['ST'], reverse=True)
    
    for i, item in enumerate(rankings):
        item['rank'] = i + 1
        
    return rankings


def extract_second_order(Si: Dict, param_names: List[str]) -> Dict:
    """提取二阶交互效应"""
    n = len(param_names)
    interactions = {}
    
    S2 = Si.get('S2', np.zeros((n, n)))
    S2_conf = Si.get('S2_conf', np.zeros((n, n)))
    
    for i in range(n):
        for j in range(i+1, n):
            key = f"{param_names[i]}_{param_names[j]}"
            interactions[key] = {
                'S2': float(S2[i, j]),
                'S2_conf': float(S2_conf[i, j])
            }
            
    return interactions


def interpret_results(results: Dict, param_names: List[str]) -> List[str]:
    """生成结果解释"""
    interpretations = []
    
    # 找出最重要的参数
    ranking = results['ranking']
    if ranking:
        top_param = ranking[0]
        interpretations.append(
            f"参数 {top_param['name']} 是最敏感的，"
            f"一阶敏感性指数为 {top_param['S1']:.3f}，"
            f"总效应为 {top_param['ST']:.3f}"
        )
        
    # 检查交互效应
    total_first_order = sum(r['S1'] for r in ranking)
    if total_first_order < 0.9:
        interpretations.append(
            f"一阶效应之和为 {total_first_order:.3f}，"
            "存在显著的参数交互效应"
        )
    else:
        interpretations.append(
            f"一阶效应之和为 {total_first_order:.3f}，"
            "参数交互效应较小，可以独立分析各参数"
        )
        
    # 识别不重要的参数
    unimportant = [r for r in ranking if r['ST'] < 0.05]
    if unimportant:
        names = [r['name'] for r in unimportant]
        interpretations.append(
            f"参数 {', '.join(names)} 的总效应小于5%，"
            "对模型输出影响较小"
        )
        
    return interpretations


def run_morris_screening(
    model: Callable,
    param_names: List[str],
    param_bounds: List[Tuple[float, float]],
    n_trajectories: int = 100,
    num_levels: int = 4
) -> Dict:
    """
    Morris筛选方法
    高效识别重要参数
    """
    try:
        from SALib.sample import morris as morris_sample
        from SALib.analyze import morris as morris_analyze
    except ImportError:
        raise ImportError("SALib not installed")
        
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': param_bounds
    }
    
    # 生成Morris样本
    samples = morris_sample.sample(problem, N=n_trajectories, num_levels=num_levels)
    
    # 计算模型输出
    Y = np.array([model(s) for s in samples])
    
    # Morris分析
    Si = morris_analyze.analyze(problem, samples, Y)
    
    return {
        'mu': dict(zip(param_names, Si['mu'])),
        'mu_star': dict(zip(param_names, Si['mu_star'])),
        'sigma': dict(zip(param_names, Si['sigma'])),
        'ranking': sorted(
            [{'name': n, 'mu_star': Si['mu_star'][i], 'sigma': Si['sigma'][i]}
             for i, n in enumerate(param_names)],
            key=lambda x: x['mu_star'],
            reverse=True
        )
    }


def run_monte_carlo(
    model: Callable,
    param_distributions: Dict,
    n_samples: int = 10000,
    confidence_level: float = 0.95
) -> Dict:
    """
    蒙特卡洛不确定性传播
    """
    samples = []
    
    for _ in range(n_samples):
        params = {}
        for name, dist in param_distributions.items():
            if hasattr(dist, 'rvs'):
                params[name] = dist.rvs()
            elif isinstance(dist, tuple):
                # (mean, std) -> normal distribution
                params[name] = np.random.normal(dist[0], dist[1])
            else:
                params[name] = dist
                
        try:
            output = model(np.array(list(params.values())))
            samples.append(output)
        except:
            continue
            
    samples = np.array(samples)
    alpha = 1 - confidence_level
    
    return {
        'n_samples': len(samples),
        'mean': float(np.mean(samples)),
        'std': float(np.std(samples)),
        'cv': float(np.std(samples) / np.mean(samples)) if np.mean(samples) != 0 else np.nan,
        'confidence_interval': {
            'lower': float(np.percentile(samples, alpha/2 * 100)),
            'upper': float(np.percentile(samples, (1-alpha/2) * 100))
        },
        'percentiles': {
            '5%': float(np.percentile(samples, 5)),
            '25%': float(np.percentile(samples, 25)),
            '50%': float(np.percentile(samples, 50)),
            '75%': float(np.percentile(samples, 75)),
            '95%': float(np.percentile(samples, 95))
        }
    }


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 定义测试模型
    def test_model(x):
        return x[0]**2 + 0.5*x[1]**2 + 0.1*x[0]*x[1] + 0.01*x[2]**2
    
    # 运行Sobol分析
    results = run_sobol_analysis(
        model=test_model,
        param_names=['x1', 'x2', 'x3'],
        param_bounds=[(0, 1), (0, 1), (0, 1)],
        n_samples=512
    )
    
    print("Sobol Analysis Results:")
    print(f"Ranking: {results['ranking']}")
    print(f"Interpretation: {results['interpretation']}")
