"""
求解器调度器
根据问题类型自动选择合适的求解器
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# 求解器优先级配置
SOLVER_PRIORITY = {
    'LP': ['scipy', 'pulp', 'cvxpy'],
    'MILP': ['pulp', 'ortools', 'scipy'],
    'NLP': ['scipy', 'nlopt'],
    'Convex': ['cvxpy', 'scipy'],
    'QP': ['cvxpy', 'scipy'],
}


class SolverError(Exception):
    """求解器错误"""
    pass


class SolverResult:
    """求解结果"""
    
    def __init__(
        self,
        status: str,
        objective_value: Optional[float] = None,
        solution: Optional[Dict] = None,
        solver_used: str = None,
        solve_time: float = 0,
        message: str = None,
        stats: Dict = None
    ):
        self.status = status
        self.objective_value = objective_value
        self.solution = solution or {}
        self.solver_used = solver_used
        self.solve_time = solve_time
        self.message = message
        self.stats = stats or {}
        
    def is_optimal(self) -> bool:
        return self.status in ['optimal', 'Optimal']
        
    def to_dict(self) -> Dict:
        return {
            'status': self.status,
            'objective_value': self.objective_value,
            'solution': self.solution,
            'solver_used': self.solver_used,
            'solve_time': self.solve_time,
            'message': self.message,
            'stats': self.stats
        }


def solve_lp_scipy(
    c: np.ndarray,
    A_ub: np.ndarray = None,
    b_ub: np.ndarray = None,
    A_eq: np.ndarray = None,
    b_eq: np.ndarray = None,
    bounds: List = None,
    **kwargs
) -> SolverResult:
    """
    使用SciPy求解线性规划
    
    Args:
        c: 目标函数系数
        A_ub: 不等式约束矩阵
        b_ub: 不等式约束右侧
        A_eq: 等式约束矩阵
        b_eq: 等式约束右侧
        bounds: 变量边界
        
    Returns:
        求解结果
    """
    from scipy.optimize import linprog
    
    start_time = time.time()
    
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs'
    )
    
    solve_time = time.time() - start_time
    
    if result.success:
        return SolverResult(
            status='optimal',
            objective_value=result.fun,
            solution={f'x_{i}': v for i, v in enumerate(result.x)},
            solver_used='scipy.linprog',
            solve_time=solve_time,
            message=result.message,
            stats={'iterations': result.nit}
        )
    else:
        return SolverResult(
            status='failed',
            solver_used='scipy.linprog',
            solve_time=solve_time,
            message=result.message
        )


def solve_lp_pulp(
    c: np.ndarray,
    A_ub: np.ndarray = None,
    b_ub: np.ndarray = None,
    A_eq: np.ndarray = None,
    b_eq: np.ndarray = None,
    bounds: List = None,
    var_types: List[str] = None,
    **kwargs
) -> SolverResult:
    """
    使用PuLP求解LP/MILP
    """
    try:
        from pulp import (
            LpProblem, LpMinimize, LpVariable, lpSum, value, LpStatus,
            LpContinuous, LpInteger, LpBinary
        )
    except ImportError:
        raise SolverError("PuLP not installed")
        
    start_time = time.time()
    
    n = len(c)
    
    # 创建问题
    prob = LpProblem("LP_Problem", LpMinimize)
    
    # 变量类型
    type_map = {
        'continuous': LpContinuous,
        'integer': LpInteger,
        'binary': LpBinary
    }
    
    if var_types is None:
        var_types = ['continuous'] * n
        
    # 创建变量
    x = []
    for i in range(n):
        lb = bounds[i][0] if bounds and bounds[i][0] is not None else 0
        ub = bounds[i][1] if bounds and len(bounds) > i else None
        vtype = type_map.get(var_types[i], LpContinuous)
        x.append(LpVariable(f'x_{i}', lowBound=lb, upBound=ub, cat=vtype))
        
    # 目标函数
    prob += lpSum([c[i] * x[i] for i in range(n)])
    
    # 不等式约束
    if A_ub is not None:
        for i, row in enumerate(A_ub):
            prob += lpSum([row[j] * x[j] for j in range(n)]) <= b_ub[i]
            
    # 等式约束
    if A_eq is not None:
        for i, row in enumerate(A_eq):
            prob += lpSum([row[j] * x[j] for j in range(n)]) == b_eq[i]
            
    # 求解
    prob.solve()
    
    solve_time = time.time() - start_time
    
    status = LpStatus[prob.status]
    
    if status == 'Optimal':
        return SolverResult(
            status='optimal',
            objective_value=value(prob.objective),
            solution={v.name: v.varValue for v in prob.variables()},
            solver_used='PuLP-CBC',
            solve_time=solve_time
        )
    else:
        return SolverResult(
            status=status.lower(),
            solver_used='PuLP-CBC',
            solve_time=solve_time,
            message=f"Problem status: {status}"
        )


def solve_nlp_scipy(
    objective: Callable,
    x0: np.ndarray,
    constraints: List[Dict] = None,
    bounds: List = None,
    method: str = 'SLSQP',
    **kwargs
) -> SolverResult:
    """
    使用SciPy求解非线性规划
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    result = minimize(
        objective,
        x0,
        method=method,
        bounds=bounds,
        constraints=constraints or [],
        options=kwargs.get('options', {})
    )
    
    solve_time = time.time() - start_time
    
    if result.success:
        return SolverResult(
            status='optimal',
            objective_value=result.fun,
            solution={f'x_{i}': v for i, v in enumerate(result.x)},
            solver_used=f'scipy.minimize-{method}',
            solve_time=solve_time,
            message=result.message,
            stats={'iterations': result.nit, 'function_evals': result.nfev}
        )
    else:
        return SolverResult(
            status='failed',
            objective_value=result.fun,
            solution={f'x_{i}': v for i, v in enumerate(result.x)},
            solver_used=f'scipy.minimize-{method}',
            solve_time=solve_time,
            message=result.message
        )


def solve_optimization(
    problem_type: str,
    **kwargs
) -> SolverResult:
    """
    通用优化求解入口
    
    Args:
        problem_type: 问题类型 ('LP', 'MILP', 'NLP', 'Convex')
        **kwargs: 问题参数
        
    Returns:
        求解结果
    """
    solvers = SOLVER_PRIORITY.get(problem_type, ['scipy'])
    
    for solver_name in solvers:
        try:
            if solver_name == 'scipy':
                if problem_type in ['LP']:
                    return solve_lp_scipy(**kwargs)
                elif problem_type == 'NLP':
                    return solve_nlp_scipy(**kwargs)
            elif solver_name == 'pulp':
                return solve_lp_pulp(**kwargs)
            # 可以添加更多求解器
        except Exception as e:
            logger.warning(f"Solver {solver_name} failed: {e}")
            continue
            
    raise SolverError(f"All solvers failed for {problem_type}")


def verify_solution(
    result: SolverResult,
    A_ub: np.ndarray = None,
    b_ub: np.ndarray = None,
    A_eq: np.ndarray = None,
    b_eq: np.ndarray = None,
    bounds: List = None,
    tol: float = 1e-6
) -> Dict:
    """
    验证求解结果
    """
    if not result.solution:
        return {'valid': False, 'reason': 'No solution'}
        
    x = np.array([result.solution.get(f'x_{i}', 0) for i in range(len(result.solution))])
    
    violations = []
    
    # 检查不等式约束
    if A_ub is not None:
        residuals = A_ub @ x - b_ub
        for i, r in enumerate(residuals):
            if r > tol:
                violations.append(f'Inequality constraint {i}: violation = {r}')
                
    # 检查等式约束
    if A_eq is not None:
        residuals = np.abs(A_eq @ x - b_eq)
        for i, r in enumerate(residuals):
            if r > tol:
                violations.append(f'Equality constraint {i}: violation = {r}')
                
    # 检查边界
    if bounds:
        for i, (lb, ub) in enumerate(bounds):
            if lb is not None and x[i] < lb - tol:
                violations.append(f'Variable x_{i} below lower bound')
            if ub is not None and x[i] > ub + tol:
                violations.append(f'Variable x_{i} above upper bound')
                
    return {
        'valid': len(violations) == 0,
        'violations': violations
    }


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试简单LP
    c = np.array([1, 2])
    A_ub = np.array([[1, 1], [2, 1]])
    b_ub = np.array([5, 8])
    bounds = [(0, None), (0, None)]
    
    result = solve_optimization(
        'LP',
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds
    )
    
    print(f"Status: {result.status}")
    print(f"Objective: {result.objective_value}")
    print(f"Solution: {result.solution}")
    print(f"Solve time: {result.solve_time:.4f}s")
