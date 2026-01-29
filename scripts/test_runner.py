"""
测试运行器
用于测试MCM/ICM自动化系统
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import MCMOrchestrator, run_mcm_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_problem(problem_id: str) -> dict:
    """
    加载测试问题
    
    Args:
        problem_id: 问题ID，如 "2024A"
        
    Returns:
        问题数据
    """
    test_dir = Path(__file__).parent.parent / 'tests' / 'historical_problems'
    
    # 解析问题ID
    year = problem_id[:4]
    problem_type = problem_id[4:]
    
    problem_file = test_dir / year / f'{problem_type}.json'
    
    if not problem_file.exists():
        # 创建示例问题
        logger.warning(f"Problem file not found: {problem_file}")
        return create_sample_problem(problem_type)
        
    with open(problem_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_sample_problem(problem_type: str) -> dict:
    """创建示例问题"""
    return {
        'problem_type': problem_type,
        'problem_text': f'''
        This is a sample {problem_type}-type problem for testing.
        
        Background: [Sample background text]
        
        Your team should:
        1. Develop a model to address the problem
        2. Analyze your results
        3. Write a memo summarizing your recommendations
        ''',
        'team_control_number': '12345',
        'data_files': []
    }


async def run_test(problem_id: str, verbose: bool = False):
    """
    运行测试
    
    Args:
        problem_id: 问题ID
        verbose: 是否详细输出
    """
    logger.info(f"Starting test for problem: {problem_id}")
    
    # 加载问题
    problem = load_test_problem(problem_id)
    
    # 运行流水线
    start_time = datetime.now()
    
    try:
        result = await run_mcm_pipeline(
            problem_text=problem['problem_text'],
            problem_type=problem['problem_type'],
            team_control_number=problem.get('team_control_number', '12345'),
            data_files=problem.get('data_files', [])
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # 输出结果
        logger.info(f"Test completed in {duration:.2f}s")
        logger.info(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}")
        
        if verbose:
            print(json.dumps(result, indent=2, default=str))
            
        return result
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


def run_all_tests():
    """运行所有测试"""
    test_problems = [
        '2024A', '2024B', '2024C',
        '2024D', '2024E', '2024F',
    ]
    
    results = {}
    
    for problem_id in test_problems:
        try:
            result = asyncio.run(run_test(problem_id))
            results[problem_id] = {
                'success': result['success'],
                'duration': result.get('total_duration_seconds', 0)
            }
        except Exception as e:
            results[problem_id] = {
                'success': False,
                'error': str(e)
            }
            
    # 输出总结
    print("\n=== Test Summary ===")
    for problem_id, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"{status} {problem_id}: {result}")
        
    return results


def main():
    parser = argparse.ArgumentParser(description='MCM/ICM Test Runner')
    parser.add_argument('--problem', '-p', type=str, help='Problem ID (e.g., 2024A)')
    parser.add_argument('--all', '-a', action='store_true', help='Run all tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_tests()
    elif args.problem:
        asyncio.run(run_test(args.problem, args.verbose))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
