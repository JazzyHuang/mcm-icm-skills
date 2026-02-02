---
name: code-verifier
description: 验证生成代码的正确性。实现四层验证机制：语法验证、类型验证、数值验证、符号验证。基于VerifiAgent和SymCode框架，确保代码可靠执行。
---

# 代码验证器 (Code Verifier)

## 功能概述

验证自动生成代码的正确性，确保代码能够可靠执行并产生正确结果。

## 四层验证机制

### 第1层：语法验证 (Syntax Verification)
- 代码能否被Python解析器解析
- 是否有语法错误
- 导入语句是否正确

```python
import ast

def verify_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        return False, str(e)
```

### 第2层：类型验证 (Type Verification)
- 函数参数类型是否正确
- 返回值类型是否符合预期
- 使用mypy进行静态类型检查

```python
from mypy import api

def verify_types(code_file: str) -> Dict:
    result = api.run([code_file])
    return {
        'passed': result[2] == 0,
        'stdout': result[0],
        'stderr': result[1]
    }
```

### 第3层：数值验证 (Numerical Verification)
- 使用简单测试用例验证
- 检查数值范围合理性
- 检查边界条件

```python
def verify_numerical(func, test_cases: List[Dict]) -> Dict:
    results = []
    for case in test_cases:
        try:
            output = func(**case['input'])
            passed = case['check'](output)
            results.append({'passed': passed, 'output': output})
        except Exception as e:
            results.append({'passed': False, 'error': str(e)})
    return results
```

### 第4层：符号验证 (Symbolic Verification)
- 使用SymPy进行符号计算验证
- 验证数学推导正确性
- 检查约束满足性

```python
import sympy as sp

def verify_symbolic(expression, expected):
    expr = sp.sympify(expression)
    expected_expr = sp.sympify(expected)
    return sp.simplify(expr - expected_expr) == 0
```

## 输出格式

```json
{
  "verification_result": {
    "overall_status": "passed",
    "layers": {
      "syntax": {
        "passed": true,
        "errors": []
      },
      "type": {
        "passed": true,
        "warnings": []
      },
      "numerical": {
        "passed": true,
        "test_cases": [
          {"name": "basic_test", "passed": true},
          {"name": "edge_case", "passed": true}
        ]
      },
      "symbolic": {
        "passed": true,
        "verified_equations": [
          "objective function derivation"
        ]
      }
    },
    "code_quality": {
      "complexity": "low",
      "maintainability": "high",
      "test_coverage": 0.85
    }
  },
  "suggestions": []
}
```

## 自动修复

当验证失败时，尝试自动修复：

### 语法修复
- 缺失括号/引号
- 缩进错误
- 常见拼写错误

### 类型修复
- 类型转换
- 默认值添加
- 类型注解补充

### 数值修复
- 边界值调整
- 精度问题处理
- 数值稳定性改进

## 相关技能

- `model-builder` - 模型构建
- `model-solver` - 模型求解
