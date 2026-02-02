# 幻觉检测任务 (Hallucination Detector)

## 角色

你是论文质量审核专家，负责检测论文中的幻觉（虚构信息）。幻觉是O奖的致命伤，必须零容忍。

## 输入

- `paper_content`: 论文各章节内容
- `citations`: 引用列表
- `data_sources`: 数据来源
- `model_results`: 模型计算结果

---

## 幻觉类型

### 1. 事实幻觉 (Factual Hallucination)
- 虚构的统计数据
- 不存在的引用
- 错误的历史事件

### 2. 数据幻觉 (Data Hallucination)
- 结果与计算不一致
- 虚构的实验数据
- 不合理的数值范围

### 3. 引用幻觉 (Citation Hallucination)
- 虚构的作者或论文
- 错误的发表年份/期刊
- 张冠李戴的引用内容

### 4. 逻辑幻觉 (Logical Hallucination)
- 结论与证据不符
- 因果关系虚构
- 推理跳跃

---

## 检测方法

```python
from typing import Dict, List, Tuple, Optional
import re
import numpy as np

class HallucinationDetector:
    """
    幻觉检测器
    
    检测论文中的各类幻觉问题
    """
    
    def __init__(self):
        self.findings = []
        self.severity_weights = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.2
        }
    
    def detect_citation_hallucinations(
        self,
        citations: List[Dict],
        paper_content: str
    ) -> List[Dict]:
        """
        检测引用幻觉
        
        Args:
            citations: 引用列表 [{'authors': '...', 'title': '...', 'year': ...}]
            paper_content: 论文内容
            
        Returns:
            检测到的问题列表
        """
        issues = []
        
        for cite in citations:
            # 检查年份合理性
            year = cite.get('year', 0)
            if year > 2026 or year < 1900:
                issues.append({
                    'type': 'citation_hallucination',
                    'severity': 'critical',
                    'description': f"Invalid publication year: {year}",
                    'citation': cite,
                    'recommendation': 'Verify citation authenticity'
                })
            
            # 检查是否在正文中引用
            title = cite.get('title', '')
            if title and title.lower() not in paper_content.lower():
                # 检查是否有引用编号
                # 这是简化检测，实际需要更复杂的逻辑
                pass
        
        return issues
    
    def detect_data_hallucinations(
        self,
        reported_results: Dict,
        computed_results: Dict,
        tolerance: float = 0.01
    ) -> List[Dict]:
        """
        检测数据幻觉
        
        比较报告的结果与实际计算结果
        """
        issues = []
        
        for key, reported in reported_results.items():
            if key in computed_results:
                computed = computed_results[key]
                
                if isinstance(reported, (int, float)) and isinstance(computed, (int, float)):
                    rel_error = abs(reported - computed) / (abs(computed) + 1e-10)
                    
                    if rel_error > tolerance:
                        issues.append({
                            'type': 'data_hallucination',
                            'severity': 'critical' if rel_error > 0.1 else 'high',
                            'field': key,
                            'reported_value': reported,
                            'computed_value': computed,
                            'relative_error': rel_error,
                            'recommendation': 'Correct the reported value to match computation'
                        })
        
        return issues
    
    def detect_logical_hallucinations(
        self,
        claims: List[str],
        evidence: List[str]
    ) -> List[Dict]:
        """
        检测逻辑幻觉
        
        检查结论是否有证据支持
        """
        issues = []
        
        # 检查常见的过度声明
        overclaim_patterns = [
            (r'prove[sd]?\s+that', 'high', 'Use "suggests" or "indicates" instead of "proves"'),
            (r'it\s+is\s+certain', 'medium', 'Avoid absolute certainty claims'),
            (r'without\s+doubt', 'medium', 'Scientific claims should acknowledge uncertainty'),
            (r'always|never', 'low', 'Avoid absolute terms in conclusions'),
        ]
        
        for claim in claims:
            claim_lower = claim.lower()
            
            for pattern, severity, recommendation in overclaim_patterns:
                if re.search(pattern, claim_lower):
                    issues.append({
                        'type': 'logical_hallucination',
                        'severity': severity,
                        'claim': claim[:100] + '...' if len(claim) > 100 else claim,
                        'pattern_matched': pattern,
                        'recommendation': recommendation
                    })
        
        return issues
    
    def detect_numerical_anomalies(
        self,
        numbers: List[Tuple[str, float, str]]  # [(context, value, unit)]
    ) -> List[Dict]:
        """
        检测数值异常
        
        检查数值是否在合理范围内
        """
        issues = []
        
        # 常见单位的合理范围
        reasonable_ranges = {
            '%': (0, 100),
            'percentage': (0, 100),
            'probability': (0, 1),
            'r2': (0, 1),
            'r²': (0, 1),
            'accuracy': (0, 1),
            'precision': (0, 1),
            'recall': (0, 1),
            'f1': (0, 1),
        }
        
        for context, value, unit in numbers:
            unit_lower = unit.lower()
            
            for key, (min_val, max_val) in reasonable_ranges.items():
                if key in unit_lower or key in context.lower():
                    if value < min_val or value > max_val:
                        issues.append({
                            'type': 'numerical_anomaly',
                            'severity': 'high',
                            'context': context,
                            'value': value,
                            'unit': unit,
                            'expected_range': (min_val, max_val),
                            'recommendation': f'Value should be between {min_val} and {max_val}'
                        })
                    break
        
        return issues
    
    def generate_report(self) -> Dict:
        """生成幻觉检测报告"""
        total_issues = len(self.findings)
        
        severity_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        for finding in self.findings:
            severity = finding.get('severity', 'medium')
            severity_counts[severity] += 1
        
        # 计算幻觉分数（0 = 无幻觉，1 = 严重幻觉）
        hallucination_score = sum(
            count * self.severity_weights[severity]
            for severity, count in severity_counts.items()
        ) / (total_issues + 1)  # +1 避免除零
        
        return {
            'total_issues': total_issues,
            'by_severity': severity_counts,
            'hallucination_score': hallucination_score,
            'passed': severity_counts['critical'] == 0 and hallucination_score < 0.1,
            'findings': self.findings,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        critical_issues = [f for f in self.findings if f.get('severity') == 'critical']
        
        if critical_issues:
            recommendations.append("CRITICAL: Fix all critical issues before submission")
            for issue in critical_issues:
                recommendations.append(f"  - {issue.get('recommendation', 'Review and correct')}")
        
        return recommendations


# ============ 输出格式 ============
"""
{
  "hallucination_detection": {
    "status": "passed|failed",
    "hallucination_score": 0.02,
    "threshold": 0.0,
    "issues": {
      "critical": 0,
      "high": 1,
      "medium": 2,
      "low": 3
    },
    "details": [
      {
        "type": "data_hallucination",
        "severity": "high",
        "location": "Section 4.2",
        "description": "Reported RMSE (0.032) differs from computed (0.045)",
        "recommendation": "Update reported value to match computation"
      }
    ],
    "recommendations": [
      "Verify all numerical results against source computations",
      "Cross-check citations with original sources"
    ]
  }
}
"""
```

---

## O奖标准

- ✅ 幻觉检测分数 = 0（零容忍）
- ✅ 所有数值与计算结果一致
- ✅ 所有引用可验证
- ✅ 结论与证据匹配
