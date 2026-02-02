"""
输出验证器模块
确保Skills输出包含质量门禁所需的标准字段
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """质量指标"""
    word_count: int = 0
    quantification_count: int = 0
    depth_score: float = 0.0
    hook_quality: float = 0.0
    chinglish_score: float = 0.0
    innovation_score: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class OutputValidator:
    """
    输出验证器
    
    验证并增强skill输出，确保包含质量门禁所需的字段
    """
    
    # 深度分析标记词
    DEPTH_MARKERS = [
        'because', 'therefore', 'indicates', 'demonstrates',
        'reveals', 'suggests', 'contributes', 'impacts',
        'consequently', 'thus', 'hence', 'implies',
        'correlates', 'explains', 'accounts for'
    ]
    
    # 中式英语标记
    CHINGLISH_PATTERNS = [
        r'with the development of',
        r'in recent years',
        r'nowadays',
        r'as we all know',
        r'it is well known that',
        r'more and more',
        r'plays an? important role',
        r'has great influence',
        r'make a contribution',
        r'put forward',
    ]
    
    # 各阶段必需的输出字段
    REQUIRED_FIELDS = {
        1: ['references', 'citation_count', 'diversity_score'],
        2: ['sub_problems', 'assumptions', 'variables'],
        3: ['selected_model', 'innovation_score', 'model_code'],
        4: ['sensitivity_indices', 'robustness_score'],
        5: ['sections', 'word_count', 'abstract_score'],
        6: ['figures', 'figure_count'],
        7: ['latex_output', 'compilation_status'],
        8: ['quality_scores', 'hallucination_score', 'consistency_score'],
        9: ['polished_content'],
        10: ['submission_ready', 'checklist_status'],
    }
    
    def __init__(self):
        self.validation_history = []
    
    def validate_and_enhance(
        self,
        skill_name: str,
        phase: int,
        output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        验证并增强skill输出
        
        Args:
            skill_name: 技能名称
            phase: 所属阶段
            output: 原始输出
            
        Returns:
            增强后的输出（包含质量门禁字段）
        """
        enhanced_output = output.copy()
        
        # 添加通用质量指标
        if 'quality_metrics' not in enhanced_output:
            enhanced_output['quality_metrics'] = {}
        
        # 如果有文本内容，计算文本质量指标
        text_content = self._extract_text_content(output)
        if text_content:
            metrics = self._compute_text_metrics(text_content)
            enhanced_output['quality_metrics'].update({
                'word_count': metrics.word_count,
                'quantification_count': metrics.quantification_count,
                'depth_score': metrics.depth_score,
                'chinglish_score': metrics.chinglish_score,
            })
        
        # 添加阶段特定的字段
        required_fields = self.REQUIRED_FIELDS.get(phase, [])
        missing_fields = [f for f in required_fields if f not in enhanced_output]
        
        if missing_fields:
            enhanced_output['_missing_fields'] = missing_fields
            logger.warning(f"Skill '{skill_name}' missing required fields: {missing_fields}")
        
        # 验证记录
        validation_result = {
            'skill_name': skill_name,
            'phase': phase,
            'missing_fields': missing_fields,
            'has_quality_metrics': 'quality_metrics' in enhanced_output,
        }
        self.validation_history.append(validation_result)
        
        return enhanced_output
    
    def _extract_text_content(self, output: Dict[str, Any]) -> str:
        """从输出中提取文本内容"""
        text_parts = []
        
        # 常见的文本字段名
        text_fields = [
            'content', 'text', 'abstract', 'section_content',
            'response', 'output', 'generated_text', 'description'
        ]
        
        for field_name in text_fields:
            if field_name in output:
                value = output[field_name]
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, dict) and 'text' in value:
                    text_parts.append(value['text'])
        
        # 递归提取嵌套内容
        if 'sections' in output:
            for section in output['sections'].values() if isinstance(output['sections'], dict) else output['sections']:
                if isinstance(section, dict) and 'content' in section:
                    text_parts.append(section['content'])
        
        return ' '.join(text_parts)
    
    def _compute_text_metrics(self, text: str) -> QualityMetrics:
        """计算文本质量指标"""
        metrics = QualityMetrics()
        
        # 词数
        words = text.split()
        metrics.word_count = len(words)
        
        # 量化表述数量（数字出现次数）
        numbers = re.findall(r'\d+\.?\d*%?', text)
        metrics.quantification_count = len(numbers)
        
        # 深度分析得分（深度标记词出现频率）
        text_lower = text.lower()
        depth_count = sum(
            1 for marker in self.DEPTH_MARKERS
            if marker in text_lower
        )
        metrics.depth_score = min(1.0, depth_count / 5)  # 5个以上得满分
        
        # 中式英语得分（越低越好）
        chinglish_count = sum(
            1 for pattern in self.CHINGLISH_PATTERNS
            if re.search(pattern, text_lower)
        )
        metrics.chinglish_score = min(1.0, chinglish_count / 3)  # 3个以上得1.0
        
        return metrics
    
    def validate_abstract(self, abstract: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证摘要输出
        
        Args:
            abstract: 摘要输出
            
        Returns:
            验证结果
        """
        text = abstract.get('text', abstract.get('abstract', ''))
        
        validation = {
            'word_count': len(text.split()),
            'word_count_valid': 300 <= len(text.split()) <= 500,
        }
        
        # 检查Hook句质量
        first_sentence = text.split('.')[0] if text else ''
        validation['hook_word_count'] = len(first_sentence.split())
        validation['hook_has_number'] = bool(re.search(r'\d', first_sentence))
        validation['hook_no_cliche'] = not any(
            re.search(pattern, first_sentence.lower())
            for pattern in self.CHINGLISH_PATTERNS[:5]  # 检查最常见的陈词滥调
        )
        
        # 计算Hook质量分数
        hook_score = 0.0
        if validation['hook_no_cliche']:
            hook_score += 0.4
        if validation['hook_has_number']:
            hook_score += 0.3
        if 20 <= validation['hook_word_count'] <= 40:
            hook_score += 0.3
        
        validation['hook_quality'] = hook_score
        
        # 检查量化密度
        numbers = re.findall(r'\d+\.?\d*%?', text)
        validation['quantification_density'] = min(1.0, len(numbers) / 6)
        
        # 计算总分
        validation['abstract_score'] = (
            0.3 * validation['hook_quality'] +
            0.3 * validation['quantification_density'] +
            0.2 * (1 if validation['word_count_valid'] else 0.5) +
            0.2 * (1 - self._compute_text_metrics(text).chinglish_score)
        )
        
        return validation
    
    def validate_model_output(self, model_output: Dict[str, Any]) -> Dict[str, Any]:
        """验证建模输出"""
        validation = {}
        
        # 检查创新性评分
        innovation_score = model_output.get('innovation_score', 0)
        validation['innovation_score'] = innovation_score
        validation['innovation_sufficient'] = innovation_score >= 0.70
        
        # 检查是否有模型选择论证
        has_justification = 'justification' in model_output or 'comparison' in model_output
        validation['has_justification'] = has_justification
        
        # 检查是否有代码
        has_code = 'code' in model_output or 'implementation' in model_output
        validation['has_code'] = has_code
        
        return validation
    
    def validate_sensitivity_output(self, sensitivity_output: Dict[str, Any]) -> Dict[str, Any]:
        """验证敏感性分析输出"""
        validation = {}
        
        # 检查是否使用全局方法（Sobol）
        method = sensitivity_output.get('method', '').lower()
        validation['uses_global_method'] = 'sobol' in method or 'morris' in method
        
        # 检查是否有参数排序
        has_ranking = 'ranking' in sensitivity_output or 'parameter_ranking' in sensitivity_output
        validation['has_ranking'] = has_ranking
        
        # 检查是否有鲁棒性结论
        has_robustness = 'robustness' in sensitivity_output or 'robustness_score' in sensitivity_output
        validation['has_robustness'] = has_robustness
        
        return validation
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """生成验证报告"""
        total = len(self.validation_history)
        with_metrics = sum(1 for v in self.validation_history if v['has_quality_metrics'])
        missing_any = sum(1 for v in self.validation_history if v['missing_fields'])
        
        return {
            'total_validations': total,
            'with_quality_metrics': with_metrics,
            'with_missing_fields': missing_any,
            'compliance_rate': with_metrics / total if total > 0 else 0,
            'details': self.validation_history
        }


# 创建全局实例
output_validator = OutputValidator()


def validate_skill_output(
    skill_name: str,
    phase: int,
    output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    便捷函数：验证并增强skill输出
    """
    return output_validator.validate_and_enhance(skill_name, phase, output)


def get_validation_report() -> Dict[str, Any]:
    """获取验证报告"""
    return output_validator.generate_validation_report()
