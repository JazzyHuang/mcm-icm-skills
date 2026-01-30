"""
引用多样性验证器
验证论文引用是否满足美赛O奖级别的多样性要求
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """验证配置"""
    min_total_citations: int = 8
    max_total_citations: int = 15
    min_categories: int = 4
    min_diversity_score: float = 0.75
    strict_mode: bool = True
    
    category_requirements: Dict[str, Dict] = field(default_factory=lambda: {
        'academic_papers': {'min': 3, 'weight': 0.40, 'required': False},
        'government_reports': {'min': 1, 'weight': 0.15, 'required': False},
        'official_data': {'min': 1, 'weight': 0.15, 'required': False},
        'problem_references': {'min': 1, 'weight': 0.15, 'required': True},
        'other_sources': {'min': 0, 'weight': 0.15, 'required': False}
    })


class CitationDiversityValidator:
    """引用多样性验证器"""
    
    # 类别标签映射
    CATEGORY_MAPPING = {
        'academic': 'academic_papers',
        'journal': 'academic_papers',
        'conference': 'academic_papers',
        'preprint': 'academic_papers',
        'arxiv': 'academic_papers',
        'government': 'government_reports',
        'report': 'government_reports',
        'techreport': 'government_reports',
        'whitepaper': 'government_reports',
        'data': 'official_data',
        'dataset': 'official_data',
        'database': 'official_data',
        'problem': 'problem_references',
        'mcm': 'problem_references',
        'icm': 'problem_references',
        'comap': 'problem_references',
        'media': 'other_sources',
        'news': 'other_sources',
        'technical': 'other_sources',
        'documentation': 'other_sources',
        'github': 'other_sources',
        'web': 'other_sources',
        'other': 'other_sources'
    }
    
    # 搜索建议
    SEARCH_SUGGESTIONS = {
        'academic_papers': [
            "[主题] peer-reviewed journal article",
            "[主题] systematic review meta-analysis",
            "[主题] mathematical modeling optimization study",
            "[主题] recent research advances 2024 2025"
        ],
        'government_reports': [
            "[主题] government report official statistics",
            "[主题] World Bank publication report",
            "[主题] UN United Nations report",
            "[主题] OECD policy analysis",
            "[主题] EPA CDC DOE official report"
        ],
        'official_data': [
            "确保data-collector已为获取的数据生成引用",
            "[主题] World Bank open data indicator",
            "[主题] UN statistics database",
            "[主题] official government statistics data"
        ],
        'problem_references': [
            "必须引用MCM/ICM官方题目声明",
            "如有数据文件，需引用官方提供的数据集",
            "使用problem-reference-extractor提取题目引用"
        ],
        'other_sources': [
            "[主题] industry report white paper",
            "[主题] technical documentation standard",
            "GitHub repository for methodology implementation"
        ]
    }
    
    def __init__(self, config: Optional[ValidationConfig] = None, config_path: Optional[str] = None):
        """
        初始化验证器
        
        Args:
            config: 验证配置对象
            config_path: 配置文件路径
        """
        if config:
            self.config = config
        elif config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = ValidationConfig()
    
    def _load_config(self, config_path: str) -> ValidationConfig:
        """从文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if data is None:
                logger.warning(f"Config file {config_path} is empty, using defaults")
                return ValidationConfig()
            
            return ValidationConfig(
                min_total_citations=data.get('validation_rules', {}).get('min_total_citations', 8),
                max_total_citations=data.get('validation_rules', {}).get('max_total_citations', 15),
                min_categories=data.get('validation_rules', {}).get('min_categories', 4),
                min_diversity_score=data.get('validation_rules', {}).get('min_diversity_score', 0.75),
                strict_mode=data.get('strict_mode', True),
                category_requirements=data.get('category_requirements', ValidationConfig().category_requirements)
            )
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return ValidationConfig()
        except yaml.YAMLError as e:
            logger.warning(f"Invalid YAML in config file {config_path}: {e}, using defaults")
            return ValidationConfig()
        except (IOError, OSError) as e:
            logger.warning(f"Failed to read config file {config_path}: {e}, using defaults")
            return ValidationConfig()
        except Exception as e:
            logger.warning(f"Unexpected error loading config from {config_path}: {e}, using defaults")
            return ValidationConfig()
    
    def validate(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证引用多样性
        
        Args:
            citations: 引用列表，每个引用需包含bibtex_key和category字段
            
        Returns:
            验证结果字典
        """
        # 1. 分类统计
        categorized = self._categorize_citations(citations)
        
        # 2. 计算各类别统计
        category_details = self._calculate_category_details(categorized)
        
        # 3. 计算多样性评分
        diversity_score = self._calculate_diversity_score(category_details)
        
        # 4. 检查是否通过
        validation_result = self._check_validation(citations, category_details, diversity_score)
        
        # 5. 生成建议和警告
        recommendations, warnings = self._generate_recommendations(
            citations, category_details, diversity_score
        )
        
        return {
            'validation_result': validation_result,
            'category_details': category_details,
            'recommendations': recommendations,
            'warnings': warnings
        }
    
    def _categorize_citations(self, citations: List[Dict]) -> Dict[str, List[Dict]]:
        """将引用按类别分组"""
        categorized = {
            'academic_papers': [],
            'government_reports': [],
            'official_data': [],
            'problem_references': [],
            'other_sources': []
        }
        
        for citation in citations:
            category_raw = citation.get('category', 'other').lower()
            category = self.CATEGORY_MAPPING.get(category_raw, 'other_sources')
            categorized[category].append(citation)
        
        return categorized
    
    def _calculate_category_details(
        self,
        categorized: Dict[str, List[Dict]]
    ) -> Dict[str, Dict[str, Any]]:
        """计算每个类别的详细信息"""
        details = {}
        
        for category, citations in categorized.items():
            req = self.config.category_requirements.get(category, {'min': 0, 'weight': 0})
            count = len(citations)
            required = req.get('min', 0)
            
            details[category] = {
                'count': count,
                'required': required,
                'status': 'pass' if count >= required else 'fail',
                'citations': [c.get('bibtex_key', 'unknown') for c in citations],
                'weight': req.get('weight', 0),
                'is_required': req.get('required', False)
            }
        
        return details
    
    def _calculate_diversity_score(self, category_details: Dict[str, Dict]) -> float:
        """计算多样性评分"""
        score = 0.0
        
        for category, details in category_details.items():
            weight = details.get('weight', 0)
            count = details.get('count', 0)
            required = details.get('required', 0)
            
            # 基础分：有该类别引用即得分
            if count > 0:
                score += weight
                
                # 奖励分：超过最低要求额外加分
                if count > required:
                    score += weight * 0.2
        
        return min(round(score, 2), 1.0)
    
    def _check_validation(
        self,
        citations: List[Dict],
        category_details: Dict[str, Dict],
        diversity_score: float
    ) -> Dict[str, Any]:
        """检查是否通过验证"""
        total_citations = len(citations)
        categories_covered = sum(1 for d in category_details.values() if d['count'] > 0)
        
        # 检查各项条件
        checks = {
            'total_citations_ok': self.config.min_total_citations <= total_citations <= self.config.max_total_citations,
            'categories_ok': categories_covered >= self.config.min_categories,
            'diversity_ok': diversity_score >= self.config.min_diversity_score,
            'required_categories_ok': all(
                details['count'] >= details['required']
                for details in category_details.values()
                if details.get('is_required', False)
            )
        }
        
        # 严格模式下所有最低要求必须满足
        if self.config.strict_mode:
            checks['all_minimums_ok'] = all(
                details['count'] >= details['required']
                for details in category_details.values()
            )
        else:
            checks['all_minimums_ok'] = True
        
        overall_status = 'pass' if all(checks.values()) else 'fail'
        
        return {
            'overall_status': overall_status,
            'diversity_score': diversity_score,
            'categories_covered': categories_covered,
            'total_citations': total_citations,
            'checks': checks
        }
    
    def _generate_recommendations(
        self,
        citations: List[Dict],
        category_details: Dict[str, Dict],
        diversity_score: float
    ) -> Tuple[List[Dict], List[str]]:
        """生成改进建议和警告"""
        recommendations = []
        warnings = []
        
        total_citations = len(citations)
        categories_covered = sum(1 for d in category_details.values() if d['count'] > 0)
        
        # 检查总引用数
        if total_citations < self.config.min_total_citations:
            recommendations.append({
                'priority': 'medium',
                'category': None,
                'message': f"当前总引用数为{total_citations}，建议增加到{self.config.min_total_citations}-{self.config.max_total_citations}篇。",
                'search_suggestions': ["使用ai-deep-search-guide进行更多搜索"]
            })
        elif total_citations > self.config.max_total_citations:
            warnings.append(f"引用数量({total_citations})超过建议上限({self.config.max_total_citations})，考虑精简")
        
        # 检查各类别
        for category, details in category_details.items():
            if details['status'] == 'fail':
                priority = 'high' if details.get('is_required', False) else 'medium'
                category_name = category.replace('_', ' ').title()
                
                recommendations.append({
                    'priority': priority,
                    'category': category,
                    'message': f"需要至少{details['required']}个{category_name}引用，当前只有{details['count']}个。",
                    'search_suggestions': self.SEARCH_SUGGESTIONS.get(category, [])
                })
        
        # 检查多样性评分
        if diversity_score < self.config.min_diversity_score:
            warnings.append(
                f"多样性评分({diversity_score})低于最低要求({self.config.min_diversity_score})"
            )
        
        # 检查类别覆盖
        if categories_covered < self.config.min_categories:
            warnings.append(
                f"仅覆盖{categories_covered}个类别，需要至少{self.config.min_categories}个类别"
            )
        
        # 按优先级排序建议
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 99))
        
        return recommendations, warnings
    
    def get_improvement_plan(self, validation_result: Dict) -> str:
        """
        生成改进计划的文本描述
        
        Args:
            validation_result: validate()方法的输出
            
        Returns:
            改进计划文本
        """
        if validation_result['validation_result']['overall_status'] == 'pass':
            return "引用多样性验证通过，无需改进。"
        
        lines = ["## 引用多样性改进计划\n"]
        
        # 添加当前状态
        result = validation_result['validation_result']
        lines.append(f"当前状态: 多样性评分 {result['diversity_score']}, "
                    f"覆盖 {result['categories_covered']} 个类别, "
                    f"共 {result['total_citations']} 个引用\n")
        
        # 添加警告
        if validation_result['warnings']:
            lines.append("### 警告")
            for warning in validation_result['warnings']:
                lines.append(f"- {warning}")
            lines.append("")
        
        # 添加建议
        if validation_result['recommendations']:
            lines.append("### 改进步骤")
            for i, rec in enumerate(validation_result['recommendations'], 1):
                priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡"
                lines.append(f"\n{i}. {priority_emoji} {rec['message']}")
                
                if rec.get('search_suggestions'):
                    lines.append("   搜索建议:")
                    for suggestion in rec['search_suggestions']:
                        lines.append(f"   - {suggestion}")
        
        return '\n'.join(lines)


def validate_citations_diversity(
    citations: List[Dict[str, Any]],
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数：验证引用多样性
    
    Args:
        citations: 引用列表
        config_path: 配置文件路径（可选）
        
    Returns:
        验证结果
    """
    validator = CitationDiversityValidator(config_path=config_path)
    return validator.validate(citations)


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试引用列表
    test_citations = [
        {'bibtex_key': 'smith2024optimization', 'category': 'academic'},
        {'bibtex_key': 'jones2023model', 'category': 'academic'},
        {'bibtex_key': 'chen2024analysis', 'category': 'academic'},
        {'bibtex_key': 'worldbank2024report', 'category': 'government'},
        {'bibtex_key': 'un2024data', 'category': 'data'},
        {'bibtex_key': 'mcm2024problema', 'category': 'problem'},
    ]
    
    validator = CitationDiversityValidator()
    result = validator.validate(test_citations)
    
    print("Validation Result:")
    print(f"Status: {result['validation_result']['overall_status']}")
    print(f"Diversity Score: {result['validation_result']['diversity_score']}")
    print(f"Categories Covered: {result['validation_result']['categories_covered']}")
    
    print("\nCategory Details:")
    for category, details in result['category_details'].items():
        print(f"  {category}: {details['count']}/{details['required']} ({details['status']})")
    
    if result['recommendations']:
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  [{rec['priority']}] {rec['message']}")
    
    if result['warnings']:
        print("\nWarnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    # 打印改进计划
    print("\n" + "="*50)
    print(validator.get_improvement_plan(result))
