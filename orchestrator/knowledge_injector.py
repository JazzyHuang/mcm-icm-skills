"""
知识库注入器模块
负责将model_database.json和o_award_features.json的知识注入到skill prompts中
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KnowledgeInjector:
    """
    知识库注入器
    
    从知识库中提取相关知识并注入到skill prompts中
    """
    
    def __init__(self, knowledge_base_dir: Optional[Path] = None):
        """
        初始化知识注入器
        
        Args:
            knowledge_base_dir: 知识库目录路径
        """
        if knowledge_base_dir is None:
            # 默认路径
            knowledge_base_dir = Path(__file__).parent.parent / "knowledge_base"
        
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.model_database = {}
        self.o_award_features = {}
        self.writing_guidelines = ""
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """加载知识库文件"""
        try:
            # 加载模型数据库
            model_db_path = self.knowledge_base_dir / "model_database.json"
            if model_db_path.exists():
                with open(model_db_path, 'r', encoding='utf-8') as f:
                    self.model_database = json.load(f)
                logger.info(f"Loaded model_database.json ({len(self.model_database)} categories)")
            
            # 加载O奖特征
            o_award_path = self.knowledge_base_dir / "o_award_features.json"
            if o_award_path.exists():
                with open(o_award_path, 'r', encoding='utf-8') as f:
                    self.o_award_features = json.load(f)
                logger.info("Loaded o_award_features.json")
            
            # 加载写作指南
            writing_path = self.knowledge_base_dir / "writing_guidelines.md"
            if writing_path.exists():
                with open(writing_path, 'r', encoding='utf-8') as f:
                    self.writing_guidelines = f.read()
                logger.info("Loaded writing_guidelines.md")
                
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
    
    def get_models_for_problem_type(self, problem_type: str) -> Dict[str, Any]:
        """
        获取特定问题类型的推荐模型
        
        Args:
            problem_type: 问题类型 (A-F)
            
        Returns:
            包含primary和innovative模型的字典
        """
        recommendations = self.model_database.get(
            "problem_type_recommendations", {}
        ).get(problem_type, {})
        
        result = {
            "primary_methods": recommendations.get("primary", []),
            "innovative_methods": recommendations.get("innovative", []),
            "visualization_types": recommendations.get("visualization", []),
            "sensitivity_methods": recommendations.get("sensitivity", [])
        }
        
        # 获取详细的模型信息
        result["model_details"] = {}
        for method in result["primary_methods"] + result["innovative_methods"]:
            model_info = self._find_model_info(method)
            if model_info:
                result["model_details"][method] = model_info
        
        return result
    
    def _find_model_info(self, model_name: str) -> Optional[Dict]:
        """在模型数据库中查找模型信息"""
        models = self.model_database.get("models", {})
        
        # 遍历所有分类查找模型
        for category, subcategories in models.items():
            if isinstance(subcategories, dict):
                # 检查是否直接是模型
                if "full_name" in subcategories:
                    if category.lower() == model_name.lower():
                        return subcategories
                # 检查子分类
                for subcat, items in subcategories.items():
                    if isinstance(items, dict):
                        if "full_name" in items:
                            if subcat.lower() == model_name.lower():
                                return items
                        else:
                            for item_name, item_data in items.items():
                                if item_name.lower() == model_name.lower():
                                    return item_data
        
        return None
    
    def get_high_innovation_combinations(
        self, 
        problem_type: Optional[str] = None
    ) -> List[Dict]:
        """
        获取高创新组合
        
        Args:
            problem_type: 可选，筛选特定问题类型
            
        Returns:
            高创新组合列表
        """
        combinations = self.model_database.get(
            "innovation_guidelines", {}
        ).get("high_innovation_combinations", [])
        
        if problem_type:
            combinations = [
                c for c in combinations 
                if problem_type in c.get("problem_types", [])
            ]
        
        return combinations
    
    def get_o_award_criteria(self) -> Dict[str, Any]:
        """获取O奖评分标准"""
        return self.o_award_features.get("o_award_criteria", {})
    
    def get_checklist(self, checklist_type: str) -> Dict[str, Any]:
        """
        获取特定类型的检查清单
        
        Args:
            checklist_type: abstract, model, sensitivity, writing
            
        Returns:
            检查清单内容
        """
        checklist_key = f"{checklist_type}_checklist"
        return self.o_award_features.get(checklist_key, {})
    
    def get_common_errors(self, category: str) -> List[Dict]:
        """
        获取特定类别的常见错误
        
        Args:
            category: abstract, model, validation, writing, visualization
            
        Returns:
            错误列表
        """
        errors = self.o_award_features.get("common_error_patterns", {})
        category_errors = errors.get(f"{category}_errors", {})
        return category_errors.get("errors", [])
    
    def get_citation_requirements(self) -> Dict[str, Any]:
        """获取引用要求"""
        return self.o_award_features.get("citation_requirements", {})
    
    def get_quality_thresholds(self) -> Dict[str, Any]:
        """获取质量门禁阈值"""
        return self.o_award_features.get("quality_thresholds", {})
    
    def get_downgrade_risks(self, severity: str = "all") -> Dict[str, Any]:
        """
        获取降级风险因素
        
        Args:
            severity: critical, high, medium, or all
            
        Returns:
            风险因素
        """
        risks = self.o_award_features.get("downgrade_risk_factors", {})
        
        if severity == "all":
            return risks
        
        return risks.get(f"{severity}_risks", {})
    
    def inject_for_skill(
        self, 
        skill_name: str, 
        problem_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        为特定skill注入所需的知识
        
        Args:
            skill_name: 技能名称
            problem_type: 问题类型
            
        Returns:
            注入的知识内容
        """
        injection = {}
        
        # 根据skill类型注入不同知识
        if skill_name in ['model-selector', 'model-builder', 'hybrid-model-designer']:
            # 建模相关skill
            if problem_type:
                injection['model_recommendations'] = self.get_models_for_problem_type(problem_type)
            injection['high_innovation_combinations'] = self.get_high_innovation_combinations(problem_type)
            injection['model_checklist'] = self.get_checklist('model')
            injection['model_errors'] = self.get_common_errors('model')
        
        elif skill_name in ['section-writer', 'section-iterative-optimizer']:
            # 写作相关skill
            injection['o_award_criteria'] = self.get_o_award_criteria()
            injection['writing_checklist'] = self.get_checklist('writing')
            injection['writing_errors'] = self.get_common_errors('writing')
            injection['quality_thresholds'] = self.get_quality_thresholds()
        
        elif skill_name in ['abstract-generator', 'abstract-iterative-optimizer']:
            # 摘要相关skill
            injection['abstract_checklist'] = self.get_checklist('abstract')
            injection['abstract_errors'] = self.get_common_errors('abstract')
            injection['quality_thresholds'] = self.get_quality_thresholds().get('abstract', {})
        
        elif skill_name in ['sensitivity-analyzer']:
            # 敏感性分析skill
            injection['sensitivity_checklist'] = self.get_checklist('sensitivity')
            injection['validation_errors'] = self.get_common_errors('validation')
        
        elif skill_name in ['ai-deep-search-guide', 'citation-diversity-validator']:
            # 引用相关skill
            injection['citation_requirements'] = self.get_citation_requirements()
        
        elif skill_name in ['chart-generator', 'figure-validator', 'infographic-generator']:
            # 可视化skill
            injection['visualization_errors'] = self.get_common_errors('visualization')
            injection['quality_thresholds'] = self.get_quality_thresholds().get('visualization', {})
        
        elif skill_name in ['quality-reviewer', 'hallucination-detector']:
            # 质量检查skill
            injection['o_award_criteria'] = self.get_o_award_criteria()
            injection['quality_thresholds'] = self.get_quality_thresholds()
            injection['downgrade_risks'] = self.get_downgrade_risks()
        
        # 通用信息
        injection['winner_traits'] = self.o_award_features.get('common_winner_traits', [])
        
        return injection
    
    def format_for_prompt(self, injection: Dict[str, Any]) -> str:
        """
        将注入的知识格式化为prompt可用的文本
        
        Args:
            injection: 注入的知识字典
            
        Returns:
            格式化的文本
        """
        lines = []
        
        for key, value in injection.items():
            if isinstance(value, dict):
                lines.append(f"\n## {key.replace('_', ' ').title()}\n")
                lines.append(json.dumps(value, indent=2, ensure_ascii=False))
            elif isinstance(value, list):
                lines.append(f"\n## {key.replace('_', ' ').title()}\n")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
                    else:
                        lines.append(f"- {item}")
            else:
                lines.append(f"\n## {key.replace('_', ' ').title()}\n")
                lines.append(str(value))
        
        return "\n".join(lines)


# 创建全局实例
_knowledge_injector: Optional[KnowledgeInjector] = None


def get_knowledge_injector() -> KnowledgeInjector:
    """获取知识注入器单例"""
    global _knowledge_injector
    if _knowledge_injector is None:
        _knowledge_injector = KnowledgeInjector()
    return _knowledge_injector


def inject_knowledge(
    skill_name: str, 
    problem_type: Optional[str] = None,
    format_text: bool = True
) -> str:
    """
    便捷函数：为skill注入知识
    
    Args:
        skill_name: 技能名称
        problem_type: 问题类型
        format_text: 是否格式化为文本
        
    Returns:
        注入的知识（字符串或字典）
    """
    injector = get_knowledge_injector()
    injection = injector.inject_for_skill(skill_name, problem_type)
    
    if format_text:
        return injector.format_for_prompt(injection)
    return injection
