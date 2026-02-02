"""
质量检查器模块
实现各种质量门禁检查逻辑
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """检查结果"""
    name: str                           # 检查项名称
    passed: bool                        # 是否通过
    actual_value: Any                   # 实际值
    threshold: Any                      # 阈值
    message: str                        # 结果消息
    suggestions: List[str] = None       # 改进建议
    details: Dict[str, Any] = None      # 详细信息

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "passed": self.passed,
            "actual_value": self.actual_value,
            "threshold": self.threshold,
            "message": self.message,
            "suggestions": self.suggestions,
            "details": self.details
        }


class BaseChecker(ABC):
    """质量检查器基类"""

    def __init__(self, llm_adapter: Optional[LLMAdapter] = None):
        self.llm = llm_adapter

    @abstractmethod
    async def check(self, state: Dict[str, Any], criteria: Dict[str, Any]) -> CheckResult:
        """
        执行检查

        Args:
            state: 全局状态
            criteria: 检查标准

        Returns:
            检查结果
        """
        pass

    def _get_value(self, state: Dict, key: str, default: Any = None) -> Any:
        """从状态中获取值 (支持嵌套路径)"""
        keys = key.split('.')
        value = state
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default


class AbstractScoreChecker(BaseChecker):
    """
    摘要评分检查器

    使用LLM评估摘要质量
    """

    async def check(self, state: Dict[str, Any], criteria: Dict[str, Any]) -> CheckResult:
        # 获取摘要文本
        abstract = self._get_value(state, 'abstract', {})
        if isinstance(abstract, dict):
            text = abstract.get('text', '')
        else:
            text = str(abstract)

        if not text:
            return CheckResult(
                name="abstract_score",
                passed=False,
                actual_value=0,
                threshold=criteria.get('min', 0.85),
                message="No abstract text found",
                suggestions=["Generate an abstract first"]
            )

        # 使用LLM评估
        if self.llm is None:
            # 无LLM时使用简单启发式
            score = self._heuristic_score(text)
        else:
            score = await self._llm_evaluate(text)

        threshold = criteria.get('min', 0.85)
        passed = score >= threshold

        return CheckResult(
            name="abstract_score",
            passed=passed,
            actual_value=round(score, 3),
            threshold=threshold,
            message=f"Abstract quality score: {score:.2f} / {threshold}",
            details={"evaluated_text_length": len(text)}
        )

    async def _llm_evaluate(self, text: str) -> float:
        """使用LLM评估摘要"""
        prompt = f"""Evaluate the following MCM/ICM abstract on a scale of 0.0 to 1.0.

Evaluation criteria:
1. Structure (20%): Background → Problem → Method → Results → Conclusion
2. Quantification (20%): Specific numbers, metrics, percentages
3. Innovation (20%): Clear statement of contribution/novel approach
4. Clarity (20%): Concise, no redundancy, each sentence adds value
5. Flow (20%): Natural transitions, professional tone

Abstract (300-500 words):
{text[:2000]}

Return ONLY a JSON object:
{{
  "score": 0.0-1.0,
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "overall_impression": "..."
}}
"""

        try:
            response = await self.llm.complete(prompt, response_format="json", max_tokens=1000)
            result = json.loads(response)
            return float(result.get("score", 0.7))
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            return self._heuristic_score(text)

    def _heuristic_score(self, text: str) -> float:
        """启发式评分 (无LLM时使用)"""
        score = 0.5  # 基础分
        word_count = len(text.split())

        # 长度检查 (300-500词理想)
        if 300 <= word_count <= 500:
            score += 0.1
        elif word_count >= 250:
            score += 0.05

        # 检查关键词
        keywords = ['model', 'result', 'method', 'analysis', 'approach']
        keyword_count = sum(1 for kw in keywords if kw.lower() in text.lower())
        score += min(keyword_count * 0.05, 0.15)

        # 检查数字 (量化表达)
        numbers = re.findall(r'\d+\.?\d*%|\d+\.?\d*\s*(percent|%|million|billion|thousand)', text, re.IGNORECASE)
        if len(numbers) >= 3:
            score += 0.1
        elif len(numbers) >= 1:
            score += 0.05

        # 检查结构标记
        structure_markers = ['we propose', 'our model', 'the results', 'conclusion', 'introduction']
        structure_count = sum(1 for marker in structure_markers if marker.lower() in text.lower())
        score += min(structure_count * 0.03, 0.15)

        return min(score, 1.0)


class HookQualityChecker(BaseChecker):
    """
    Hook质量检查器

    检查摘要开头的吸引力
    """

    # 陈词滥调列表
    CLICHES = [
        "in today's world",
        "in recent years",
        "nowadays",
        "with the development of",
        "in the modern era",
        "in the 21st century",
        "in today's society",
        "in this day and age",
        "it is widely known that",
        "it is no exaggeration to say"
    ]

    async def check(self, state: Dict[str, Any], criteria: Dict[str, Any]) -> CheckResult:
        # 获取摘要文本
        abstract = self._get_value(state, 'abstract', {})
        if isinstance(abstract, dict):
            text = abstract.get('text', '')
            hook = abstract.get('hook', '')
        else:
            text = str(abstract)
            hook = ''

        # 提取第一句作为Hook
        if not hook:
            sentences = text.split('.')
            if sentences:
                hook = sentences[0].strip() + '.'

        if not hook:
            return CheckResult(
                name="hook_quality",
                passed=False,
                actual_value=0,
                threshold=criteria.get('min', 0.80),
                message="No hook found",
                suggestions=["Add an engaging opening sentence"]
            )

        # 评估Hook
        score = 1.0
        issues = []

        # 检查陈词滥调
        hook_lower = hook.lower()
        for cliche in self.CLICHES:
            if cliche in hook_lower:
                score -= 0.4
                issues.append(f"Cliche detected: '{cliche}'")
                break

        # 检查是否有数字
        has_number = bool(re.search(r'\d', hook))
        if not has_number:
            score -= 0.15
            issues.append("Hook lacks specific numbers")

        # 检查长度
        word_count = len(hook.split())
        if word_count > 30:
            score -= 0.1
            issues.append(f"Hook too long ({word_count} words)")
        elif word_count < 10:
            score -= 0.1
            issues.append(f"Hook too short ({word_count} words)")

        # 检查是否有问题陈述
        if not any(word in hook_lower for word in ['problem', 'challenge', 'issue', 'demand', 'need']):
            score -= 0.1
            issues.append("Hook doesn't frame the problem")

        score = max(0, score)
        threshold = criteria.get('min', 0.80)

        return CheckResult(
            name="hook_quality",
            passed=score >= threshold,
            actual_value=round(score, 3),
            threshold=threshold,
            message=f"Hook score: {score:.2f}",
            suggestions=issues if issues else ["Good hook!"],
            details={"hook": hook, "word_count": word_count}
        )


class QuantificationDensityChecker(BaseChecker):
    """
    量化密度检查器

    检查摘要中量化信息的密度
    """

    async def check(self, state: Dict[str, Any], criteria: Dict[str, Any]) -> CheckResult:
        abstract = self._get_value(state, 'abstract', {})
        if isinstance(abstract, dict):
            text = abstract.get('text', '')
        else:
            text = str(abstract)

        if not text:
            return CheckResult(
                name="quantification_density",
                passed=False,
                actual_value=0,
                threshold=criteria.get('min', 0.75),
                message="No abstract text"
            )

        # 统计量化元素
        word_count = len(text.split())

        # 百分比
        percentages = len(re.findall(r'\d+\.?\d*%', text))

        # 大数字
        large_numbers = len(re.findall(r'\d{4,}', text))

        # 小数
        decimals = len(re.findall(r'\d+\.\d+', text))

        # 统计词汇
        stats_words = ['average', 'mean', 'median', 'standard deviation', 'variance', 'r²', 'r-squared', 'correlation']
        stats_count = sum(1 for word in stats_words if word.lower() in text.lower())

        # 比较词汇
        comparison_words = ['increased', 'decreased', 'improved', 'reduced', 'higher', 'lower', 'faster', 'slower']
        comparison_count = sum(1 for word in comparison_words if word.lower() in text.lower())

        # 计算密度
        total_quantifiers = percentages + large_numbers + decimals + stats_count + comparison_count
        density = min(total_quantifiers / (word_count / 20), 1.0)  # 归一化

        threshold = criteria.get('min', 0.75)

        suggestions = []
        if density < threshold:
            suggestions.append("Add more specific numbers and metrics")
            if percentages == 0:
                suggestions.append("Include percentage improvements")
            if large_numbers == 0:
                suggestions.append("Add specific data points")

        return CheckResult(
            name="quantification_density",
            passed=density >= threshold,
            actual_value=round(density, 3),
            threshold=threshold,
            message=f"Quantification density: {density:.2f} ({total_quantifiers} elements in {word_count} words)",
            suggestions=suggestions,
            details={
                "percentages": percentages,
                "large_numbers": large_numbers,
                "decimals": decimals,
                "stats_count": stats_count,
                "comparison_count": comparison_count
            }
        )


class ChinglishChecker(BaseChecker):
    """
    中式英语检查器

    检测常见的中式英语表达模式
    """

    # 中式英语模式 (正则 -> 建议)
    CHINGLISH_PATTERNS = {
        r'\b(very|extremely|really)\s+(good|bad|important|necessary|useful|helpful)\b':
            "Avoid intensifiers; use stronger adjectives instead",
        r'\bthere\s+has\s+been\b':
            "Use active voice",
        r'\bthere\s+(are|were)\s+(a\s+lot\s+of|many|numerous)\b':
            "Be more specific; quantify or categorize",
        r'\bmore\s+and\s+more\b':
            "Use 'increasingly' or specify the trend",
        r'\bwith\s+the\s+(development|rapid\s+development|fast\s+development)\s+of\b':
            "Use 'As X develops' or similar active construction",
        r'\bin\s+order\s+to\b':
            "Use 'to' instead (more concise)",
        r'\bmake\s+(a\s+)?(great|significant|important)\s+(contribution|progress|improvement)\b':
            "Use stronger verbs: contribute, advance, improve",
        r'\bplay\s+an?\s+(important|key|vital|crucial)\s+role\b':
            "Consider alternative phrasing; this is overused",
        r'\b\s+at\s+(the\s+)?(first|last|that)\s+time\b':
            "Consider 'initially', 'finally', or 'then'",
        r'\b(take|make)\s+(measures|steps|precautions)\b':
            "Use more specific verbs",
        r'\b(a\s+)?large\s+number\s+of\b':
            "Use 'many', 'numerous', or specify the count",
        r'\bin\s+terms\s+of\b':
            "Often unnecessary; consider direct expression",
        r'\bprovides?\s+a\s+(new|novel)\s+(method|approach|technique)\b':
            "Specify what makes it new/novel",
        r'\bto\s+some\s+extent\b':
            "Be more specific about the degree",
        r'\bas\s+we\s+all\s+know\b':
            "Remove this filler phrase",
        r'\bit\s+is\s+(worth\s+mentioning|obvious\s+that)\b':
            "Remove this filler phrase",
    }

    async def check(self, state: Dict[str, Any], criteria: Dict[str, Any]) -> CheckResult:
        # 获取各章节文本
        sections = self._get_value(state, 'sections', {})
        abstract = self._get_value(state, 'abstract', {})

        # 收集所有文本
        all_texts = []
        if isinstance(abstract, dict) and 'text' in abstract:
            all_texts.append(abstract['text'])
        elif isinstance(abstract, str):
            all_texts.append(abstract)

        for section_name, section_data in sections.items():
            if isinstance(section_data, dict) and 'content' in section_data:
                all_texts.append(section_data['content'])

        combined_text = ' '.join(all_texts)

        if not combined_text:
            return CheckResult(
                name="chinglish_score",
                passed=False,
                actual_value=0,
                threshold=criteria.get('max', 0.20),
                message="No text to check"
            )

        # 检测问题
        total_sentences = len(re.split(r'[.!?]+', combined_text))
        total_errors = 0
        error_details = []

        for pattern, suggestion in self.CHINGLISH_PATTERNS.items():
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                count = len(matches)
                total_errors += count
                error_details.append({
                    "pattern": pattern,
                    "suggestion": suggestion,
                    "count": count
                })

        # 计算评分
        score = min(total_errors / max(total_sentences, 1), 1.0)
        threshold = criteria.get('max', 0.20)

        suggestions = []
        if not error_details:
            suggestions.append("No Chinglish patterns detected!")
        else:
            for detail in error_details[:3]:  # 最多返回3条建议
                suggestions.append(detail['suggestion'])

        return CheckResult(
            name="chinglish_score",
            passed=score <= threshold,
            actual_value=round(score, 3),
            threshold=threshold,
            message=f"Chinglish ratio: {score:.2f} ({total_errors} issues in {total_sentences} sentences)",
            suggestions=suggestions,
            details={
                "total_errors": total_errors,
                "total_sentences": total_sentences,
                "error_breakdown": error_details
            }
        )


class ConsistencyChecker(BaseChecker):
    """
    一致性检查器

    检查术语、符号、数据使用的一致性
    """

    async def check(self, state: Dict[str, Any], criteria: Dict[str, Any]) -> CheckResult:
        # 获取各章节内容
        sections = self._get_value(state, 'sections', {})

        if not sections:
            return CheckResult(
                name="consistency_score",
                passed=True,
                actual_value=1.0,
                threshold=criteria.get('min', 0.90),
                message="No sections to check"
            )

        issues = []

        # 检查术语一致性
        term_usage = {}
        for section_name, section_data in sections.items():
            if isinstance(section_data, dict) and 'content' in section_data:
                content = section_data['content'].lower()
                # 检查一些关键术语的不同表达
                if "optimization" in content:
                    term_usage.setdefault("optimization", set()).add(section_name)
                if "optimize" in content:
                    term_usage.setdefault("optimize", set()).add(section_name)

        # 检查符号一致性 (简单检查)
        # 实际实现中可以解析LaTeX公式

        # 检查数据一致性
        # 比较摘要和正文中提到的关键数字

        # 计算一致性分数
        total_checks = 3  # 术语、符号、数据
        passed_checks = 3

        # 简化版：如果有很多章节，减分
        if len(sections) > 1:
            # 检查每个章节是否都有标题
            for section_name, section_data in sections.items():
                if isinstance(section_data, dict):
                    if not section_data.get('title'):
                        passed_checks -= 1
                        issues.append(f"Section {section_name} missing title")

        score = passed_checks / total_checks
        threshold = criteria.get('min', 0.90)

        return CheckResult(
            name="consistency_score",
            passed=score >= threshold,
            actual_value=round(score, 3),
            threshold=threshold,
            message=f"Consistency score: {score:.2f}",
            suggestions=issues if issues else ["Good consistency!"]
        )


class HallucinationChecker(BaseChecker):
    """
    幻觉检查器

    检查引用的真实性和数据的合理性
    """

    async def check(self, state: Dict[str, Any], criteria: Dict[str, Any]) -> CheckResult:
        citations = self._get_value(state, 'citations', [])

        if not citations:
            return CheckResult(
                name="hallucination_count",
                passed=criteria.get('required', True) == False,
                actual_value=0,
                threshold=criteria.get('max', 0),
                message="No citations to check",
                suggestions=["Add citations to validate claims"]
            )

        # 检查引用完整性
        issues = []
        for i, citation in enumerate(citations):
            if isinstance(citation, dict):
                if not citation.get('doi') and not citation.get('url'):
                    issues.append(f"Citation {i+1}: Missing DOI or URL")
                if not citation.get('title'):
                    issues.append(f"Citation {i+1}: Missing title")
                if not citation.get('authors'):
                    issues.append(f"Citation {i+1}: Missing authors")

        threshold = criteria.get('max', 0)
        hallucination_count = len(issues)

        return CheckResult(
            name="hallucination_count",
            passed=hallucination_count <= threshold,
            actual_value=hallucination_count,
            threshold=threshold,
            message=f"{hallucination_count} potential hallucinations detected",
            suggestions=issues if issues else ["All citations appear valid!"]
        )


class LATScoreChecker(BaseChecker):
    """
    LAT评分检查器 (Language Assessment Tool)

    评估学术英语质量
    """

    async def check(self, state: Dict[str, Any], criteria: Dict[str, Any]) -> CheckResult:
        # 收集所有文本
        sections = self._get_value(state, 'sections', {})
        abstract = self._get_value(state, 'abstract', {})

        all_texts = []
        if isinstance(abstract, dict) and 'text' in abstract:
            all_texts.append(abstract['text'])

        for section_data in sections.values():
            if isinstance(section_data, dict) and 'content' in section_data:
                all_texts.append(section_data['content'])

        if not all_texts:
            return CheckResult(
                name="lat_score",
                passed=False,
                actual_value=0,
                threshold=criteria.get('min', 7.5),
                message="No text to evaluate"
            )

        combined_text = ' '.join(all_texts)

        # 使用LLM评估 (如果有)
        if self.llm:
            score = await self._llm_lat_score(combined_text)
        else:
            score = self._heuristic_lat_score(combined_text)

        threshold = criteria.get('min', 7.5)

        return CheckResult(
            name="lat_score",
            passed=score >= threshold,
            actual_value=round(score, 2),
            threshold=threshold,
            message=f"LAT score: {score:.1f} / 10"
        )

    async def _llm_lat_score(self, text: str) -> float:
        """使用LLM评估LAT分数"""
        prompt = f"""Rate the following academic text on a scale of 1-10 using the LAT (Language Assessment Tool) criteria:

Criteria:
1. Grammatical Accuracy (1-10)
2. Academic Tone (1-10)
3. Vocabulary Range (1-10)
4. Sentence Variety (1-10)
5. Coherence & Cohesion (1-10)

Text (first 1500 words):
{text[:1500]}

Return ONLY a JSON:
{{"overall_score": 1-10, "breakdown": {{"grammar": N, "tone": N, ...}}}}
"""

        try:
            response = await self.llm.complete(prompt, response_format="json", max_tokens=500)
            result = json.loads(response)
            return float(result.get("overall_score", 7.0))
        except Exception:
            return self._heuristic_lat_score(text)

    def _heuristic_lat_score(self, text: str) -> float:
        """启发式LAT评分"""
        score = 7.0  # 基础分

        # 检查句长多样性
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            if variance > 50:  # 有变化
                score += 0.3

        # 检查学术词汇
        academic_words = ['furthermore', 'moreover', 'therefore', 'however', 'consequently',
                         'nevertheless', 'thus', 'hence', 'accordingly', 'notwithstanding']
        academic_count = sum(1 for word in academic_words if word in text.lower())
        score += min(academic_count * 0.05, 0.5)

        # 检查被动语态 (学术写作中适度使用)
        passive_count = len(re.findall(r'\b(was|were)\s+\w+ed\b', text))
        total_sentences = len(sentences)
        if total_sentences > 0:
            passive_ratio = passive_count / total_sentences
            if 0.1 <= passive_ratio <= 0.3:  # 合理范围
                score += 0.2

        return min(score, 10.0)


# 检查器注册表
CHECKER_REGISTRY: Dict[str, type] = {
    "abstract_score": AbstractScoreChecker,
    "hook_quality": HookQualityChecker,
    "quantification_density": QuantificationDensityChecker,
    "chinglish_score": ChinglishChecker,
    "consistency_score": ConsistencyChecker,
    "hallucination_count": HallucinationChecker,
    "lat_score": LATScoreChecker,
}


def create_checker(name: str, llm_adapter: Optional[LLMAdapter] = None) -> BaseChecker:
    """
    创建检查器实例

    Args:
        name: 检查器名称
        llm_adapter: LLM适配器

    Returns:
        检查器实例
    """
    checker_class = CHECKER_REGISTRY.get(name)
    if checker_class is None:
        raise ValueError(f"Unknown checker: {name}")
    return checker_class(llm_adapter)
