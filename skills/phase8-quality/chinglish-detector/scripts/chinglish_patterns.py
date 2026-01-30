"""
中式英语模式检测器
Chinglish Pattern Detector

包含50+常见中式英语模式的检测和修正功能
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChinglishPattern:
    """中式英语模式"""
    id: str
    category: str
    pattern: str  # 正则表达式模式
    chinglish: str  # 中式英语描述
    corrections: List[str]  # 可能的修正
    explanation: str
    severity: str  # high, medium, low
    

# 中式英语模式库
CHINGLISH_PATTERNS = {
    # 类别1: 陈词滥调开头
    "cliche_openings": [
        ChinglishPattern(
            id="C01",
            category="cliche_openings",
            pattern=r"\b[Ww]ith the (rapid )?development of\b",
            chinglish="With the development of...",
            corrections=["As X advances", "Recent advances in X", "The evolution of X"],
            explanation="过度使用的陈词滥调开头",
            severity="high"
        ),
        ChinglishPattern(
            id="C02",
            category="cliche_openings",
            pattern=r"\b[Ii]n recent years\b",
            chinglish="In recent years...",
            corrections=["Recently", "Over the past decade", "In the past X years"],
            explanation="不够精确的时间表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="C03",
            category="cliche_openings",
            pattern=r"\b[Nn]owadays\b",
            chinglish="Nowadays...",
            corrections=["Currently", "At present", "Today"],
            explanation="口语化表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="C04",
            category="cliche_openings",
            pattern=r"\b[Aa]s we all know\b",
            chinglish="As we all know...",
            corrections=["[直接陈述事实]"],
            explanation="假设读者知识，应直接陈述",
            severity="high"
        ),
        ChinglishPattern(
            id="C05",
            category="cliche_openings",
            pattern=r"\b[Ii]t is well[- ]known that\b",
            chinglish="It is well known that...",
            corrections=["[直接陈述事实]"],
            explanation="无意义填充，应直接陈述",
            severity="high"
        ),
        ChinglishPattern(
            id="C06",
            category="cliche_openings",
            pattern=r"\b[Aa]s is known to all\b",
            chinglish="As is known to all...",
            corrections=["[直接陈述事实]"],
            explanation="无意义填充，应直接陈述",
            severity="high"
        ),
        ChinglishPattern(
            id="C07",
            category="cliche_openings",
            pattern=r"\b[Uu]nder the background of\b",
            chinglish="Under the background of...",
            corrections=["In the context of", "Given", "Considering"],
            explanation="直译表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="C08",
            category="cliche_openings",
            pattern=r"\b[Ww]ith the coming of\b",
            chinglish="With the coming of...",
            corrections=["The emergence of", "The advent of", "With the arrival of"],
            explanation="直译表达",
            severity="medium"
        ),
    ],
    
    # 类别2: 程度表达
    "degree_expressions": [
        ChinglishPattern(
            id="D01",
            category="degree_expressions",
            pattern=r"\bmore and more\b",
            chinglish="more and more",
            corrections=["increasingly", "an increasing number of", "a growing number of"],
            explanation="口语化表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="D02",
            category="degree_expressions",
            pattern=r"\ba lot of\b",
            chinglish="a lot of",
            corrections=["numerous", "substantial", "many", "considerable"],
            explanation="口语化表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="D03",
            category="degree_expressions",
            pattern=r"\blots of\b",
            chinglish="lots of",
            corrections=["numerous", "many", "a large number of"],
            explanation="口语化表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="D04",
            category="degree_expressions",
            pattern=r"\bvery very\b",
            chinglish="very very",
            corrections=["extremely", "highly", "exceptionally"],
            explanation="重复强调",
            severity="high"
        ),
        ChinglishPattern(
            id="D05",
            category="degree_expressions",
            pattern=r"\bvery unique\b",
            chinglish="very unique",
            corrections=["unique"],
            explanation="逻辑错误，unique无需修饰",
            severity="medium"
        ),
        ChinglishPattern(
            id="D06",
            category="degree_expressions",
            pattern=r"\bget better\b",
            chinglish="get better",
            corrections=["improve", "enhance", "ameliorate"],
            explanation="口语化表达",
            severity="low"
        ),
        ChinglishPattern(
            id="D07",
            category="degree_expressions",
            pattern=r"\bget worse\b",
            chinglish="get worse",
            corrections=["deteriorate", "decline", "worsen"],
            explanation="口语化表达",
            severity="low"
        ),
    ],
    
    # 类别3: 动词搭配
    "verb_collocations": [
        ChinglishPattern(
            id="V01",
            category="verb_collocations",
            pattern=r"\bplay[s]? an? (important|vital|crucial|key|significant) role\b",
            chinglish="plays an important role",
            corrections=["serves as", "functions as", "is essential for", "is crucial for"],
            explanation="过度使用的表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="V02",
            category="verb_collocations",
            pattern=r"\bhas? (great|significant|profound) influence on\b",
            chinglish="has great influence on",
            corrections=["significantly affects", "considerably influences", "substantially impacts"],
            explanation="更直接的表达更好",
            severity="low"
        ),
        ChinglishPattern(
            id="V03",
            category="verb_collocations",
            pattern=r"\bmake[s]? a contribution to\b",
            chinglish="make a contribution to",
            corrections=["contribute to", "aid", "support"],
            explanation="名词化，应使用动词",
            severity="medium"
        ),
        ChinglishPattern(
            id="V04",
            category="verb_collocations",
            pattern=r"\bgive[s]? a description of\b",
            chinglish="give a description of",
            corrections=["describe", "characterize", "depict"],
            explanation="名词化，应使用动词",
            severity="medium"
        ),
        ChinglishPattern(
            id="V05",
            category="verb_collocations",
            pattern=r"\bmake[s]? an? analysis of\b",
            chinglish="make an analysis of",
            corrections=["analyze", "examine", "investigate"],
            explanation="名词化，应使用动词",
            severity="medium"
        ),
        ChinglishPattern(
            id="V06",
            category="verb_collocations",
            pattern=r"\bconduct[s]? research on\b",
            chinglish="conduct research on",
            corrections=["research", "investigate", "study"],
            explanation="更简洁的表达",
            severity="low"
        ),
        ChinglishPattern(
            id="V07",
            category="verb_collocations",
            pattern=r"\bput[s]? forward\b",
            chinglish="put forward",
            corrections=["propose", "present", "suggest", "introduce"],
            explanation="直译表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="V08",
            category="verb_collocations",
            pattern=r"\bsolve the problem\b",
            chinglish="solve the problem",
            corrections=["address the problem", "tackle the issue", "resolve the challenge"],
            explanation="更学术的表达",
            severity="low"
        ),
    ],
    
    # 类别4: 冗余表达
    "redundancy": [
        ChinglishPattern(
            id="R01",
            category="redundancy",
            pattern=r"\bbasic fundamentals?\b",
            chinglish="basic fundamentals",
            corrections=["fundamentals", "basics"],
            explanation="同义重复",
            severity="medium"
        ),
        ChinglishPattern(
            id="R02",
            category="redundancy",
            pattern=r"\bfuture prospects?\b",
            chinglish="future prospects",
            corrections=["prospects", "future outlook"],
            explanation="同义重复",
            severity="medium"
        ),
        ChinglishPattern(
            id="R03",
            category="redundancy",
            pattern=r"\bpast history\b",
            chinglish="past history",
            corrections=["history"],
            explanation="同义重复",
            severity="medium"
        ),
        ChinglishPattern(
            id="R04",
            category="redundancy",
            pattern=r"\btrue facts?\b",
            chinglish="true fact",
            corrections=["fact"],
            explanation="同义重复",
            severity="medium"
        ),
        ChinglishPattern(
            id="R05",
            category="redundancy",
            pattern=r"\bfinal conclusions?\b",
            chinglish="final conclusion",
            corrections=["conclusion"],
            explanation="同义重复",
            severity="low"
        ),
        ChinglishPattern(
            id="R06",
            category="redundancy",
            pattern=r"\bcompletely eliminate\b",
            chinglish="completely eliminate",
            corrections=["eliminate"],
            explanation="同义重复",
            severity="low"
        ),
        ChinglishPattern(
            id="R07",
            category="redundancy",
            pattern=r"\babsolutely essential\b",
            chinglish="absolutely essential",
            corrections=["essential"],
            explanation="同义重复",
            severity="low"
        ),
        ChinglishPattern(
            id="R08",
            category="redundancy",
            pattern=r"\bend results?\b",
            chinglish="end result",
            corrections=["result", "outcome"],
            explanation="同义重复",
            severity="low"
        ),
    ],
    
    # 类别5: 介词使用
    "preposition_usage": [
        ChinglishPattern(
            id="P01",
            category="preposition_usage",
            pattern=r"\bin the aspect of\b",
            chinglish="in the aspect of",
            corrections=["in terms of", "regarding", "concerning"],
            explanation="直译表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="P02",
            category="preposition_usage",
            pattern=r"\bin the field of ([A-Za-z]+)\b",
            chinglish="in the field of X",
            corrections=["in X"],
            explanation="冗余表达",
            severity="low"
        ),
        ChinglishPattern(
            id="P03",
            category="preposition_usage",
            pattern=r"\bin the process of\b",
            chinglish="in the process of",
            corrections=["while", "when", "during"],
            explanation="冗余表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="P04",
            category="preposition_usage",
            pattern=r"\bthrough the method of\b",
            chinglish="through the method of",
            corrections=["by", "using", "via"],
            explanation="冗余表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="P05",
            category="preposition_usage",
            pattern=r"\bfor the purpose of\b",
            chinglish="for the purpose of",
            corrections=["to", "for"],
            explanation="冗余表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="P06",
            category="preposition_usage",
            pattern=r"\bdue to the fact that\b",
            chinglish="due to the fact that",
            corrections=["because", "since", "as"],
            explanation="冗余表达",
            severity="medium"
        ),
        ChinglishPattern(
            id="P07",
            category="preposition_usage",
            pattern=r"\bon the basis of\b",
            chinglish="on the basis of",
            corrections=["based on", "according to"],
            explanation="更简洁的表达",
            severity="low"
        ),
        ChinglishPattern(
            id="P08",
            category="preposition_usage",
            pattern=r"\bby means of\b",
            chinglish="by means of",
            corrections=["by", "through", "using"],
            explanation="更简洁的表达",
            severity="low"
        ),
    ],
    
    # 类别6: 句式结构
    "sentence_structure": [
        ChinglishPattern(
            id="S01",
            category="sentence_structure",
            pattern=r"\b[Tt]he reason is because\b",
            chinglish="The reason is because...",
            corrections=["The reason is that", "Because"],
            explanation="重复表达",
            severity="high"
        ),
        ChinglishPattern(
            id="S02",
            category="sentence_structure",
            pattern=r"\b[Ii]t can be seen that\b",
            chinglish="It can be seen that...",
            corrections=["[直接陈述]", "Clearly,", "Evidently,"],
            explanation="冗余开头",
            severity="medium"
        ),
        ChinglishPattern(
            id="S03",
            category="sentence_structure",
            pattern=r"\b[Ii]t should be noted that\b",
            chinglish="It should be noted that...",
            corrections=["Notably,", "Note that", "Importantly,"],
            explanation="更简洁的表达",
            severity="low"
        ),
        ChinglishPattern(
            id="S04",
            category="sentence_structure",
            pattern=r"\b[Tt]here is no doubt that\b",
            chinglish="There is no doubt that...",
            corrections=["Clearly,", "Undoubtedly,", "Certainly,"],
            explanation="更简洁的表达",
            severity="low"
        ),
        ChinglishPattern(
            id="S05",
            category="sentence_structure",
            pattern=r"\b[Ii]n order to\b",
            chinglish="In order to...",
            corrections=["To"],
            explanation="更简洁的表达",
            severity="low"
        ),
        ChinglishPattern(
            id="S06",
            category="sentence_structure",
            pattern=r"\b[Ss]o as to\b",
            chinglish="So as to...",
            corrections=["To"],
            explanation="更简洁的表达",
            severity="low"
        ),
        ChinglishPattern(
            id="S07",
            category="sentence_structure",
            pattern=r"\b[Aa]s far as .+ is concerned\b",
            chinglish="As far as X is concerned...",
            corrections=["Regarding X,", "For X,", "Concerning X,"],
            explanation="更简洁的表达",
            severity="medium"
        ),
    ],
    
    # 类别7: 学术写作特有
    "academic_writing": [
        ChinglishPattern(
            id="A01",
            category="academic_writing",
            pattern=r"\b[Tt]his paper will (discuss|analyze|present|describe)\b",
            chinglish="This paper will discuss...",
            corrections=["This paper discusses", "We discuss", "This study presents"],
            explanation="学术写作使用现在时",
            severity="medium"
        ),
        ChinglishPattern(
            id="A02",
            category="academic_writing",
            pattern=r"\b[Ww]e will (propose|present|develop|analyze)\b",
            chinglish="We will propose...",
            corrections=["We propose", "We present", "We develop"],
            explanation="学术写作使用现在时",
            severity="medium"
        ),
        ChinglishPattern(
            id="A03",
            category="academic_writing",
            pattern=r"\b[Tt]hrough this (study|research|paper)\b",
            chinglish="Through this study...",
            corrections=["This study", "In this study", "Our research"],
            explanation="冗余表达",
            severity="low"
        ),
        ChinglishPattern(
            id="A04",
            category="academic_writing",
            pattern=r"\b[Tt]he experimental results show that\b",
            chinglish="The experimental results show that...",
            corrections=["Results indicate that", "Results show that", "Experiments demonstrate that"],
            explanation="冗余表达",
            severity="low"
        ),
        ChinglishPattern(
            id="A05",
            category="academic_writing",
            pattern=r"\b[Aa]fter careful (analysis|examination|consideration)\b",
            chinglish="After careful analysis...",
            corrections=["Analysis reveals", "Examination shows", "Our analysis indicates"],
            explanation="主观表达，应更客观",
            severity="medium"
        ),
        ChinglishPattern(
            id="A06",
            category="academic_writing",
            pattern=r"\b[Ww]e (think|believe) that\b",
            chinglish="We think/believe that...",
            corrections=["Evidence suggests that", "Results indicate that", "Our analysis shows that"],
            explanation="主观表达，应基于证据",
            severity="high"
        ),
        ChinglishPattern(
            id="A07",
            category="academic_writing",
            pattern=r"\b[Ii]t is (proved|certain) that\b",
            chinglish="It is proved/certain that...",
            corrections=["This demonstrates that", "This indicates that", "Evidence suggests that"],
            explanation="过强声明，应更谨慎",
            severity="medium"
        ),
    ],
}


class ChinglishDetector:
    """中式英语检测器"""
    
    def __init__(self, patterns: Dict = None):
        """
        初始化检测器
        
        Args:
            patterns: 自定义模式库，默认使用内置模式库
        """
        self.patterns = patterns or CHINGLISH_PATTERNS
        self._compile_patterns()
    
    def _compile_patterns(self):
        """编译正则表达式模式"""
        self.compiled_patterns = {}
        for category, pattern_list in self.patterns.items():
            self.compiled_patterns[category] = [
                (p, re.compile(p.pattern, re.IGNORECASE))
                for p in pattern_list
            ]
    
    def detect(self, text: str) -> Dict:
        """
        检测文本中的中式英语问题
        
        Args:
            text: 待检测文本
            
        Returns:
            检测结果
        """
        issues = []
        issues_by_category = {}
        
        for category, patterns in self.compiled_patterns.items():
            category_issues = []
            
            for pattern_obj, compiled in patterns:
                for match in compiled.finditer(text):
                    issue = {
                        'id': pattern_obj.id,
                        'category': category,
                        'matched_text': match.group(),
                        'position': (match.start(), match.end()),
                        'chinglish_pattern': pattern_obj.chinglish,
                        'corrections': pattern_obj.corrections,
                        'explanation': pattern_obj.explanation,
                        'severity': pattern_obj.severity
                    }
                    category_issues.append(issue)
                    issues.append(issue)
            
            issues_by_category[category] = len(category_issues)
        
        # 计算评分
        chinglish_score = self._calculate_score(text, issues)
        
        # 生成修正建议
        corrections = self._generate_corrections(text, issues)
        
        return {
            'total_issues': len(issues),
            'chinglish_score': chinglish_score,
            'quality_level': self._get_quality_level(chinglish_score),
            'issues_by_category': issues_by_category,
            'issues': issues,
            'corrections': corrections
        }
    
    def _calculate_score(self, text: str, issues: List[Dict]) -> float:
        """计算中式英语评分"""
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        # 严重程度权重
        severity_weights = {'high': 1.5, 'medium': 1.0, 'low': 0.5}
        
        weighted_sum = sum(
            severity_weights.get(issue['severity'], 1.0)
            for issue in issues
        )
        
        # 归一化评分
        score = min(1.0, weighted_sum / word_count * 10)
        return round(score, 3)
    
    def _get_quality_level(self, score: float) -> str:
        """获取质量等级"""
        if score <= 0.15:
            return "excellent"
        elif score <= 0.25:
            return "good"
        elif score <= 0.40:
            return "fair"
        elif score <= 0.60:
            return "poor"
        else:
            return "critical"
    
    def _generate_corrections(self, text: str, issues: List[Dict]) -> Dict:
        """生成修正建议"""
        # 按位置倒序排列
        sorted_issues = sorted(issues, key=lambda x: x['position'][0], reverse=True)
        
        corrected_text = text
        corrections = []
        
        for issue in sorted_issues:
            original = issue['matched_text']
            suggested = issue['corrections'][0] if issue['corrections'] else original
            
            # 保持大小写
            if original[0].isupper() and suggested[0].islower():
                suggested = suggested[0].upper() + suggested[1:]
            
            correction = {
                'original': original,
                'suggested': suggested,
                'position': issue['position'],
                'category': issue['category'],
                'severity': issue['severity']
            }
            corrections.append(correction)
            
            # 应用修正
            start, end = issue['position']
            corrected_text = corrected_text[:start] + suggested + corrected_text[end:]
        
        return {
            'corrections': list(reversed(corrections)),
            'corrected_text': corrected_text
        }
    
    def correct(self, text: str) -> str:
        """
        检测并修正文本
        
        Args:
            text: 待修正文本
            
        Returns:
            修正后的文本
        """
        result = self.detect(text)
        return result['corrections']['corrected_text']


def load_patterns_from_json(filepath: str) -> Dict:
    """从JSON文件加载模式库"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    patterns = {}
    for category, pattern_list in data.items():
        patterns[category] = [
            ChinglishPattern(**p) for p in pattern_list
        ]
    
    return patterns


def detect_chinglish(text: str) -> Dict:
    """便捷函数：检测中式英语"""
    detector = ChinglishDetector()
    return detector.detect(text)


def correct_chinglish(text: str) -> str:
    """便捷函数：修正中式英语"""
    detector = ChinglishDetector()
    return detector.correct(text)


# 测试代码
if __name__ == '__main__':
    test_text = """
    With the development of artificial intelligence, machine learning 
    plays an important role in many fields. More and more researchers 
    conduct research on this topic. In order to solve this problem, 
    we put forward a new method. Through the method of deep learning, 
    we can get better results. It is well known that this approach 
    is very very effective. The reason is because it uses advanced 
    algorithms. We believe that our method will become more and more 
    popular in the future.
    """
    
    detector = ChinglishDetector()
    result = detector.detect(test_text)
    
    print(f"检测到 {result['total_issues']} 处中式英语问题")
    print(f"评分: {result['chinglish_score']} ({result['quality_level']})")
    print(f"\n按类别统计: {result['issues_by_category']}")
    
    print("\n检测到的问题:")
    for issue in result['issues']:
        print(f"  [{issue['severity']}] {issue['matched_text']}")
        print(f"       → 建议: {issue['corrections'][0]}")
    
    print(f"\n修正后的文本:\n{result['corrections']['corrected_text']}")
