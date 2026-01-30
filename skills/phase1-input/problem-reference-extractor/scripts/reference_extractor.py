"""
题目信息引用提取器
从MCM/ICM题目中提取所有可引用的信息源
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExtractedReference:
    """提取的引用信息"""
    bibtex_key: str
    title: str
    ref_type: str  # problem, data_file, external_source, background
    year: int
    authors: List[str] = field(default_factory=lambda: ["{COMAP}"])
    url: Optional[str] = None
    institution: Optional[str] = None
    note: Optional[str] = None
    filename: Optional[str] = None
    extra_fields: Dict[str, str] = field(default_factory=dict)
    
    def to_bibtex(self) -> str:
        """生成BibTeX条目"""
        if self.ref_type == 'problem':
            return self._problem_bibtex()
        elif self.ref_type == 'data_file':
            return self._data_file_bibtex()
        elif self.ref_type == 'external_source':
            return self._external_source_bibtex()
        elif self.ref_type == 'background':
            return self._background_bibtex()
        else:
            return self._misc_bibtex()
    
    def _problem_bibtex(self) -> str:
        """题目声明BibTeX"""
        authors = ' and '.join(self.authors)
        lines = [
            f"@misc{{{self.bibtex_key},",
            f"  author = {{{authors}}},",
            f"  title = {{{{{self.title}}}}},",
            f"  year = {{{self.year}}},",
            f"  howpublished = {{\\url{{https://www.contest.comap.com/undergraduate/contests/mcm/}}}},",
            f"  note = {{{self.note or 'Mathematical Contest in Modeling - Official Problem Statement'}}}"
        ]
        lines.append("}")
        return '\n'.join(lines)
    
    def _data_file_bibtex(self) -> str:
        """数据文件BibTeX"""
        authors = ' and '.join(self.authors)
        escaped_filename = self.filename.replace('_', '\\_') if self.filename else 'data.csv'
        lines = [
            f"@misc{{{self.bibtex_key},",
            f"  author = {{{authors}}},",
            f"  title = {{{{{self.title}}}}},",
            f"  year = {{{self.year}}},",
            f"  howpublished = {{Provided dataset file: {escaped_filename}}},",
            f"  note = {{{self.note or 'Official contest dataset provided with problem statement'}}}"
        ]
        lines.append("}")
        return '\n'.join(lines)
    
    def _external_source_bibtex(self) -> str:
        """外部数据源BibTeX"""
        authors = ' and '.join(self.authors) if self.authors else '{Data Provider}'
        lines = [
            f"@online{{{self.bibtex_key},",
            f"  author = {{{authors}}},",
            f"  title = {{{{{self.title}}}}},",
            f"  year = {{{self.year}}},"
        ]
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
            lines.append(f"  urldate = {{{datetime.now().strftime('%Y-%m-%d')}}},")
        if self.note:
            lines.append(f"  note = {{{self.note}}}")
        else:
            lines.append(f"  note = {{Referenced in problem statement as data source}}")
        lines[-1] = lines[-1].rstrip(',')
        lines.append("}")
        return '\n'.join(lines)
    
    def _background_bibtex(self) -> str:
        """背景资料BibTeX"""
        authors = ' and '.join(self.authors) if self.authors else '{Institution}'
        lines = [
            f"@techreport{{{self.bibtex_key},",
            f"  author = {{{authors}}},",
            f"  title = {{{{{self.title}}}}},"
        ]
        if self.institution:
            lines.append(f"  institution = {{{self.institution}}},")
        lines.append(f"  year = {{{self.year}}},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        if self.note:
            lines.append(f"  note = {{{self.note}}}")
        else:
            lines.append(f"  note = {{Background report referenced in problem statement}}")
        lines[-1] = lines[-1].rstrip(',')
        lines.append("}")
        return '\n'.join(lines)
    
    def _misc_bibtex(self) -> str:
        """其他类型BibTeX"""
        authors = ' and '.join(self.authors) if self.authors else 'Unknown'
        lines = [
            f"@misc{{{self.bibtex_key},",
            f"  author = {{{authors}}},",
            f"  title = {{{{{self.title}}}}},",
            f"  year = {{{self.year}}}"
        ]
        if self.url:
            lines[-1] += ","
            lines.append(f"  howpublished = {{\\url{{{self.url}}}}}")
        lines.append("}")
        return '\n'.join(lines)


class ProblemReferenceExtractor:
    """题目引用提取器"""
    
    # URL识别模式
    URL_PATTERNS = [
        r'https?://[^\s<>"\')\]]+',
        r'www\.[^\s<>"\')\]]+',
    ]
    
    # 数据源关键词映射
    DATA_SOURCE_KEYWORDS = {
        'World Bank': {
            'keywords': ['World Bank', 'worldbank', 'data.worldbank.org'],
            'url': 'https://data.worldbank.org/',
            'author': '{World Bank}'
        },
        'United Nations': {
            'keywords': ['UN Data', 'United Nations', 'data.un.org', 'UN statistics'],
            'url': 'https://data.un.org/',
            'author': '{United Nations}'
        },
        'OECD': {
            'keywords': ['OECD', 'oecd.org', 'Organisation for Economic'],
            'url': 'https://data.oecd.org/',
            'author': '{OECD}'
        },
        'US Census': {
            'keywords': ['Census Bureau', 'census.gov', 'U.S. Census'],
            'url': 'https://www.census.gov/',
            'author': '{U.S. Census Bureau}'
        },
        'EPA': {
            'keywords': ['EPA', 'Environmental Protection Agency', 'epa.gov'],
            'url': 'https://www.epa.gov/',
            'author': '{U.S. Environmental Protection Agency}'
        },
        'CDC': {
            'keywords': ['CDC', 'Centers for Disease Control', 'cdc.gov'],
            'url': 'https://www.cdc.gov/',
            'author': '{Centers for Disease Control and Prevention}'
        },
        'NASA': {
            'keywords': ['NASA', 'nasa.gov', 'National Aeronautics'],
            'url': 'https://www.nasa.gov/',
            'author': '{NASA}'
        },
        'NOAA': {
            'keywords': ['NOAA', 'National Oceanic', 'noaa.gov'],
            'url': 'https://www.noaa.gov/',
            'author': '{NOAA}'
        },
        'WHO': {
            'keywords': ['WHO', 'World Health Organization', 'who.int'],
            'url': 'https://www.who.int/',
            'author': '{World Health Organization}'
        },
        'IMF': {
            'keywords': ['IMF', 'International Monetary Fund', 'imf.org'],
            'url': 'https://www.imf.org/',
            'author': '{International Monetary Fund}'
        },
        'Kaggle': {
            'keywords': ['Kaggle', 'kaggle.com'],
            'url': 'https://www.kaggle.com/',
            'author': '{Kaggle}'
        },
        'GitHub': {
            'keywords': ['GitHub', 'github.com'],
            'url': 'https://github.com/',
            'author': '{GitHub}'
        }
    }
    
    # 报告/文献识别模式
    REPORT_PATTERNS = [
        r'according to (?:the )?["\']?(.+?)["\']? report',
        r'(?:the )?["\']?(.+?)["\']? published by (.+)',
        r'data from (?:the )?["\']?(.+?)["\']?(?:\s|,|\.)',
        r'(?:as reported|as stated) (?:by|in) ["\']?(.+?)["\']?(?:\s|,|\.)',
        r'(?:see|refer to) (?:the )?["\']?(.+?)["\']? for more'
    ]
    
    def __init__(self):
        """初始化提取器"""
        self.references: List[ExtractedReference] = []
        self.problem_info: Dict[str, Any] = {}
    
    def extract_from_text(
        self,
        problem_text: str,
        problem_type: str,
        year: int,
        problem_title: Optional[str] = None,
        data_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        从题目文本提取引用
        
        Args:
            problem_text: 题目全文
            problem_type: 题目类型 (A/B/C/D/E/F)
            year: 年份
            problem_title: 题目标题（可选，会尝试自动提取）
            data_files: 数据文件列表
            
        Returns:
            提取结果
        """
        self.references = []
        self.problem_info = {
            'year': year,
            'type': problem_type,
            'title': problem_title
        }
        
        # 1. 尝试提取题目标题
        if not problem_title:
            problem_title = self._extract_problem_title(problem_text, problem_type)
            self.problem_info['title'] = problem_title
        
        # 2. 创建题目声明引用
        self._add_problem_statement_reference(problem_type, problem_title, year)
        
        # 3. 添加数据文件引用
        if data_files:
            for filename in data_files:
                self._add_data_file_reference(filename, problem_type, problem_title, year)
        
        # 4. 提取外部数据源引用
        self._extract_external_sources(problem_text, year)
        
        # 5. 提取URL引用
        self._extract_urls(problem_text, year)
        
        # 6. 提取背景报告引用
        self._extract_background_reports(problem_text, year)
        
        return self._generate_results()
    
    def extract_from_parsed(
        self,
        parsed_problem: Dict[str, Any],
        year: int
    ) -> Dict[str, Any]:
        """
        从解析后的题目结构提取引用
        
        Args:
            parsed_problem: problem-parser输出的结构化题目信息
            year: 年份
            
        Returns:
            提取结果
        """
        problem_type = parsed_problem.get('problem_type', 'X')
        problem_title = parsed_problem.get('problem_title', 'Unknown Problem')
        background = parsed_problem.get('background', '')
        
        # 获取数据文件
        provided_data = parsed_problem.get('provided_data', {})
        data_files = provided_data.get('files', [])
        
        # 合并所有文本进行搜索
        full_text = ' '.join([
            background,
            ' '.join(q.get('description', '') for q in parsed_problem.get('main_questions', [])),
            provided_data.get('description', '')
        ])
        
        return self.extract_from_text(
            problem_text=full_text,
            problem_type=problem_type,
            year=year,
            problem_title=problem_title,
            data_files=data_files
        )
    
    def _extract_problem_title(self, text: str, problem_type: str) -> str:
        """从文本中提取题目标题"""
        patterns = [
            rf'Problem {problem_type}[:\s]+(.+?)(?:\n|$)',
            rf'MCM Problem {problem_type}[:\s]+(.+?)(?:\n|$)',
            rf'{problem_type}[:\s]+(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return f"Problem {problem_type}"
    
    def _add_problem_statement_reference(
        self,
        problem_type: str,
        problem_title: str,
        year: int
    ) -> None:
        """添加题目声明引用"""
        ref = ExtractedReference(
            bibtex_key=f"mcm{year}problem{problem_type.lower()}",
            title=f"{year} MCM/ICM Problem {problem_type}: {problem_title}",
            ref_type='problem',
            year=year,
            authors=["{COMAP}"],
            note="Mathematical Contest in Modeling - Official Problem Statement"
        )
        self.references.append(ref)
    
    def _add_data_file_reference(
        self,
        filename: str,
        problem_type: str,
        problem_title: str,
        year: int
    ) -> None:
        """添加数据文件引用"""
        # 生成唯一的key
        file_index = len([r for r in self.references if r.ref_type == 'data_file']) + 1
        suffix = str(file_index) if file_index > 1 else ''
        
        ref = ExtractedReference(
            bibtex_key=f"mcm{year}data{problem_type.lower()}{suffix}",
            title=f"{year} MCM/ICM Problem {problem_type} Dataset: {problem_title}",
            ref_type='data_file',
            year=year,
            authors=["{COMAP}"],
            filename=filename,
            note=f"Official contest dataset: {filename}"
        )
        self.references.append(ref)
    
    def _extract_external_sources(self, text: str, year: int) -> None:
        """提取外部数据源引用"""
        text_lower = text.lower()
        
        for source_name, source_info in self.DATA_SOURCE_KEYWORDS.items():
            for keyword in source_info['keywords']:
                if keyword.lower() in text_lower:
                    # 检查是否已添加此数据源
                    existing = [r for r in self.references 
                               if r.ref_type == 'external_source' and source_name.lower() in r.bibtex_key]
                    if not existing:
                        key_name = source_name.lower().replace(' ', '_').replace('.', '')
                        ref = ExtractedReference(
                            bibtex_key=f"{key_name}{year}data",
                            title=f"{source_name} Data",
                            ref_type='external_source',
                            year=year,
                            authors=[source_info['author']],
                            url=source_info['url'],
                            note=f"Data source referenced in problem statement"
                        )
                        self.references.append(ref)
                    break
    
    def _extract_urls(self, text: str, year: int) -> None:
        """提取URL引用"""
        for pattern in self.URL_PATTERNS:
            matches = re.findall(pattern, text)
            for url in matches:
                # 清理URL
                url = url.rstrip('.,;:')
                
                # 检查是否已通过数据源关键词添加
                already_added = any(
                    r.url and url in r.url
                    for r in self.references
                )
                
                if not already_added:
                    # 尝试从URL提取站点名称
                    site_name = self._extract_site_name(url)
                    key_name = site_name.lower().replace(' ', '_').replace('.', '')[:20]
                    
                    ref = ExtractedReference(
                        bibtex_key=f"web_{key_name}_{year}",
                        title=f"Data from {site_name}",
                        ref_type='external_source',
                        year=year,
                        authors=[f"{{{site_name}}}"],
                        url=url,
                        note="Web resource referenced in problem statement"
                    )
                    self.references.append(ref)
    
    def _extract_site_name(self, url: str) -> str:
        """从URL提取站点名称"""
        import urllib.parse
        try:
            parsed = urllib.parse.urlparse(url if url.startswith('http') else f'https://{url}')
            domain = parsed.netloc or parsed.path.split('/')[0]
            # 移除www.前缀
            domain = domain.replace('www.', '')
            # 取主域名
            parts = domain.split('.')
            if len(parts) >= 2:
                return parts[-2].capitalize()
            return domain.capitalize() if domain else "Web Source"
        except (ValueError, AttributeError, IndexError) as e:
            logger.debug(f"Failed to extract site name from URL {url}: {e}")
            return "Web Source"
        except Exception as e:
            logger.warning(f"Unexpected error extracting site name from URL {url}: {e}")
            return "Web Source"
    
    def _extract_background_reports(self, text: str, year: int) -> None:
        """提取背景报告引用"""
        for pattern in self.REPORT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    title = match[0]
                    institution = match[1] if len(match) > 1 else None
                else:
                    title = match
                    institution = None
                
                # 清理标题
                title = title.strip().strip('"\'')
                
                # 跳过太短或通用的标题
                if len(title) < 5 or title.lower() in ['the', 'this', 'that', 'data']:
                    continue
                
                # 检查是否已添加
                already_added = any(
                    title.lower() in r.title.lower()
                    for r in self.references
                )
                
                if not already_added:
                    key_name = title.split()[0].lower() if title else 'report'
                    inst_key = institution.split()[0].lower() if institution else 'unknown'
                    
                    ref = ExtractedReference(
                        bibtex_key=f"{inst_key}{year}{key_name}",
                        title=title,
                        ref_type='background',
                        year=year,
                        authors=[f"{{{institution}}}" if institution else "{Unknown}"],
                        institution=institution,
                        note="Background report referenced in problem statement"
                    )
                    self.references.append(ref)
    
    def _generate_results(self) -> Dict[str, Any]:
        """生成提取结果"""
        # 分类引用
        categorized = {
            'problem_statement': [],
            'data_files': [],
            'external_sources': [],
            'background_references': []
        }
        
        category_map = {
            'problem': 'problem_statement',
            'data_file': 'data_files',
            'external_source': 'external_sources',
            'background': 'background_references'
        }
        
        for ref in self.references:
            category = category_map.get(ref.ref_type, 'external_sources')
            categorized[category].append({
                'bibtex_key': ref.bibtex_key,
                'title': ref.title,
                'bibtex': ref.to_bibtex(),
                'url': ref.url,
                'filename': ref.filename
            })
        
        # 生成完整的BibTeX内容
        bibtex_parts = [
            "% ==================================================",
            f"% Problem Statement References",
            f"% Extracted from MCM/ICM {self.problem_info.get('year', '')} Problem {self.problem_info.get('type', '')}",
            "% =================================================="
        ]
        
        for ref in self.references:
            bibtex_parts.append("")
            bibtex_parts.append(f"% {ref.ref_type.replace('_', ' ').title()}")
            bibtex_parts.append(ref.to_bibtex())
        
        bibtex_content = '\n'.join(bibtex_parts)
        
        return {
            'problem_info': self.problem_info,
            'extracted_references': categorized,
            'total_citations': len(self.references),
            'bibtex_file_content': bibtex_content
        }
    
    def export_bibtex(self, output_path: str) -> None:
        """导出BibTeX文件"""
        results = self._generate_results()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(results['bibtex_file_content'])
        
        logger.info(f"Problem references exported to {output_path}")
    
    def get_references(self) -> List[ExtractedReference]:
        """获取所有提取的引用"""
        return self.references


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    extractor = ProblemReferenceExtractor()
    
    # 测试文本
    test_text = """
    2026 MCM Problem A: Optimizing Solar Panel Placement
    
    Background: Solar energy is becoming increasingly important for sustainable development.
    According to the International Energy Agency report, solar power capacity has grown significantly.
    
    Your team should use data from the World Bank and UN Data to analyze global trends.
    Additional data can be found at https://www.nrel.gov/gis/solar.html
    
    The CDC has published guidelines on environmental health impacts that may be relevant.
    """
    
    results = extractor.extract_from_text(
        problem_text=test_text,
        problem_type="A",
        year=2026,
        problem_title="Optimizing Solar Panel Placement",
        data_files=["2026_MCM_Problem_A_Data.csv"]
    )
    
    print("Extraction Results:")
    print(f"Total citations: {results['total_citations']}")
    print("\nBibTeX Content:")
    print(results['bibtex_file_content'])
