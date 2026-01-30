"""
多源混合文献搜索器
整合多个学术和非学术数据源，确保引用多样性
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """引用数据类"""
    bibtex_key: str
    title: str
    authors: List[str]
    year: int
    category: str  # academic, government, data, problem, media, technical
    source: str    # semantic_scholar, openalex, arxiv, etc.
    doi: Optional[str] = None
    url: Optional[str] = None
    venue: Optional[str] = None
    citation_count: int = 0
    relevance_score: float = 0.0
    abstract: Optional[str] = None
    accessed_date: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_bibtex(self) -> str:
        """生成BibTeX条目"""
        if self.category == 'academic':
            return self._academic_bibtex()
        elif self.category == 'government':
            return self._government_bibtex()
        elif self.category == 'data':
            return self._data_bibtex()
        elif self.category == 'problem':
            return self._problem_bibtex()
        else:
            return self._misc_bibtex()
    
    def _academic_bibtex(self) -> str:
        """学术论文BibTeX"""
        entry_type = 'article' if self.venue else 'misc'
        authors = ' and '.join(self.authors) if self.authors else 'Unknown'
        
        lines = [
            f"@{entry_type}{{{self.bibtex_key},",
            f"  author = {{{authors}}},",
            f"  title = {{{self.title}}},",
            f"  year = {{{self.year}}},"
        ]
        
        if self.venue:
            lines.append(f"  journal = {{{self.venue}}},")
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        
        lines[-1] = lines[-1].rstrip(',')
        lines.append("}")
        return '\n'.join(lines)
    
    def _government_bibtex(self) -> str:
        """政府报告BibTeX"""
        authors = ' and '.join(self.authors) if self.authors else '{Unknown}'
        institution = self.extra_fields.get('institution', 'Government Agency')
        
        lines = [
            f"@techreport{{{self.bibtex_key},",
            f"  author = {{{authors}}},",
            f"  title = {{{self.title}}},",
            f"  institution = {{{institution}}},",
            f"  year = {{{self.year}}},"
        ]
        
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        if self.accessed_date:
            lines.append(f"  note = {{Accessed: {self.accessed_date}}},")
        
        lines[-1] = lines[-1].rstrip(',')
        lines.append("}")
        return '\n'.join(lines)
    
    def _data_bibtex(self) -> str:
        """数据源BibTeX"""
        authors = ' and '.join(self.authors) if self.authors else '{Data Provider}'
        
        lines = [
            f"@online{{{self.bibtex_key},",
            f"  author = {{{authors}}},",
            f"  title = {{{self.title}}},",
            f"  year = {{{self.year}}},"
        ]
        
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        if self.accessed_date:
            lines.append(f"  urldate = {{{self.accessed_date}}},")
        
        note = self.extra_fields.get('note', '')
        if note:
            lines.append(f"  note = {{{note}}},")
        
        lines[-1] = lines[-1].rstrip(',')
        lines.append("}")
        return '\n'.join(lines)
    
    def _problem_bibtex(self) -> str:
        """题目引用BibTeX"""
        lines = [
            f"@misc{{{self.bibtex_key},",
            f"  author = {{{{COMAP}}}},",
            f"  title = {{{self.title}}},",
            f"  year = {{{self.year}}},",
            f"  howpublished = {{MCM/ICM Contest}},"
        ]
        
        note = self.extra_fields.get('note', 'Official problem statement')
        lines.append(f"  note = {{{note}}}")
        lines.append("}")
        return '\n'.join(lines)
    
    def _misc_bibtex(self) -> str:
        """其他类型BibTeX"""
        authors = ' and '.join(self.authors) if self.authors else 'Unknown'
        
        lines = [
            f"@misc{{{self.bibtex_key},",
            f"  author = {{{authors}}},",
            f"  title = {{{self.title}}},",
            f"  year = {{{self.year}}},"
        ]
        
        if self.url:
            lines.append(f"  howpublished = {{\\url{{{self.url}}}}},")
        if self.accessed_date:
            lines.append(f"  note = {{Accessed: {self.accessed_date}}},")
        
        lines[-1] = lines[-1].rstrip(',')
        lines.append("}")
        return '\n'.join(lines)


class MultiSourceSearcher:
    """多源混合搜索器"""
    
    # 数据源分类
    SOURCES = {
        'academic': ['semantic_scholar', 'openalex', 'arxiv', 'crossref', 'google_scholar', 'pubmed'],
        'government': ['worldbank_publications', 'un_publications', 'oecd_library'],
        'data': ['worldbank_data', 'un_data', 'kaggle', 'github_datasets'],
        'media': ['news_api', 'reliable_news_sources'],
        'technical': ['github_repos', 'documentation']
    }
    
    # 最低要求
    MIN_REQUIRED = {
        'academic_papers': 3,
        'government_reports': 1,
        'official_data': 1,
        'problem_references': 1,
        'other_sources': 0
    }
    
    # 类别权重
    CATEGORY_WEIGHTS = {
        'academic_papers': 0.40,
        'government_reports': 0.15,
        'official_data': 0.15,
        'problem_references': 0.15,
        'other_sources': 0.15
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化搜索器"""
        self.config = self._load_config(config_path)
        self.citations: Dict[str, List[Citation]] = {
            'academic_papers': [],
            'government_reports': [],
            'official_data': [],
            'problem_references': [],
            'other_sources': []
        }
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        default_config = {
            'academic_sources': {
                'semantic_scholar': {'enabled': True, 'api_key_required': False, 'priority': 1},
                'openalex': {'enabled': True, 'api_key_required': False, 'priority': 2},
                'arxiv': {'enabled': True, 'api_key_required': False, 'priority': 3},
                'crossref': {'enabled': True, 'api_key_required': False, 'priority': 4},
                'google_scholar': {'enabled': True, 'api_key_required': True, 'priority': 5}
            },
            'diversity_requirements': {
                'min_categories': 4,
                'min_academic': 3,
                'min_government': 1,
                'min_official_data': 1,
                'require_problem_statement': True
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        return default_config
    
    def search_all_sources(
        self,
        query: str,
        problem_context: Dict[str, Any],
        min_citations: int = 10,
        ensure_diversity: bool = True
    ) -> Dict[str, Any]:
        """
        执行全源搜索
        
        Args:
            query: 主搜索查询
            problem_context: 题目上下文信息
            min_citations: 最少引用数量
            ensure_diversity: 是否确保多样性
            
        Returns:
            包含所有引用和多样性指标的字典
        """
        logger.info(f"Starting multi-source search: {query}")
        
        # 重置引用
        self.citations = {k: [] for k in self.citations}
        
        # 1. 学术论文搜索
        self._search_academic_sources(query, problem_context)
        
        # 2. 政府报告搜索（需要通过WebSearch工具）
        # 此处标记需要AI执行WebSearch
        self._mark_government_search_needed(query, problem_context)
        
        # 3. 官方数据源引用（通过data-collector生成）
        # 此处标记需要数据源引用
        self._mark_data_sources_needed(problem_context)
        
        # 4. 题目引用（通过problem-reference-extractor生成）
        # 此处标记需要题目引用
        self._mark_problem_references_needed(problem_context)
        
        # 5. 检查并补充多样性
        if ensure_diversity:
            diversity_metrics = self.calculate_diversity_metrics()
            if diversity_metrics['diversity_score'] < 0.75:
                self._supplement_missing_categories(query, problem_context)
        
        # 生成最终结果
        return self._generate_results()
    
    def _search_academic_sources(self, query: str, problem_context: Dict) -> None:
        """搜索学术数据源"""
        sources = self.config.get('academic_sources', {})
        
        # 按优先级排序
        sorted_sources = sorted(
            sources.items(),
            key=lambda x: x[1].get('priority', 99)
        )
        
        for source_name, source_config in sorted_sources:
            if not source_config.get('enabled', False):
                continue
            
            try:
                if source_name == 'semantic_scholar':
                    papers = self._search_semantic_scholar(query)
                elif source_name == 'openalex':
                    papers = self._search_openalex(query)
                elif source_name == 'arxiv':
                    papers = self._search_arxiv(query)
                elif source_name == 'crossref':
                    papers = self._search_crossref(query)
                else:
                    continue
                
                # 转换为Citation对象
                for paper in papers:
                    citation = self._paper_to_citation(paper, source_name)
                    if citation:
                        self.citations['academic_papers'].append(citation)
                
                # 检查是否已有足够数量
                if len(self.citations['academic_papers']) >= self.MIN_REQUIRED['academic_papers'] * 2:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to search {source_name}: {e}")
                continue
    
    def _search_semantic_scholar(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索Semantic Scholar"""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': limit,
            'fields': 'paperId,title,abstract,year,venue,citationCount,authors,externalIds,url'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []
    
    def _search_openalex(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索OpenAlex"""
        url = "https://api.openalex.org/works"
        params = {
            'search': query,
            'per_page': limit,
            'select': 'id,title,authorships,publication_year,primary_location,cited_by_count,doi'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('results', [])
        except Exception as e:
            logger.error(f"OpenAlex search failed: {e}")
            return []
    
    def _search_arxiv(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索arXiv"""
        import urllib.parse
        
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:{urllib.parse.quote(query)}',
            'start': 0,
            'max_results': limit
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            # 解析Atom XML需要额外处理
            # 此处简化返回空列表，实际使用时应解析XML
            return []
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []
    
    def _search_crossref(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索CrossRef"""
        url = "https://api.crossref.org/works"
        params = {
            'query': query,
            'rows': limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('message', {}).get('items', [])
        except Exception as e:
            logger.error(f"CrossRef search failed: {e}")
            return []
    
    def _paper_to_citation(self, paper: Dict, source: str) -> Optional[Citation]:
        """将论文数据转换为Citation对象"""
        try:
            if source == 'semantic_scholar':
                authors = [a.get('name', '') for a in paper.get('authors', [])]
                first_author = authors[0].split()[-1] if authors else 'Unknown'
                year = paper.get('year', datetime.now().year)
                title_word = paper.get('title', '').split()[0] if paper.get('title') else 'paper'
                
                external_ids = paper.get('externalIds', {}) or {}
                
                return Citation(
                    bibtex_key=f"{first_author.lower()}{year}{title_word.lower()}",
                    title=paper.get('title', ''),
                    authors=authors,
                    year=year,
                    category='academic',
                    source='semantic_scholar',
                    doi=external_ids.get('DOI'),
                    url=paper.get('url', ''),
                    venue=paper.get('venue', ''),
                    citation_count=paper.get('citationCount', 0),
                    abstract=paper.get('abstract', '')
                )
            
            elif source == 'openalex':
                authorships = paper.get('authorships', [])
                authors = [a.get('author', {}).get('display_name', '') for a in authorships]
                first_author = authors[0].split()[-1] if authors else 'Unknown'
                year = paper.get('publication_year', datetime.now().year)
                title = paper.get('title', '')
                title_word = title.split()[0] if title else 'paper'
                
                primary_location = paper.get('primary_location', {}) or {}
                source_info = primary_location.get('source', {}) or {}
                
                return Citation(
                    bibtex_key=f"{first_author.lower()}{year}{title_word.lower()}",
                    title=title,
                    authors=authors,
                    year=year,
                    category='academic',
                    source='openalex',
                    doi=paper.get('doi', '').replace('https://doi.org/', '') if paper.get('doi') else None,
                    venue=source_info.get('display_name', ''),
                    citation_count=paper.get('cited_by_count', 0)
                )
            
            elif source == 'crossref':
                authors_data = paper.get('author', [])
                authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors_data]
                first_author = authors_data[0].get('family', 'Unknown') if authors_data else 'Unknown'
                
                # CrossRef可能有多种日期格式
                date_parts = paper.get('published', {}).get('date-parts', [[datetime.now().year]])
                year = date_parts[0][0] if date_parts and date_parts[0] else datetime.now().year
                
                title_list = paper.get('title', [''])
                title = title_list[0] if title_list else ''
                title_word = title.split()[0] if title else 'paper'
                
                container = paper.get('container-title', [''])
                venue = container[0] if container else ''
                
                return Citation(
                    bibtex_key=f"{first_author.lower()}{year}{title_word.lower()}",
                    title=title,
                    authors=authors,
                    year=year,
                    category='academic',
                    source='crossref',
                    doi=paper.get('DOI'),
                    url=paper.get('URL', ''),
                    venue=venue,
                    citation_count=paper.get('is-referenced-by-count', 0)
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to convert paper from {source}: {e}")
            return None
    
    def _mark_government_search_needed(self, query: str, problem_context: Dict) -> None:
        """标记需要通过WebSearch搜索政府报告"""
        # 生成政府报告搜索建议
        domain = problem_context.get('domain', '')
        search_suggestions = [
            f"{domain} government report official statistics",
            f"{domain} policy report white paper",
            f"{domain} World Bank UN OECD report"
        ]
        
        logger.info(f"Government search needed. Suggested queries: {search_suggestions}")
        # 实际搜索将由AI通过WebSearch工具执行
    
    def _mark_data_sources_needed(self, problem_context: Dict) -> None:
        """标记需要数据源引用"""
        logger.info("Data source citations needed. Will be generated by data-collector skill.")
    
    def _mark_problem_references_needed(self, problem_context: Dict) -> None:
        """标记需要题目引用"""
        logger.info("Problem statement citations needed. Will be generated by problem-reference-extractor skill.")
    
    def _supplement_missing_categories(self, query: str, problem_context: Dict) -> None:
        """补充缺失的引用类别"""
        current_metrics = self.calculate_diversity_metrics()
        
        for category, count in current_metrics['category_counts'].items():
            min_required = self.MIN_REQUIRED.get(category, 0)
            if count < min_required:
                logger.warning(f"Category {category} has {count} citations, need at least {min_required}")
    
    def calculate_diversity_metrics(self) -> Dict[str, Any]:
        """计算引用多样性指标"""
        category_counts = {k: len(v) for k, v in self.citations.items()}
        
        # 计算多样性评分
        score = 0.0
        for category, weight in self.CATEGORY_WEIGHTS.items():
            count = category_counts.get(category, 0)
            min_required = self.MIN_REQUIRED.get(category, 0)
            
            if count > 0:
                score += weight
                # 超过最低要求额外加分
                if count > min_required:
                    score += weight * 0.2
        
        score = min(score, 1.0)
        
        # 计算覆盖的类别数
        categories_covered = sum(1 for v in self.citations.values() if len(v) > 0)
        
        return {
            'diversity_score': round(score, 2),
            'categories_covered': categories_covered,
            'category_counts': category_counts,
            'total_citations': sum(category_counts.values()),
            'meets_requirements': all(
                category_counts.get(cat, 0) >= min_req
                for cat, min_req in self.MIN_REQUIRED.items()
            )
        }
    
    def _generate_results(self) -> Dict[str, Any]:
        """生成最终搜索结果"""
        diversity_metrics = self.calculate_diversity_metrics()
        
        # 生成BibTeX文件内容
        bibtex_entries = []
        for category, citations in self.citations.items():
            if citations:
                bibtex_entries.append(f"% {category.replace('_', ' ').title()}")
                for citation in citations:
                    bibtex_entries.append(citation.to_bibtex())
                bibtex_entries.append("")
        
        bibtex_content = '\n'.join(bibtex_entries)
        
        return {
            'citations': {
                category: [
                    {
                        'bibtex_key': c.bibtex_key,
                        'title': c.title,
                        'authors': c.authors,
                        'year': c.year,
                        'source': c.source,
                        'category': c.category,
                        'doi': c.doi,
                        'url': c.url,
                        'citation_count': c.citation_count,
                        'relevance_score': c.relevance_score
                    }
                    for c in citations
                ]
                for category, citations in self.citations.items()
            },
            'diversity_metrics': diversity_metrics,
            'bibtex_content': bibtex_content,
            'search_guidance': {
                'government_search_needed': len(self.citations['government_reports']) < self.MIN_REQUIRED['government_reports'],
                'data_citations_needed': len(self.citations['official_data']) < self.MIN_REQUIRED['official_data'],
                'problem_citations_needed': len(self.citations['problem_references']) < self.MIN_REQUIRED['problem_references']
            }
        }
    
    def add_citation(self, citation: Citation) -> None:
        """添加引用到对应类别"""
        category_map = {
            'academic': 'academic_papers',
            'government': 'government_reports',
            'data': 'official_data',
            'problem': 'problem_references',
            'media': 'other_sources',
            'technical': 'other_sources'
        }
        
        target_category = category_map.get(citation.category, 'other_sources')
        self.citations[target_category].append(citation)
    
    def export_bibtex(self, output_path: str) -> None:
        """导出BibTeX文件"""
        results = self._generate_results()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(results['bibtex_content'])
        
        logger.info(f"BibTeX exported to {output_path}")


def create_government_citation(
    title: str,
    institution: str,
    year: int,
    url: str,
    accessed_date: Optional[str] = None
) -> Citation:
    """创建政府报告引用的便捷函数"""
    key_word = title.split()[0].lower() if title else 'report'
    inst_word = institution.split()[0].lower() if institution else 'gov'
    
    return Citation(
        bibtex_key=f"{inst_word}{year}{key_word}",
        title=title,
        authors=[f"{{{institution}}}"],
        year=year,
        category='government',
        source='websearch',
        url=url,
        accessed_date=accessed_date or datetime.now().strftime('%Y-%m-%d'),
        extra_fields={'institution': institution}
    )


def create_data_citation(
    title: str,
    provider: str,
    year: int,
    url: str,
    indicator: Optional[str] = None,
    accessed_date: Optional[str] = None
) -> Citation:
    """创建数据源引用的便捷函数"""
    key_word = indicator.replace('.', '_') if indicator else title.split()[0].lower()
    provider_word = provider.split()[0].lower()
    
    return Citation(
        bibtex_key=f"{provider_word}_data_{key_word}",
        title=title,
        authors=[f"{{{provider}}}"],
        year=year,
        category='data',
        source='data_api',
        url=url,
        accessed_date=accessed_date or datetime.now().strftime('%Y-%m-%d'),
        extra_fields={'note': f"Data indicator: {indicator}" if indicator else ""}
    )


def create_problem_citation(
    problem_type: str,
    problem_title: str,
    year: int,
    has_data: bool = False
) -> Citation:
    """创建题目引用的便捷函数"""
    return Citation(
        bibtex_key=f"mcm{year}problem{problem_type.lower()}",
        title=f"{year} MCM/ICM Problem {problem_type}: {problem_title}",
        authors=["{COMAP}"],
        year=year,
        category='problem',
        source='mcm_icm',
        url="https://www.contest.comap.com/undergraduate/contests/mcm/",
        extra_fields={
            'note': 'Official problem statement' + (' with provided dataset' if has_data else '')
        }
    )


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    searcher = MultiSourceSearcher()
    
    # 测试搜索
    results = searcher.search_all_sources(
        query="solar panel optimization energy efficiency",
        problem_context={
            'type': 'A',
            'domain': 'renewable energy',
            'keywords': ['solar', 'optimization', 'efficiency']
        }
    )
    
    print("Search Results:")
    print(f"Total citations: {results['diversity_metrics']['total_citations']}")
    print(f"Diversity score: {results['diversity_metrics']['diversity_score']}")
    print(f"Categories covered: {results['diversity_metrics']['categories_covered']}")
    
    for category, count in results['diversity_metrics']['category_counts'].items():
        print(f"  {category}: {count}")
