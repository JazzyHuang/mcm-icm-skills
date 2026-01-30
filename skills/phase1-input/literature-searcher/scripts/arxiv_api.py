"""
arXiv API 文献检索
使用arXiv API检索预印本论文
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

# 添加common模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common.api_utils import (
    api_request_with_retry,
    APIRequestError,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# API配置
BASE_URL = "http://export.arxiv.org/api/query"

# 命名空间
NAMESPACES = {
    'atom': 'http://www.w3.org/2005/Atom',
    'arxiv': 'http://arxiv.org/schemas/atom'
}


def search_papers(
    query: str,
    limit: int = 20,
    start: int = 0,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    categories: Optional[List[str]] = None
) -> List[Dict]:
    """
    搜索arXiv论文
    
    Args:
        query: 搜索查询 (支持arXiv查询语法)
        limit: 返回数量限制 (最大2000)
        start: 起始索引
        sort_by: 排序字段 (relevance, lastUpdatedDate, submittedDate)
        sort_order: 排序方向 (ascending, descending)
        categories: 限制分类 (如 ['cs.LG', 'stat.ML'])
        
    Returns:
        论文列表
    """
    # 构建查询
    search_query = query
    if categories:
        cat_query = ' OR '.join(f"cat:{cat}" for cat in categories)
        search_query = f"({query}) AND ({cat_query})"
    
    params = {
        'search_query': f"all:{search_query}",
        'start': start,
        'max_results': min(limit, 2000),
        'sortBy': sort_by,
        'sortOrder': sort_order
    }
    
    logger.info(f"Searching arXiv: {query}")
    
    try:
        response = api_request_with_retry(
            BASE_URL,
            params=params,
            timeout=DEFAULT_TIMEOUT,
            max_retries=DEFAULT_MAX_RETRIES,
            return_json=False  # arXiv返回XML
        )
        
        # 解析XML响应
        if hasattr(response, 'text'):
            xml_content = response.text
        else:
            xml_content = str(response)
        
        results = parse_arxiv_response(xml_content)
        logger.info(f"Found {len(results)} papers")
        return results
        
    except APIRequestError as e:
        logger.error(f"Error searching arXiv: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error searching arXiv: {e}")
        return []


def get_paper_by_id(arxiv_id: str) -> Optional[Dict]:
    """
    通过arXiv ID获取论文详情
    
    Args:
        arxiv_id: arXiv ID (如 '2301.12345' 或 'arxiv:2301.12345')
        
    Returns:
        论文详情
    """
    # 清理ID
    arxiv_id = arxiv_id.replace('arxiv:', '').replace('arXiv:', '')
    
    params = {
        'id_list': arxiv_id
    }
    
    try:
        response = api_request_with_retry(
            BASE_URL,
            params=params,
            timeout=DEFAULT_TIMEOUT,
            max_retries=DEFAULT_MAX_RETRIES,
            return_json=False
        )
        
        if hasattr(response, 'text'):
            xml_content = response.text
        else:
            xml_content = str(response)
        
        results = parse_arxiv_response(xml_content)
        return results[0] if results else None
        
    except APIRequestError as e:
        logger.error(f"Error getting paper {arxiv_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting paper: {e}")
        return None


def parse_arxiv_response(xml_content: str) -> List[Dict]:
    """
    解析arXiv XML响应
    
    Args:
        xml_content: XML字符串
        
    Returns:
        论文列表
    """
    results = []
    
    try:
        root = ElementTree.fromstring(xml_content)
        
        for entry in root.findall('atom:entry', NAMESPACES):
            paper = parse_entry(entry)
            if paper:
                results.append(paper)
                
    except ElementTree.ParseError as e:
        logger.error(f"XML parse error: {e}")
    
    return results


def parse_entry(entry: ElementTree.Element) -> Optional[Dict]:
    """
    解析单个论文条目
    
    Args:
        entry: XML元素
        
    Returns:
        论文数据
    """
    # 提取ID
    id_elem = entry.find('atom:id', NAMESPACES)
    if id_elem is None:
        return None
    
    arxiv_url = id_elem.text or ''
    arxiv_id = arxiv_url.split('/abs/')[-1] if '/abs/' in arxiv_url else arxiv_url
    
    # 提取标题
    title_elem = entry.find('atom:title', NAMESPACES)
    title = (title_elem.text or '').strip().replace('\n', ' ') if title_elem is not None else ''
    
    if not title:
        return None
    
    # 提取作者
    authors = []
    for author_elem in entry.findall('atom:author', NAMESPACES):
        name_elem = author_elem.find('atom:name', NAMESPACES)
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text.strip())
    
    if not authors:
        authors = ['Unknown']
    
    # 提取摘要
    summary_elem = entry.find('atom:summary', NAMESPACES)
    abstract = (summary_elem.text or '').strip().replace('\n', ' ') if summary_elem is not None else ''
    
    # 提取发布日期
    published_elem = entry.find('atom:published', NAMESPACES)
    published = published_elem.text if published_elem is not None else ''
    year = int(published[:4]) if published and len(published) >= 4 else None
    
    # 提取分类
    categories = []
    for cat_elem in entry.findall('arxiv:primary_category', NAMESPACES):
        term = cat_elem.get('term')
        if term:
            categories.append(term)
    for cat_elem in entry.findall('atom:category', NAMESPACES):
        term = cat_elem.get('term')
        if term and term not in categories:
            categories.append(term)
    
    # 提取PDF链接
    pdf_url = ''
    for link_elem in entry.findall('atom:link', NAMESPACES):
        if link_elem.get('title') == 'pdf':
            pdf_url = link_elem.get('href', '')
            break
    
    # 生成BibTeX key
    first_author = authors[0].split()[-1] if authors else 'unknown'
    clean_author = re.sub(r'[^\w]', '', first_author.lower())
    title_parts = title.split() if title else ['untitled']
    clean_title = re.sub(r'[^\w]', '', title_parts[0].lower())
    bibtex_key = f"{clean_author}{year or 'XXXX'}{clean_title}"
    
    return {
        'arxiv_id': arxiv_id,
        'title': title,
        'authors': authors,
        'year': year,
        'published': published,
        'abstract': abstract,
        'categories': categories,
        'primary_category': categories[0] if categories else '',
        'url': arxiv_url,
        'pdf_url': pdf_url,
        'bibtex_key': bibtex_key
    }


def generate_bibtex(paper: Dict) -> str:
    """
    生成BibTeX条目
    
    Args:
        paper: 论文数据
        
    Returns:
        BibTeX字符串
    """
    key = paper.get('bibtex_key', 'unknown')
    authors = ' and '.join(paper.get('authors', ['Unknown']))
    
    fields = [
        f"  author = {{{authors}}}",
        f"  title = {{{paper.get('title', '')}}}",
        f"  year = {{{paper.get('year', '')}}}",
        f"  eprint = {{{paper.get('arxiv_id', '')}}}",
        f"  archivePrefix = {{arXiv}}",
        f"  primaryClass = {{{paper.get('primary_category', '')}}}"
    ]
    
    if paper.get('url'):
        fields.append(f"  url = {{{paper.get('url')}}}")
    
    bibtex = f"@article{{{key},\n"
    bibtex += ",\n".join(fields)
    bibtex += "\n}"
    
    return bibtex


def search_and_export(
    query: str,
    output_file: str,
    limit: int = 20,
    categories: Optional[List[str]] = None
) -> Dict:
    """
    搜索并导出为BibTeX文件
    
    Args:
        query: 搜索查询
        output_file: 输出文件路径
        limit: 返回数量
        categories: 限制分类
        
    Returns:
        搜索结果摘要
    """
    papers = search_papers(query, limit, categories=categories)
    
    bibtex_entries = [generate_bibtex(paper) for paper in papers]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(bibtex_entries))
    
    return {
        'query': query,
        'papers_found': len(papers),
        'output_file': output_file,
        'papers': papers
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # 测试搜索
    papers = search_papers(
        "neural network optimization",
        limit=5,
        categories=['cs.LG', 'stat.ML']
    )
    
    print(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title'][:80]}...")
        print(f"   Authors: {', '.join(paper['authors'][:3])}...")
        print(f"   Year: {paper['year']}")
        print(f"   arXiv ID: {paper['arxiv_id']}")
        print(f"   Categories: {', '.join(paper['categories'][:3])}")
