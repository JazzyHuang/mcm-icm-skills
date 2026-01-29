"""
Semantic Scholar API 文献检索
使用Semantic Scholar的学术图谱API检索相关论文
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# API配置
BASE_URL = "https://api.semanticscholar.org/graph/v1"
RATE_LIMIT_DELAY = 1.0  # 秒


def search_papers(
    query: str,
    limit: int = 20,
    year_range: Optional[tuple] = None,
    fields: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    搜索论文
    
    Args:
        query: 搜索查询
        limit: 返回数量限制
        year_range: 年份范围 (start, end)
        fields: 需要返回的字段
        api_key: API密钥
        
    Returns:
        论文列表
    """
    if fields is None:
        fields = [
            'paperId', 'title', 'abstract', 'year', 'venue',
            'citationCount', 'authors', 'externalIds', 'url'
        ]
        
    url = f"{BASE_URL}/paper/search"
    
    params = {
        'query': query,
        'limit': min(limit, 100),  # API限制
        'fields': ','.join(fields)
    }
    
    if year_range:
        params['year'] = f"{year_range[0]}-{year_range[1]}"
        
    headers = {}
    if api_key:
        headers['x-api-key'] = api_key
        
    logger.info(f"Searching Semantic Scholar: {query}")
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        papers = data.get('data', [])
        
        # 处理结果
        results = []
        for paper in papers:
            processed = process_paper(paper)
            if processed:
                results.append(processed)
                
        logger.info(f"Found {len(results)} papers")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching Semantic Scholar: {e}")
        return []


def get_paper_details(
    paper_id: str,
    fields: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> Optional[Dict]:
    """
    获取论文详情
    
    Args:
        paper_id: Semantic Scholar论文ID
        fields: 需要返回的字段
        api_key: API密钥
        
    Returns:
        论文详情
    """
    if fields is None:
        fields = [
            'paperId', 'title', 'abstract', 'year', 'venue',
            'citationCount', 'authors', 'externalIds', 'url',
            'references', 'citations'
        ]
        
    url = f"{BASE_URL}/paper/{paper_id}"
    
    params = {'fields': ','.join(fields)}
    
    headers = {}
    if api_key:
        headers['x-api-key'] = api_key
        
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        return process_paper(response.json())
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting paper details: {e}")
        return None


def get_paper_citations(
    paper_id: str,
    limit: int = 50,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    获取引用该论文的文献
    
    Args:
        paper_id: 论文ID
        limit: 返回数量限制
        api_key: API密钥
        
    Returns:
        引用论文列表
    """
    url = f"{BASE_URL}/paper/{paper_id}/citations"
    
    params = {
        'limit': min(limit, 100),
        'fields': 'paperId,title,year,citationCount,authors'
    }
    
    headers = {}
    if api_key:
        headers['x-api-key'] = api_key
        
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return [
            process_paper(item.get('citingPaper', {}))
            for item in data.get('data', [])
            if item.get('citingPaper')
        ]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting citations: {e}")
        return []


def get_paper_references(
    paper_id: str,
    limit: int = 50,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    获取论文的参考文献
    
    Args:
        paper_id: 论文ID
        limit: 返回数量限制
        api_key: API密钥
        
    Returns:
        参考文献列表
    """
    url = f"{BASE_URL}/paper/{paper_id}/references"
    
    params = {
        'limit': min(limit, 100),
        'fields': 'paperId,title,year,citationCount,authors'
    }
    
    headers = {}
    if api_key:
        headers['x-api-key'] = api_key
        
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return [
            process_paper(item.get('citedPaper', {}))
            for item in data.get('data', [])
            if item.get('citedPaper')
        ]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting references: {e}")
        return []


def process_paper(paper: Dict) -> Optional[Dict]:
    """
    处理论文数据
    
    Args:
        paper: 原始论文数据
        
    Returns:
        处理后的论文数据
    """
    if not paper or not paper.get('title'):
        return None
        
    # 提取作者
    authors = []
    for author in paper.get('authors', []):
        if isinstance(author, dict):
            authors.append(author.get('name', ''))
        elif isinstance(author, str):
            authors.append(author)
            
    # 提取DOI
    external_ids = paper.get('externalIds', {}) or {}
    doi = external_ids.get('DOI')
    arxiv_id = external_ids.get('ArXiv')
    
    # 生成BibTeX key
    first_author = authors[0].split()[-1] if authors else 'Unknown'
    year = paper.get('year', 'XXXX')
    title_word = paper.get('title', '').split()[0] if paper.get('title') else 'untitled'
    bibtex_key = f"{first_author.lower()}{year}{title_word.lower()}"
    
    return {
        'paper_id': paper.get('paperId'),
        'title': paper.get('title'),
        'authors': authors,
        'year': year,
        'venue': paper.get('venue', ''),
        'citation_count': paper.get('citationCount', 0),
        'abstract': paper.get('abstract', ''),
        'doi': doi,
        'arxiv_id': arxiv_id,
        'url': paper.get('url', ''),
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
    entry_type = 'article'
    key = paper.get('bibtex_key', 'unknown')
    
    # 格式化作者
    authors = ' and '.join(paper.get('authors', ['Unknown']))
    
    fields = [
        f"  author = {{{authors}}}",
        f"  title = {{{paper.get('title', '')}}}",
        f"  year = {{{paper.get('year', '')}}}",
    ]
    
    if paper.get('venue'):
        fields.append(f"  journal = {{{paper.get('venue')}}}")
        
    if paper.get('doi'):
        fields.append(f"  doi = {{{paper.get('doi')}}}")
        
    if paper.get('url'):
        fields.append(f"  url = {{{paper.get('url')}}}")
        
    bibtex = f"@{entry_type}{{{key},\n"
    bibtex += ",\n".join(fields)
    bibtex += "\n}"
    
    return bibtex


def search_and_export(
    query: str,
    output_file: str,
    limit: int = 20,
    year_range: Optional[tuple] = None,
    api_key: Optional[str] = None
) -> Dict:
    """
    搜索并导出为BibTeX文件
    
    Args:
        query: 搜索查询
        output_file: 输出文件路径
        limit: 返回数量
        year_range: 年份范围
        api_key: API密钥
        
    Returns:
        搜索结果摘要
    """
    papers = search_papers(query, limit, year_range, api_key=api_key)
    
    # 生成BibTeX
    bibtex_entries = []
    for paper in papers:
        bibtex = generate_bibtex(paper)
        bibtex_entries.append(bibtex)
        
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(bibtex_entries))
        
    return {
        'query': query,
        'papers_found': len(papers),
        'output_file': output_file,
        'papers': papers
    }


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试搜索
    papers = search_papers(
        "mathematical modeling optimization solar energy",
        limit=5,
        year_range=(2020, 2024)
    )
    
    print(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}...")
        print(f"   Year: {paper['year']}, Citations: {paper['citation_count']}")
        print(f"   DOI: {paper['doi']}")
