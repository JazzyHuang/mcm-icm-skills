"""
Google Scholar API 文献检索
通过SerpAPI代理访问Google Scholar搜索结果
需要配置SerpAPI Key
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

# API配置
SERPAPI_BASE_URL = "https://serpapi.com/search"


def search_papers(
    query: str,
    limit: int = 20,
    year_range: Optional[Tuple[int, int]] = None,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    搜索学术论文
    
    Args:
        query: 搜索查询
        limit: 返回数量限制
        year_range: 年份范围 (start, end)
        api_key: SerpAPI密钥
        
    Returns:
        论文列表
    """
    if not api_key:
        api_key = os.environ.get('SERPAPI_KEY')
    
    if not api_key:
        logger.error("SerpAPI key not provided. Set SERPAPI_KEY environment variable or pass api_key parameter.")
        return []
    
    params = {
        'engine': 'google_scholar',
        'q': query,
        'api_key': api_key,
        'num': min(limit, 20),  # Google Scholar每页最多20个结果
        'hl': 'en'
    }
    
    # 添加年份过滤
    if year_range:
        params['as_ylo'] = year_range[0]
        params['as_yhi'] = year_range[1]
    
    logger.info(f"Searching Google Scholar: {query}")
    
    try:
        response = requests.get(SERPAPI_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'error' in data:
            logger.error(f"SerpAPI error: {data['error']}")
            return []
        
        organic_results = data.get('organic_results', [])
        
        # 处理结果
        results = []
        for item in organic_results[:limit]:
            processed = process_result(item)
            if processed:
                results.append(processed)
        
        logger.info(f"Found {len(results)} papers from Google Scholar")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching Google Scholar: {e}")
        return []


def search_citations(
    paper_id: str,
    limit: int = 10,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    获取引用某篇论文的文献
    
    Args:
        paper_id: Google Scholar论文ID（从搜索结果中获取）
        limit: 返回数量限制
        api_key: SerpAPI密钥
        
    Returns:
        引用论文列表
    """
    if not api_key:
        api_key = os.environ.get('SERPAPI_KEY')
    
    if not api_key:
        logger.error("SerpAPI key not provided.")
        return []
    
    params = {
        'engine': 'google_scholar',
        'cites': paper_id,
        'api_key': api_key,
        'num': min(limit, 20),
        'hl': 'en'
    }
    
    try:
        response = requests.get(SERPAPI_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        organic_results = data.get('organic_results', [])
        
        return [process_result(item) for item in organic_results[:limit] if process_result(item)]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting citations: {e}")
        return []


def search_author(
    author_name: str,
    api_key: Optional[str] = None
) -> Optional[Dict]:
    """
    搜索作者信息
    
    Args:
        author_name: 作者姓名
        api_key: SerpAPI密钥
        
    Returns:
        作者信息
    """
    if not api_key:
        api_key = os.environ.get('SERPAPI_KEY')
    
    if not api_key:
        logger.error("SerpAPI key not provided.")
        return None
    
    params = {
        'engine': 'google_scholar_profiles',
        'mauthors': author_name,
        'api_key': api_key,
        'hl': 'en'
    }
    
    try:
        response = requests.get(SERPAPI_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        profiles = data.get('profiles', [])
        
        if profiles:
            return profiles[0]
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching author: {e}")
        return None


def process_result(item: Dict) -> Optional[Dict]:
    """
    处理Google Scholar搜索结果
    
    Args:
        item: 原始搜索结果项
        
    Returns:
        处理后的论文数据
    """
    if not item.get('title'):
        return None
    
    # 提取作者
    publication_info = item.get('publication_info', {})
    authors_str = publication_info.get('authors', [])
    
    if isinstance(authors_str, list):
        authors = [a.get('name', '') for a in authors_str]
    else:
        # 可能是字符串格式
        authors = [authors_str] if authors_str else []
    
    # 提取年份
    summary = publication_info.get('summary', '')
    year = extract_year(summary)
    
    # 提取期刊/来源
    venue = extract_venue(summary)
    
    # 生成BibTeX key
    first_author = authors[0].split()[-1] if authors else 'Unknown'
    title_word = item.get('title', '').split()[0] if item.get('title') else 'paper'
    bibtex_key = f"{first_author.lower()}{year}{title_word.lower()}"
    
    # 清理bibtex_key中的特殊字符
    bibtex_key = ''.join(c for c in bibtex_key if c.isalnum())
    
    # 提取引用信息
    inline_links = item.get('inline_links', {})
    cited_by = inline_links.get('cited_by', {})
    citation_count = cited_by.get('total', 0) if isinstance(cited_by, dict) else 0
    
    return {
        'result_id': item.get('result_id', ''),
        'title': item.get('title', ''),
        'authors': authors,
        'year': year,
        'venue': venue,
        'citation_count': citation_count,
        'snippet': item.get('snippet', ''),
        'link': item.get('link', ''),
        'bibtex_key': bibtex_key,
        'source': 'google_scholar',
        'resources': item.get('resources', [])  # PDF链接等
    }


def extract_year(summary: str) -> int:
    """从摘要信息中提取年份"""
    import re
    
    # 尝试匹配4位数年份
    year_match = re.search(r'\b(19|20)\d{2}\b', summary)
    if year_match:
        return int(year_match.group())
    
    return datetime.now().year


def extract_venue(summary: str) -> str:
    """从摘要信息中提取期刊/来源"""
    # summary格式通常是: "作者 - 期刊, 年份 - 出版商"
    parts = summary.split(' - ')
    if len(parts) >= 2:
        venue_part = parts[1]
        # 移除年份部分
        import re
        venue = re.sub(r',?\s*\d{4}', '', venue_part).strip()
        return venue
    return ''


def generate_bibtex(paper: Dict) -> str:
    """
    生成BibTeX条目
    
    Args:
        paper: 论文数据
        
    Returns:
        BibTeX字符串
    """
    entry_type = 'article' if paper.get('venue') else 'misc'
    key = paper.get('bibtex_key', 'unknown')
    
    # 格式化作者
    authors = ' and '.join(paper.get('authors', ['Unknown']))
    
    lines = [
        f"@{entry_type}{{{key},",
        f"  author = {{{authors}}},",
        f"  title = {{{paper.get('title', '')}}},",
        f"  year = {{{paper.get('year', '')}}},",
    ]
    
    if paper.get('venue'):
        lines.append(f"  journal = {{{paper.get('venue')}}},")
    
    if paper.get('link'):
        lines.append(f"  url = {{{paper.get('link')}}},")
    
    # 添加访问日期（对于网络资源）
    lines.append(f"  note = {{Accessed via Google Scholar on {datetime.now().strftime('%Y-%m-%d')}}},")
    
    # 移除最后一行的逗号
    lines[-1] = lines[-1].rstrip(',')
    lines.append("}")
    
    return '\n'.join(lines)


def search_and_export(
    query: str,
    output_file: str,
    limit: int = 20,
    year_range: Optional[Tuple[int, int]] = None,
    api_key: Optional[str] = None
) -> Dict:
    """
    搜索并导出为BibTeX文件
    
    Args:
        query: 搜索查询
        output_file: 输出文件路径
        limit: 返回数量
        year_range: 年份范围
        api_key: SerpAPI密钥
        
    Returns:
        搜索结果摘要
    """
    papers = search_papers(query, limit, year_range, api_key)
    
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


def check_api_key(api_key: Optional[str] = None) -> bool:
    """
    检查API Key是否有效
    
    Args:
        api_key: SerpAPI密钥
        
    Returns:
        是否有效
    """
    if not api_key:
        api_key = os.environ.get('SERPAPI_KEY')
    
    if not api_key:
        return False
    
    # 执行简单的API调用测试
    params = {
        'engine': 'google_scholar',
        'q': 'test',
        'api_key': api_key,
        'num': 1
    }
    
    try:
        response = requests.get(SERPAPI_BASE_URL, params=params, timeout=10)
        data = response.json()
        return 'error' not in data
    except:
        return False


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 检查API Key
    api_key = os.environ.get('SERPAPI_KEY')
    
    if not api_key:
        print("Warning: SERPAPI_KEY not set. Set it to test Google Scholar API.")
        print("Example: export SERPAPI_KEY='your-api-key'")
    else:
        print("Testing Google Scholar API...")
        papers = search_papers(
            "mathematical modeling optimization",
            limit=5,
            year_range=(2020, 2024),
            api_key=api_key
        )
        
        print(f"\nFound {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'][:3])}...")
            print(f"   Year: {paper['year']}, Citations: {paper['citation_count']}")
            print(f"   Venue: {paper['venue']}")
