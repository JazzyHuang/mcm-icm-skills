"""
OpenAlex API 文献检索
使用OpenAlex免费API检索学术文献（240M+文献，无需API Key）
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

# API配置
BASE_URL = "https://api.openalex.org"
RATE_LIMIT_DELAY = 0.1  # OpenAlex允许较高请求频率
USER_AGENT = "MCM-ICM-AutoPaper/1.0 (mailto:mcm-icm@example.com)"


def search_works(
    query: str,
    limit: int = 20,
    year_range: Optional[Tuple[int, int]] = None,
    sort_by: str = "cited_by_count:desc",
    filters: Optional[Dict[str, str]] = None
) -> List[Dict]:
    """
    搜索学术文献
    
    Args:
        query: 搜索查询
        limit: 返回数量限制
        year_range: 年份范围 (start, end)
        sort_by: 排序方式，默认按引用量降序
        filters: 额外的过滤条件
        
    Returns:
        论文列表
    """
    url = f"{BASE_URL}/works"
    
    params = {
        'search': query,
        'per_page': min(limit, 200),  # OpenAlex最大200
        'sort': sort_by,
        'select': 'id,title,authorships,publication_year,primary_location,cited_by_count,doi,abstract_inverted_index,type,open_access'
    }
    
    # 添加年份过滤
    filter_parts = []
    if year_range:
        filter_parts.append(f"publication_year:{year_range[0]}-{year_range[1]}")
    
    # 添加其他过滤条件
    if filters:
        for key, value in filters.items():
            filter_parts.append(f"{key}:{value}")
    
    if filter_parts:
        params['filter'] = ','.join(filter_parts)
    
    headers = {
        'User-Agent': USER_AGENT
    }
    
    logger.info(f"Searching OpenAlex: {query}")
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        works = data.get('results', [])
        
        # 处理结果
        results = []
        for work in works:
            processed = process_work(work)
            if processed:
                results.append(processed)
        
        logger.info(f"Found {len(results)} works from OpenAlex")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching OpenAlex: {e}")
        return []


def get_work_details(work_id: str) -> Optional[Dict]:
    """
    获取文献详情
    
    Args:
        work_id: OpenAlex Work ID (例如 W2741809807)
        
    Returns:
        文献详情
    """
    # 处理ID格式
    if not work_id.startswith('W') and not work_id.startswith('https://'):
        work_id = f"W{work_id}"
    
    if work_id.startswith('https://'):
        url = work_id
    else:
        url = f"{BASE_URL}/works/{work_id}"
    
    headers = {
        'User-Agent': USER_AGENT
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return process_work(data)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting work details: {e}")
        return None


def search_by_doi(doi: str) -> Optional[Dict]:
    """
    通过DOI搜索文献
    
    Args:
        doi: DOI标识符
        
    Returns:
        文献信息
    """
    # 清理DOI格式
    doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
    
    url = f"{BASE_URL}/works/https://doi.org/{doi}"
    
    headers = {
        'User-Agent': USER_AGENT
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 404:
            logger.warning(f"DOI not found in OpenAlex: {doi}")
            return None
        response.raise_for_status()
        
        data = response.json()
        return process_work(data)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching by DOI: {e}")
        return None


def get_author_works(author_id: str, limit: int = 20) -> List[Dict]:
    """
    获取作者的所有文献
    
    Args:
        author_id: OpenAlex Author ID
        limit: 返回数量限制
        
    Returns:
        文献列表
    """
    url = f"{BASE_URL}/works"
    
    params = {
        'filter': f"authorships.author.id:{author_id}",
        'per_page': min(limit, 200),
        'sort': 'cited_by_count:desc',
        'select': 'id,title,authorships,publication_year,primary_location,cited_by_count,doi'
    }
    
    headers = {
        'User-Agent': USER_AGENT
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        works = data.get('results', [])
        
        return [process_work(w) for w in works if process_work(w)]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting author works: {e}")
        return []


def get_citations(work_id: str, limit: int = 20) -> List[Dict]:
    """
    获取引用该文献的论文
    
    Args:
        work_id: OpenAlex Work ID
        limit: 返回数量限制
        
    Returns:
        引用论文列表
    """
    url = f"{BASE_URL}/works"
    
    params = {
        'filter': f"cites:{work_id}",
        'per_page': min(limit, 200),
        'sort': 'cited_by_count:desc',
        'select': 'id,title,authorships,publication_year,primary_location,cited_by_count,doi'
    }
    
    headers = {
        'User-Agent': USER_AGENT
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        works = data.get('results', [])
        
        return [process_work(w) for w in works if process_work(w)]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting citations: {e}")
        return []


def get_references(work_id: str, limit: int = 20) -> List[Dict]:
    """
    获取该文献的参考文献
    
    Args:
        work_id: OpenAlex Work ID
        limit: 返回数量限制
        
    Returns:
        参考文献列表
    """
    # 首先获取文献详情以获取referenced_works
    url = f"{BASE_URL}/works/{work_id}"
    
    headers = {
        'User-Agent': USER_AGENT
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        referenced_ids = data.get('referenced_works', [])[:limit]
        
        # 获取每个参考文献的详情
        references = []
        for ref_id in referenced_ids:
            ref_detail = get_work_details(ref_id)
            if ref_detail:
                references.append(ref_detail)
            time.sleep(RATE_LIMIT_DELAY)
        
        return references
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting references: {e}")
        return []


def process_work(work: Dict) -> Optional[Dict]:
    """
    处理OpenAlex返回的文献数据
    
    Args:
        work: 原始文献数据
        
    Returns:
        处理后的文献数据
    """
    if not work or not work.get('title'):
        return None
    
    # 提取作者
    authorships = work.get('authorships', [])
    authors = []
    for authorship in authorships:
        author = authorship.get('author', {})
        if author.get('display_name'):
            authors.append(author['display_name'])
    
    # 提取DOI
    doi = work.get('doi', '')
    if doi:
        doi = doi.replace('https://doi.org/', '')
    
    # 提取期刊/来源信息
    primary_location = work.get('primary_location', {}) or {}
    source = primary_location.get('source', {}) or {}
    venue = source.get('display_name', '')
    
    # 提取摘要
    abstract = reconstruct_abstract(work.get('abstract_inverted_index', {}))
    
    # 生成BibTeX key
    first_author = authors[0].split()[-1] if authors else 'Unknown'
    year = work.get('publication_year', datetime.now().year)
    title_word = work.get('title', '').split()[0] if work.get('title') else 'paper'
    bibtex_key = f"{first_author.lower()}{year}{title_word.lower()}"
    
    # 清理bibtex_key中的特殊字符
    bibtex_key = ''.join(c for c in bibtex_key if c.isalnum())
    
    return {
        'work_id': work.get('id', '').replace('https://openalex.org/', ''),
        'title': work.get('title', ''),
        'authors': authors,
        'year': year,
        'venue': venue,
        'citation_count': work.get('cited_by_count', 0),
        'doi': doi,
        'abstract': abstract,
        'type': work.get('type', 'article'),
        'open_access': work.get('open_access', {}).get('is_oa', False),
        'url': work.get('id', ''),  # OpenAlex URL
        'bibtex_key': bibtex_key,
        'source': 'openalex'
    }


def reconstruct_abstract(inverted_index: Dict) -> str:
    """
    从倒排索引重建摘要文本
    
    Args:
        inverted_index: OpenAlex的倒排索引格式摘要
        
    Returns:
        重建的摘要文本
    """
    if not inverted_index:
        return ''
    
    # 构建位置到词的映射
    position_word = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word[pos] = word
    
    # 按位置排序并连接
    if not position_word:
        return ''
    
    max_pos = max(position_word.keys())
    words = [position_word.get(i, '') for i in range(max_pos + 1)]
    
    return ' '.join(words)


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
    
    if paper.get('doi'):
        lines.append(f"  doi = {{{paper.get('doi')}}},")
    
    if paper.get('url'):
        lines.append(f"  url = {{{paper.get('url')}}},")
    
    # 移除最后一行的逗号
    lines[-1] = lines[-1].rstrip(',')
    lines.append("}")
    
    return '\n'.join(lines)


def search_and_export(
    query: str,
    output_file: str,
    limit: int = 20,
    year_range: Optional[Tuple[int, int]] = None
) -> Dict:
    """
    搜索并导出为BibTeX文件
    
    Args:
        query: 搜索查询
        output_file: 输出文件路径
        limit: 返回数量
        year_range: 年份范围
        
    Returns:
        搜索结果摘要
    """
    papers = search_works(query, limit, year_range)
    
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


def search_related_works(seed_work_id: str, limit: int = 10) -> List[Dict]:
    """
    搜索相关文献（基于引用关系）
    
    Args:
        seed_work_id: 种子文献ID
        limit: 返回数量
        
    Returns:
        相关文献列表
    """
    url = f"{BASE_URL}/works"
    
    params = {
        'filter': f"related_to:{seed_work_id}",
        'per_page': min(limit, 50),
        'sort': 'cited_by_count:desc',
        'select': 'id,title,authorships,publication_year,primary_location,cited_by_count,doi'
    }
    
    headers = {
        'User-Agent': USER_AGENT
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        works = data.get('results', [])
        
        return [process_work(w) for w in works if process_work(w)]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting related works: {e}")
        return []


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试搜索
    print("Testing OpenAlex API...")
    papers = search_works(
        "mathematical modeling optimization renewable energy",
        limit=5,
        year_range=(2020, 2024)
    )
    
    print(f"\nFound {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}...")
        print(f"   Year: {paper['year']}, Citations: {paper['citation_count']}")
        print(f"   DOI: {paper['doi']}")
        print(f"   Open Access: {paper['open_access']}")
    
    # 测试DOI搜索
    if papers and papers[0].get('doi'):
        print(f"\nTesting DOI search for: {papers[0]['doi']}")
        result = search_by_doi(papers[0]['doi'])
        if result:
            print(f"Found: {result['title']}")
