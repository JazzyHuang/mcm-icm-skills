"""
CrossRef API 文献检索
使用CrossRef API检索学术文献元数据
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加common模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common.api_utils import (
    api_request_with_retry,
    APIRequestError,
    safe_get_nested,
    safe_list_get,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# API配置
BASE_URL = "https://api.crossref.org"
USER_AGENT = "MCM-ICM-Skills/1.0 (mailto:research@example.com)"


def search_works(
    query: str,
    limit: int = 20,
    offset: int = 0,
    filter_params: Optional[Dict[str, str]] = None,
    sort: str = "relevance",
    order: str = "desc"
) -> List[Dict]:
    """
    搜索CrossRef文献
    
    Args:
        query: 搜索查询
        limit: 返回数量限制 (最大1000)
        offset: 偏移量
        filter_params: 过滤参数 (如 from-pub-date, type)
        sort: 排序字段 (relevance, published, indexed, score)
        order: 排序方向 (asc, desc)
        
    Returns:
        文献列表
    """
    url = f"{BASE_URL}/works"
    
    params = {
        'query': query,
        'rows': min(limit, 1000),
        'offset': offset,
        'sort': sort,
        'order': order
    }
    
    # 添加过滤参数
    if filter_params:
        filter_str = ','.join(f"{k}:{v}" for k, v in filter_params.items())
        params['filter'] = filter_str
    
    headers = {
        'User-Agent': USER_AGENT
    }
    
    logger.info(f"Searching CrossRef: {query}")
    
    try:
        data = api_request_with_retry(
            url,
            params=params,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
            max_retries=DEFAULT_MAX_RETRIES,
            return_json=True
        )
        
        items = safe_get_nested(data, 'message', 'items', default=[])
        
        results = []
        for item in items:
            processed = process_work(item)
            if processed:
                results.append(processed)
        
        logger.info(f"Found {len(results)} works")
        return results
        
    except APIRequestError as e:
        logger.error(f"Error searching CrossRef: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error searching CrossRef: {e}")
        return []


def get_work_by_doi(doi: str) -> Optional[Dict]:
    """
    通过DOI获取文献详情
    
    Args:
        doi: DOI标识符
        
    Returns:
        文献详情
    """
    url = f"{BASE_URL}/works/{doi}"
    
    headers = {
        'User-Agent': USER_AGENT
    }
    
    try:
        data = api_request_with_retry(
            url,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
            max_retries=DEFAULT_MAX_RETRIES,
            return_json=True
        )
        
        message = data.get('message', {}) if isinstance(data, dict) else {}
        return process_work(message)
        
    except APIRequestError as e:
        logger.error(f"Error getting work by DOI {doi}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting work by DOI: {e}")
        return None


def process_work(work: Dict) -> Optional[Dict]:
    """
    处理CrossRef文献数据
    
    Args:
        work: 原始文献数据
        
    Returns:
        处理后的文献数据
    """
    if not work:
        return None
    
    # 提取标题
    title_list = work.get('title', [])
    title = safe_list_get(title_list, 0, '')
    
    if not title:
        return None
    
    # 提取作者
    authors = []
    for author in work.get('author', []) or []:
        given = author.get('given', '')
        family = author.get('family', '')
        if given and family:
            authors.append(f"{given} {family}")
        elif family:
            authors.append(family)
    
    if not authors:
        authors = ['Unknown']
    
    # 提取年份
    year = None
    for date_field in ['published-print', 'published-online', 'created']:
        date_parts = safe_get_nested(work, date_field, 'date-parts', default=[[]])
        if date_parts:
            first_date = safe_list_get(date_parts, 0, [])
            if first_date:
                year = safe_list_get(first_date, 0)
                if year:
                    break
    
    # 提取期刊名
    container = work.get('container-title', [])
    venue = safe_list_get(container, 0, '')
    
    # 生成BibTeX key
    import re
    first_author = authors[0].split()[-1] if authors else 'unknown'
    clean_author = re.sub(r'[^\w]', '', first_author.lower())
    title_parts = title.split() if title else ['untitled']
    clean_title = re.sub(r'[^\w]', '', title_parts[0].lower())
    bibtex_key = f"{clean_author}{year or 'XXXX'}{clean_title}"
    
    return {
        'doi': work.get('DOI'),
        'title': title,
        'authors': authors,
        'year': year,
        'venue': venue,
        'type': work.get('type', ''),
        'publisher': work.get('publisher', ''),
        'url': work.get('URL', ''),
        'citation_count': work.get('is-referenced-by-count', 0),
        'abstract': work.get('abstract', ''),
        'bibtex_key': bibtex_key
    }


def generate_bibtex(work: Dict) -> str:
    """
    生成BibTeX条目
    
    Args:
        work: 文献数据
        
    Returns:
        BibTeX字符串
    """
    work_type = work.get('type', 'article')
    
    # 映射CrossRef类型到BibTeX类型
    type_mapping = {
        'journal-article': 'article',
        'book': 'book',
        'book-chapter': 'inbook',
        'proceedings-article': 'inproceedings',
        'report': 'techreport'
    }
    entry_type = type_mapping.get(work_type, 'misc')
    
    key = work.get('bibtex_key', 'unknown')
    authors = ' and '.join(work.get('authors', ['Unknown']))
    
    fields = [
        f"  author = {{{authors}}}",
        f"  title = {{{work.get('title', '')}}}",
        f"  year = {{{work.get('year', '')}}}",
    ]
    
    if work.get('venue'):
        if entry_type == 'article':
            fields.append(f"  journal = {{{work.get('venue')}}}")
        elif entry_type == 'inproceedings':
            fields.append(f"  booktitle = {{{work.get('venue')}}}")
    
    if work.get('doi'):
        fields.append(f"  doi = {{{work.get('doi')}}}")
    
    if work.get('url'):
        fields.append(f"  url = {{{work.get('url')}}}")
    
    if work.get('publisher'):
        fields.append(f"  publisher = {{{work.get('publisher')}}}")
    
    bibtex = f"@{entry_type}{{{key},\n"
    bibtex += ",\n".join(fields)
    bibtex += "\n}"
    
    return bibtex


def search_and_export(
    query: str,
    output_file: str,
    limit: int = 20,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None
) -> Dict:
    """
    搜索并导出为BibTeX文件
    
    Args:
        query: 搜索查询
        output_file: 输出文件路径
        limit: 返回数量
        year_from: 起始年份
        year_to: 结束年份
        
    Returns:
        搜索结果摘要
    """
    filter_params = {}
    if year_from:
        filter_params['from-pub-date'] = str(year_from)
    if year_to:
        filter_params['until-pub-date'] = str(year_to)
    
    works = search_works(query, limit, filter_params=filter_params if filter_params else None)
    
    bibtex_entries = [generate_bibtex(work) for work in works]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(bibtex_entries))
    
    return {
        'query': query,
        'works_found': len(works),
        'output_file': output_file,
        'works': works
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # 测试搜索
    works = search_works(
        "machine learning optimization",
        limit=5
    )
    
    print(f"Found {len(works)} works:")
    for i, work in enumerate(works, 1):
        print(f"\n{i}. {work['title']}")
        print(f"   Authors: {', '.join(work['authors'][:3])}...")
        print(f"   Year: {work['year']}, Citations: {work['citation_count']}")
        print(f"   DOI: {work['doi']}")
