"""
DOI验证器
验证学术引用的真实性
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# CrossRef API
CROSSREF_URL = "https://api.crossref.org/works"


def validate_doi(doi: str) -> Tuple[bool, Optional[Dict]]:
    """
    验证DOI是否有效
    
    Args:
        doi: DOI字符串
        
    Returns:
        (是否有效, 元数据)
    """
    if not doi:
        return False, None
        
    # 清理DOI
    doi = clean_doi(doi)
    
    url = f"{CROSSREF_URL}/{doi}"
    
    try:
        data = api_request_with_retry(
            url,
            timeout=DEFAULT_TIMEOUT,
            max_retries=DEFAULT_MAX_RETRIES,
            return_json=True,
            raise_for_status=False  # 手动处理404
        )
        
        # 检查是否为Response对象（非JSON响应）
        if not isinstance(data, dict):
            # 检查响应状态码
            if hasattr(data, 'status_code'):
                if data.status_code == 404:
                    logger.warning(f"DOI not found: {doi}")
                    return False, None
                else:
                    logger.error(f"Error validating DOI {doi}: {data.status_code}")
                    return False, None
            return False, None
        
        message = data.get('message', {})
        
        # 安全提取标题和期刊名
        title_list = message.get('title')
        title = safe_list_get(title_list, 0, '') if title_list else ''
        
        venue_list = message.get('container-title')
        venue = safe_list_get(venue_list, 0, '') if venue_list else ''
        
        metadata = {
            'doi': message.get('DOI'),
            'title': title,
            'authors': extract_authors(message),
            'year': extract_year(message),
            'venue': venue,
            'type': message.get('type', ''),
            'publisher': message.get('publisher', '')
        }
        
        return True, metadata
            
    except APIRequestError as e:
        if hasattr(e, 'status_code') and e.status_code == 404:
            logger.warning(f"DOI not found: {doi}")
            return False, None
        logger.error(f"Error validating DOI {doi}: {e}")
        return False, None
    except Exception as e:
        logger.error(f"Unexpected error validating DOI: {e}")
        return False, None


def clean_doi(doi: str) -> str:
    """清理DOI字符串"""
    # 移除URL前缀
    doi = re.sub(r'^https?://doi\.org/', '', doi)
    doi = re.sub(r'^doi:', '', doi, flags=re.IGNORECASE)
    return doi.strip()


def extract_authors(message: Dict) -> List[str]:
    """提取作者列表"""
    authors = []
    for author in message.get('author', []):
        name_parts = []
        if author.get('given'):
            name_parts.append(author['given'])
        if author.get('family'):
            name_parts.append(author['family'])
        if name_parts:
            authors.append(' '.join(name_parts))
    return authors


def extract_year(message: Dict) -> Optional[int]:
    """提取发表年份"""
    # 尝试不同的日期字段
    for field in ['published-print', 'published-online', 'created']:
        date_parts = safe_get_nested(message, field, 'date-parts', default=[[]])
        if date_parts:
            first_date = safe_list_get(date_parts, 0, [])
            if first_date:
                year = safe_list_get(first_date, 0)
                if year is not None:
                    return year
    return None


def validate_citation(citation: Dict) -> Dict:
    """
    验证单个引用
    
    Args:
        citation: 引用信息字典
        
    Returns:
        验证结果
    """
    result = {
        'bibtex_key': citation.get('bibtex_key', 'unknown'),
        'original': citation,
        'status': 'unverified',
        'verification_method': None,
        'confidence': 0.0,
        'warnings': []
    }
    
    # 1. 尝试DOI验证
    doi = citation.get('doi')
    if doi:
        is_valid, metadata = validate_doi(doi)
        if is_valid:
            result['status'] = 'verified'
            result['verification_method'] = 'doi'
            result['confidence'] = 0.95
            result['verified_metadata'] = metadata
            
            # 检查标题匹配
            if citation.get('title') and metadata.get('title'):
                title_match = compare_titles(citation['title'], metadata['title'])
                if not title_match:
                    result['warnings'].append('标题与DOI记录不完全匹配')
                    result['confidence'] = 0.8
                    
            return result
            
    # 2. 尝试标题搜索验证
    title = citation.get('title')
    if title:
        search_result = search_by_title(title)
        if search_result:
            result['status'] = 'likely_valid'
            result['verification_method'] = 'title_search'
            result['confidence'] = search_result.get('confidence', 0.7)
            result['verified_metadata'] = search_result.get('metadata')
            
            if search_result.get('confidence', 0) < 0.8:
                result['warnings'].append('标题搜索结果匹配度较低')
                
            return result
            
    # 3. 无法验证
    result['status'] = 'suspicious'
    result['confidence'] = 0.1
    result['warnings'].append('无法通过DOI或标题搜索验证此引用')
    
    return result


def search_by_title(title: str) -> Optional[Dict]:
    """
    通过标题搜索验证
    
    Args:
        title: 论文标题
        
    Returns:
        搜索结果
    """
    url = f"{CROSSREF_URL}"
    params = {
        'query.title': title,
        'rows': 3
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('message', {}).get('items', [])
            
            for item in items:
                item_title = item.get('title', [''])[0] if item.get('title') else ''
                similarity = compare_titles(title, item_title)
                
                if similarity > 0.8:
                    return {
                        'confidence': similarity,
                        'metadata': {
                            'doi': item.get('DOI'),
                            'title': item_title,
                            'authors': extract_authors(item),
                            'year': extract_year(item),
                            'venue': item.get('container-title', [''])[0] if item.get('container-title') else ''
                        }
                    }
                    
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching by title: {e}")
        return None


def compare_titles(title1: str, title2: str) -> float:
    """
    比较两个标题的相似度
    
    Args:
        title1: 第一个标题
        title2: 第二个标题
        
    Returns:
        相似度分数 (0-1)
    """
    if not title1 or not title2:
        return 0.0
        
    # 简单的Jaccard相似度
    words1 = set(title1.lower().split())
    words2 = set(title2.lower().split())
    
    # 移除常见停用词
    stopwords = {'the', 'a', 'an', 'of', 'in', 'on', 'for', 'and', 'or', 'to'}
    words1 = words1 - stopwords
    words2 = words2 - stopwords
    
    if not words1 or not words2:
        return 0.0
        
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def validate_bibtex_file(bibtex_content: str) -> Dict:
    """
    验证BibTeX文件中的所有引用
    
    Args:
        bibtex_content: BibTeX文件内容
        
    Returns:
        验证报告
    """
    # 解析BibTeX条目
    citations = parse_bibtex(bibtex_content)
    
    results = []
    verified_count = 0
    suspicious_count = 0
    
    for citation in citations:
        result = validate_citation(citation)
        results.append(result)
        
        if result['status'] == 'verified':
            verified_count += 1
        elif result['status'] == 'suspicious':
            suspicious_count += 1
            
    return {
        'citations': results,
        'summary': {
            'total': len(citations),
            'verified': verified_count,
            'suspicious': suspicious_count,
            'verification_rate': verified_count / len(citations) if citations else 0
        }
    }


def parse_bibtex(content: str) -> List[Dict]:
    """
    简单的BibTeX解析
    
    Args:
        content: BibTeX内容
        
    Returns:
        引用列表
    """
    citations = []
    
    # 匹配BibTeX条目
    pattern = r'@(\w+)\{([^,]+),([^@]*)\}'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for entry_type, key, fields in matches:
        citation = {
            'entry_type': entry_type,
            'bibtex_key': key.strip()
        }
        
        # 提取字段
        field_pattern = r'(\w+)\s*=\s*\{([^}]*)\}'
        field_matches = re.findall(field_pattern, fields)
        
        for field_name, field_value in field_matches:
            citation[field_name.lower()] = field_value.strip()
            
        citations.append(citation)
        
    return citations


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试DOI验证
    test_doi = "10.1038/nature12373"
    is_valid, metadata = validate_doi(test_doi)
    print(f"DOI {test_doi} is valid: {is_valid}")
    if metadata:
        print(f"Title: {metadata.get('title')}")
        print(f"Authors: {', '.join(metadata.get('authors', []))}")
        print(f"Year: {metadata.get('year')}")
