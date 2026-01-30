"""
PubMed API 文献检索
使用NCBI E-utilities API检索生物医学文献
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
    safe_get_nested,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# API配置
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SEARCH_URL = f"{BASE_URL}/esearch.fcgi"
FETCH_URL = f"{BASE_URL}/efetch.fcgi"
SUMMARY_URL = f"{BASE_URL}/esummary.fcgi"


def search_papers(
    query: str,
    limit: int = 20,
    start: int = 0,
    sort: str = "relevance",
    date_range: Optional[tuple] = None,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    搜索PubMed文献
    
    Args:
        query: 搜索查询 (支持PubMed查询语法)
        limit: 返回数量限制
        start: 起始索引
        sort: 排序方式 (relevance, pub_date)
        date_range: 日期范围 (start_year, end_year)
        api_key: NCBI API密钥 (可选，但建议使用)
        
    Returns:
        论文列表
    """
    # 第一步：搜索获取ID列表
    search_params = {
        'db': 'pubmed',
        'term': query,
        'retmax': min(limit, 10000),
        'retstart': start,
        'sort': sort,
        'retmode': 'json'
    }
    
    if date_range:
        search_params['mindate'] = str(date_range[0])
        search_params['maxdate'] = str(date_range[1])
        search_params['datetype'] = 'pdat'
    
    if api_key:
        search_params['api_key'] = api_key
    
    logger.info(f"Searching PubMed: {query}")
    
    try:
        # 搜索
        search_data = api_request_with_retry(
            SEARCH_URL,
            params=search_params,
            timeout=DEFAULT_TIMEOUT,
            max_retries=DEFAULT_MAX_RETRIES,
            return_json=True
        )
        
        id_list = safe_get_nested(search_data, 'esearchresult', 'idlist', default=[])
        
        if not id_list:
            logger.info("No results found")
            return []
        
        # 获取详情
        papers = fetch_paper_details(id_list, api_key)
        logger.info(f"Found {len(papers)} papers")
        return papers
        
    except APIRequestError as e:
        logger.error(f"Error searching PubMed: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error searching PubMed: {e}")
        return []


def fetch_paper_details(
    pmids: List[str],
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    获取论文详情
    
    Args:
        pmids: PubMed ID列表
        api_key: NCBI API密钥
        
    Returns:
        论文详情列表
    """
    if not pmids:
        return []
    
    params = {
        'db': 'pubmed',
        'id': ','.join(pmids),
        'retmode': 'xml',
        'rettype': 'abstract'
    }
    
    if api_key:
        params['api_key'] = api_key
    
    try:
        response = api_request_with_retry(
            FETCH_URL,
            params=params,
            timeout=DEFAULT_TIMEOUT,
            max_retries=DEFAULT_MAX_RETRIES,
            return_json=False
        )
        
        if hasattr(response, 'text'):
            xml_content = response.text
        else:
            xml_content = str(response)
        
        return parse_pubmed_xml(xml_content)
        
    except APIRequestError as e:
        logger.error(f"Error fetching paper details: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching paper details: {e}")
        return []


def parse_pubmed_xml(xml_content: str) -> List[Dict]:
    """
    解析PubMed XML响应
    
    Args:
        xml_content: XML字符串
        
    Returns:
        论文列表
    """
    results = []
    
    try:
        root = ElementTree.fromstring(xml_content)
        
        for article in root.findall('.//PubmedArticle'):
            paper = parse_article(article)
            if paper:
                results.append(paper)
                
    except ElementTree.ParseError as e:
        logger.error(f"XML parse error: {e}")
    
    return results


def parse_article(article: ElementTree.Element) -> Optional[Dict]:
    """
    解析单篇文章
    
    Args:
        article: XML元素
        
    Returns:
        论文数据
    """
    medline = article.find('.//MedlineCitation')
    if medline is None:
        return None
    
    # 提取PMID
    pmid_elem = medline.find('.//PMID')
    pmid = pmid_elem.text if pmid_elem is not None else None
    
    article_elem = medline.find('.//Article')
    if article_elem is None:
        return None
    
    # 提取标题
    title_elem = article_elem.find('.//ArticleTitle')
    title = title_elem.text if title_elem is not None else ''
    
    if not title:
        return None
    
    # 提取作者
    authors = []
    author_list = article_elem.find('.//AuthorList')
    if author_list is not None:
        for author_elem in author_list.findall('.//Author'):
            last_name = author_elem.find('.//LastName')
            fore_name = author_elem.find('.//ForeName')
            if last_name is not None:
                name = last_name.text or ''
                if fore_name is not None and fore_name.text:
                    name = f"{fore_name.text} {name}"
                authors.append(name)
    
    if not authors:
        authors = ['Unknown']
    
    # 提取摘要
    abstract_text = ''
    abstract_elem = article_elem.find('.//Abstract')
    if abstract_elem is not None:
        abstract_parts = []
        for text_elem in abstract_elem.findall('.//AbstractText'):
            if text_elem.text:
                label = text_elem.get('Label', '')
                if label:
                    abstract_parts.append(f"{label}: {text_elem.text}")
                else:
                    abstract_parts.append(text_elem.text)
        abstract_text = ' '.join(abstract_parts)
    
    # 提取期刊信息
    journal_elem = article_elem.find('.//Journal')
    journal_title = ''
    year = None
    
    if journal_elem is not None:
        journal_title_elem = journal_elem.find('.//Title')
        if journal_title_elem is not None:
            journal_title = journal_title_elem.text or ''
        
        # 提取年份
        pub_date = journal_elem.find('.//PubDate')
        if pub_date is not None:
            year_elem = pub_date.find('.//Year')
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text)
                except ValueError:
                    pass
    
    # 提取DOI
    doi = None
    article_id_list = article.find('.//ArticleIdList')
    if article_id_list is not None:
        for id_elem in article_id_list.findall('.//ArticleId'):
            if id_elem.get('IdType') == 'doi':
                doi = id_elem.text
                break
    
    # 生成BibTeX key
    first_author = authors[0].split()[-1] if authors else 'unknown'
    clean_author = re.sub(r'[^\w]', '', first_author.lower())
    title_parts = title.split() if title else ['untitled']
    clean_title = re.sub(r'[^\w]', '', title_parts[0].lower())
    bibtex_key = f"{clean_author}{year or 'XXXX'}{clean_title}"
    
    return {
        'pmid': pmid,
        'title': title,
        'authors': authors,
        'year': year,
        'journal': journal_title,
        'abstract': abstract_text,
        'doi': doi,
        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else '',
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
        f"  journal = {{{paper.get('journal', '')}}}"
    ]
    
    if paper.get('doi'):
        fields.append(f"  doi = {{{paper.get('doi')}}}")
    
    if paper.get('pmid'):
        fields.append(f"  pmid = {{{paper.get('pmid')}}}")
    
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
    date_range: Optional[tuple] = None,
    api_key: Optional[str] = None
) -> Dict:
    """
    搜索并导出为BibTeX文件
    
    Args:
        query: 搜索查询
        output_file: 输出文件路径
        limit: 返回数量
        date_range: 日期范围
        api_key: NCBI API密钥
        
    Returns:
        搜索结果摘要
    """
    papers = search_papers(query, limit, date_range=date_range, api_key=api_key)
    
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
        "machine learning drug discovery",
        limit=5
    )
    
    print(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title'][:80]}...")
        print(f"   Authors: {', '.join(paper['authors'][:3])}...")
        print(f"   Year: {paper['year']}")
        print(f"   Journal: {paper['journal']}")
        print(f"   PMID: {paper['pmid']}")
