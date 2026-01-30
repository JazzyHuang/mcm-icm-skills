"""
引用格式化工具
提供统一的BibTeX生成和引用格式化功能
"""

import re
from typing import Dict, List, Optional


class CitationFormatter:
    """引用格式化器"""
    
    @staticmethod
    def generate_bibtex_key(author: str, year: int, title: str) -> str:
        """
        生成标准BibTeX key
        
        Args:
            author: 作者名（可以是完整名或仅姓氏）
            year: 年份
            title: 标题
            
        Returns:
            BibTeX key字符串
        """
        # 提取姓氏（如果是完整名，取最后一部分）
        author_parts = author.split()
        first_author = author_parts[-1] if author_parts else 'unknown'
        
        # 清理作者名
        clean_author = re.sub(r'[^\w]', '', first_author.lower())
        if not clean_author:
            clean_author = 'unknown'
        
        # 提取标题首词
        title_parts = title.split() if title else ['untitled']
        first_word = title_parts[0] if title_parts else 'untitled'
        clean_title = re.sub(r'[^\w]', '', first_word.lower())
        if not clean_title:
            clean_title = 'untitled'
        
        # 处理年份
        year_str = str(year) if year else 'XXXX'
        
        return f"{clean_author}{year_str}{clean_title}"
    
    @staticmethod
    def format_authors_bibtex(authors: List[str]) -> str:
        """
        格式化作者列表为BibTeX格式
        
        Args:
            authors: 作者名列表
            
        Returns:
            BibTeX格式的作者字符串
        """
        if not authors:
            return 'Unknown'
        return ' and '.join(authors)
    
    @staticmethod
    def escape_bibtex(text: str) -> str:
        """
        转义BibTeX特殊字符
        
        Args:
            text: 原始文本
            
        Returns:
            转义后的文本
        """
        if not text:
            return ''
        
        # 转义特殊字符
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '_': r'\_',
            '#': r'\#',
            '$': r'\$',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}'
        }
        
        for char, escaped in replacements.items():
            text = text.replace(char, escaped)
        
        return text
    
    @staticmethod
    def format_article(
        key: str,
        title: str,
        authors: List[str],
        year: int,
        journal: str = '',
        volume: str = '',
        number: str = '',
        pages: str = '',
        doi: str = '',
        url: str = '',
        abstract: str = ''
    ) -> str:
        """
        格式化期刊文章BibTeX条目
        
        Returns:
            BibTeX字符串
        """
        author_str = CitationFormatter.format_authors_bibtex(authors)
        
        fields = [
            f"  author = {{{author_str}}}",
            f"  title = {{{{{title}}}}}",  # 双括号保护大小写
            f"  year = {{{year}}}"
        ]
        
        if journal:
            fields.append(f"  journal = {{{journal}}}")
        if volume:
            fields.append(f"  volume = {{{volume}}}")
        if number:
            fields.append(f"  number = {{{number}}}")
        if pages:
            fields.append(f"  pages = {{{pages}}}")
        if doi:
            fields.append(f"  doi = {{{doi}}}")
        if url:
            fields.append(f"  url = {{{url}}}")
        if abstract:
            # 摘要通常不包含在BibTeX中，但可以添加
            pass
        
        return f"@article{{{key},\n" + ",\n".join(fields) + "\n}"
    
    @staticmethod
    def format_inproceedings(
        key: str,
        title: str,
        authors: List[str],
        year: int,
        booktitle: str = '',
        pages: str = '',
        doi: str = '',
        url: str = ''
    ) -> str:
        """
        格式化会议论文BibTeX条目
        
        Returns:
            BibTeX字符串
        """
        author_str = CitationFormatter.format_authors_bibtex(authors)
        
        fields = [
            f"  author = {{{author_str}}}",
            f"  title = {{{{{title}}}}}",
            f"  year = {{{year}}}"
        ]
        
        if booktitle:
            fields.append(f"  booktitle = {{{booktitle}}}")
        if pages:
            fields.append(f"  pages = {{{pages}}}")
        if doi:
            fields.append(f"  doi = {{{doi}}}")
        if url:
            fields.append(f"  url = {{{url}}}")
        
        return f"@inproceedings{{{key},\n" + ",\n".join(fields) + "\n}"
    
    @staticmethod
    def format_misc(
        key: str,
        title: str,
        authors: List[str],
        year: int,
        howpublished: str = '',
        note: str = '',
        url: str = ''
    ) -> str:
        """
        格式化杂项BibTeX条目（用于网页、数据集等）
        
        Returns:
            BibTeX字符串
        """
        author_str = CitationFormatter.format_authors_bibtex(authors)
        
        fields = [
            f"  author = {{{author_str}}}",
            f"  title = {{{{{title}}}}}",
            f"  year = {{{year}}}"
        ]
        
        if howpublished:
            fields.append(f"  howpublished = {{{howpublished}}}")
        if note:
            fields.append(f"  note = {{{note}}}")
        if url:
            fields.append(f"  url = {{{url}}}")
        
        return f"@misc{{{key},\n" + ",\n".join(fields) + "\n}"
    
    @staticmethod
    def format_techreport(
        key: str,
        title: str,
        authors: List[str],
        year: int,
        institution: str = '',
        number: str = '',
        url: str = ''
    ) -> str:
        """
        格式化技术报告BibTeX条目
        
        Returns:
            BibTeX字符串
        """
        author_str = CitationFormatter.format_authors_bibtex(authors)
        
        fields = [
            f"  author = {{{author_str}}}",
            f"  title = {{{{{title}}}}}",
            f"  year = {{{year}}}"
        ]
        
        if institution:
            fields.append(f"  institution = {{{institution}}}")
        if number:
            fields.append(f"  number = {{{number}}}")
        if url:
            fields.append(f"  url = {{{url}}}")
        
        return f"@techreport{{{key},\n" + ",\n".join(fields) + "\n}"
    
    @staticmethod
    def format_from_dict(citation: Dict) -> str:
        """
        从字典格式化BibTeX条目
        
        自动检测类型并选择合适的格式
        
        Args:
            citation: 引用数据字典
            
        Returns:
            BibTeX字符串
        """
        # 获取或生成key
        key = citation.get('bibtex_key')
        if not key:
            authors = citation.get('authors', ['Unknown'])
            first_author = authors[0] if authors else 'Unknown'
            key = CitationFormatter.generate_bibtex_key(
                first_author,
                citation.get('year'),
                citation.get('title', '')
            )
        
        title = citation.get('title', '')
        authors = citation.get('authors', ['Unknown'])
        year = citation.get('year', '')
        
        # 检测类型
        cit_type = citation.get('type', '').lower()
        
        if cit_type in ['journal-article', 'article'] or citation.get('journal'):
            return CitationFormatter.format_article(
                key=key,
                title=title,
                authors=authors,
                year=year,
                journal=citation.get('journal') or citation.get('venue', ''),
                volume=citation.get('volume', ''),
                pages=citation.get('pages', ''),
                doi=citation.get('doi', ''),
                url=citation.get('url', '')
            )
        elif cit_type in ['proceedings-article', 'inproceedings'] or citation.get('booktitle'):
            return CitationFormatter.format_inproceedings(
                key=key,
                title=title,
                authors=authors,
                year=year,
                booktitle=citation.get('booktitle') or citation.get('venue', ''),
                pages=citation.get('pages', ''),
                doi=citation.get('doi', ''),
                url=citation.get('url', '')
            )
        elif cit_type in ['report', 'techreport'] or citation.get('institution'):
            return CitationFormatter.format_techreport(
                key=key,
                title=title,
                authors=authors,
                year=year,
                institution=citation.get('institution', ''),
                url=citation.get('url', '')
            )
        else:
            return CitationFormatter.format_misc(
                key=key,
                title=title,
                authors=authors,
                year=year,
                url=citation.get('url', ''),
                note=citation.get('note', '')
            )


# 便捷函数
def generate_bibtex_key(author: str, year: int, title: str) -> str:
    """生成BibTeX key的便捷函数"""
    return CitationFormatter.generate_bibtex_key(author, year, title)


def format_bibtex(citation: Dict) -> str:
    """格式化BibTeX的便捷函数"""
    return CitationFormatter.format_from_dict(citation)


if __name__ == '__main__':
    # 测试代码
    formatter = CitationFormatter()
    
    # 测试key生成
    key = formatter.generate_bibtex_key("John Smith", 2024, "A Novel Approach")
    print(f"Generated key: {key}")
    
    # 测试文章格式化
    citation = {
        'title': 'Deep Learning for Optimization',
        'authors': ['Alice Johnson', 'Bob Smith'],
        'year': 2024,
        'journal': 'Journal of Machine Learning',
        'doi': '10.1234/example'
    }
    
    bibtex = formatter.format_from_dict(citation)
    print(f"\nGenerated BibTeX:\n{bibtex}")
