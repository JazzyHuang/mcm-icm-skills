"""
数据源引用生成器
为从各种数据源获取的数据自动生成BibTeX引用
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DataCitation:
    """数据源引用"""
    bibtex_key: str
    title: str
    author: str
    year: int
    url: str
    accessed_date: str
    source_type: str  # worldbank, un, oecd, kaggle, github, etc.
    note: Optional[str] = None
    indicator: Optional[str] = None
    countries: Optional[List[str]] = None
    date_range: Optional[str] = None
    extra_fields: Dict[str, str] = field(default_factory=dict)
    
    def to_bibtex(self) -> str:
        """生成BibTeX条目"""
        lines = [
            f"@online{{{self.bibtex_key},",
            f"  author = {{{self.author}}},",
            f"  title = {{{{{self.title}}}}},",
            f"  year = {{{self.year}}},",
            f"  url = {{{self.url}}},",
            f"  urldate = {{{self.accessed_date}}},"
        ]
        
        # 添加note
        note_parts = []
        if self.countries:
            note_parts.append(f"Countries: {', '.join(self.countries)}")
        if self.date_range:
            note_parts.append(f"Period: {self.date_range}")
        if self.note:
            note_parts.append(self.note)
        
        if note_parts:
            lines.append(f"  note = {{{'; '.join(note_parts)}}}")
        
        lines[-1] = lines[-1].rstrip(',')
        lines.append("}")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'bibtex_key': self.bibtex_key,
            'title': self.title,
            'author': self.author,
            'year': self.year,
            'url': self.url,
            'accessed_date': self.accessed_date,
            'source_type': self.source_type,
            'note': self.note,
            'indicator': self.indicator,
            'countries': self.countries,
            'date_range': self.date_range,
            'category': 'data',  # 用于多样性验证
            'bibtex': self.to_bibtex()
        }


class DataCitationGenerator:
    """数据源引用生成器"""
    
    # 数据源配置
    SOURCE_CONFIG = {
        'worldbank': {
            'author': '{World Bank}',
            'base_url': 'https://data.worldbank.org/indicator/',
            'title_prefix': 'World Bank Open Data'
        },
        'un': {
            'author': '{United Nations}',
            'base_url': 'https://data.un.org/',
            'title_prefix': 'UN Data'
        },
        'oecd': {
            'author': '{OECD}',
            'base_url': 'https://data.oecd.org/',
            'title_prefix': 'OECD Data'
        },
        'who': {
            'author': '{World Health Organization}',
            'base_url': 'https://www.who.int/data/',
            'title_prefix': 'WHO Data'
        },
        'imf': {
            'author': '{International Monetary Fund}',
            'base_url': 'https://data.imf.org/',
            'title_prefix': 'IMF Data'
        },
        'kaggle': {
            'author': '{Kaggle}',
            'base_url': 'https://www.kaggle.com/datasets/',
            'title_prefix': 'Kaggle Dataset'
        },
        'github': {
            'author': '{GitHub}',
            'base_url': 'https://github.com/',
            'title_prefix': 'GitHub Repository'
        },
        'census': {
            'author': '{U.S. Census Bureau}',
            'base_url': 'https://data.census.gov/',
            'title_prefix': 'U.S. Census Data'
        },
        'eurostat': {
            'author': '{Eurostat}',
            'base_url': 'https://ec.europa.eu/eurostat/',
            'title_prefix': 'Eurostat Data'
        }
    }
    
    def __init__(self):
        """初始化生成器"""
        self.citations: List[DataCitation] = []
    
    def generate_worldbank_citation(
        self,
        indicator: str,
        indicator_name: Optional[str] = None,
        countries: Optional[List[str]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> DataCitation:
        """
        生成World Bank数据引用
        
        Args:
            indicator: 指标代码 (如 NY.GDP.PCAP.CD)
            indicator_name: 指标名称 (如 GDP per capita)
            countries: 国家列表
            start_year: 起始年份
            end_year: 结束年份
            
        Returns:
            DataCitation对象
        """
        config = self.SOURCE_CONFIG['worldbank']
        year = datetime.now().year
        accessed = datetime.now().strftime('%Y-%m-%d')
        
        # 生成bibtex_key
        indicator_key = indicator.replace('.', '_').lower()
        bibtex_key = f"worldbank_{indicator_key}_{year}"
        
        # 生成标题
        if indicator_name:
            title = f"{config['title_prefix']}: {indicator_name}"
        else:
            title = f"{config['title_prefix']}: {indicator}"
        
        # 生成URL
        url = f"{config['base_url']}{indicator}"
        
        # 生成日期范围
        date_range = None
        if start_year and end_year:
            date_range = f"{start_year}-{end_year}"
        elif start_year:
            date_range = f"{start_year}-present"
        
        citation = DataCitation(
            bibtex_key=bibtex_key,
            title=title,
            author=config['author'],
            year=year,
            url=url,
            accessed_date=accessed,
            source_type='worldbank',
            indicator=indicator,
            countries=countries,
            date_range=date_range
        )
        
        self.citations.append(citation)
        return citation
    
    def generate_un_citation(
        self,
        dataset_name: str,
        url: Optional[str] = None,
        note: Optional[str] = None
    ) -> DataCitation:
        """生成UN Data引用"""
        config = self.SOURCE_CONFIG['un']
        year = datetime.now().year
        accessed = datetime.now().strftime('%Y-%m-%d')
        
        # 生成bibtex_key
        key_name = dataset_name.lower().replace(' ', '_')[:30]
        bibtex_key = f"undata_{key_name}_{year}"
        
        citation = DataCitation(
            bibtex_key=bibtex_key,
            title=f"{config['title_prefix']}: {dataset_name}",
            author=config['author'],
            year=year,
            url=url or config['base_url'],
            accessed_date=accessed,
            source_type='un',
            note=note
        )
        
        self.citations.append(citation)
        return citation
    
    def generate_oecd_citation(
        self,
        dataset_name: str,
        url: Optional[str] = None,
        note: Optional[str] = None
    ) -> DataCitation:
        """生成OECD Data引用"""
        config = self.SOURCE_CONFIG['oecd']
        year = datetime.now().year
        accessed = datetime.now().strftime('%Y-%m-%d')
        
        # 生成bibtex_key
        key_name = dataset_name.lower().replace(' ', '_')[:30]
        bibtex_key = f"oecd_{key_name}_{year}"
        
        citation = DataCitation(
            bibtex_key=bibtex_key,
            title=f"{config['title_prefix']}: {dataset_name}",
            author=config['author'],
            year=year,
            url=url or config['base_url'],
            accessed_date=accessed,
            source_type='oecd',
            note=note
        )
        
        self.citations.append(citation)
        return citation
    
    def generate_generic_citation(
        self,
        source_type: str,
        dataset_name: str,
        url: str,
        author: Optional[str] = None,
        note: Optional[str] = None
    ) -> DataCitation:
        """
        生成通用数据源引用
        
        Args:
            source_type: 数据源类型
            dataset_name: 数据集名称
            url: 数据URL
            author: 作者/机构（可选，会尝试从配置获取）
            note: 备注
            
        Returns:
            DataCitation对象
        """
        config = self.SOURCE_CONFIG.get(source_type, {})
        year = datetime.now().year
        accessed = datetime.now().strftime('%Y-%m-%d')
        
        # 生成bibtex_key
        key_name = dataset_name.lower().replace(' ', '_')[:30]
        key_name = ''.join(c for c in key_name if c.isalnum() or c == '_')
        bibtex_key = f"{source_type}_{key_name}_{year}"
        
        # 确定作者
        if not author:
            author = config.get('author', f'{{{source_type.capitalize()}}}')
        
        # 确定标题前缀
        title_prefix = config.get('title_prefix', source_type.capitalize())
        
        citation = DataCitation(
            bibtex_key=bibtex_key,
            title=f"{title_prefix}: {dataset_name}",
            author=author,
            year=year,
            url=url,
            accessed_date=accessed,
            source_type=source_type,
            note=note
        )
        
        self.citations.append(citation)
        return citation
    
    def get_all_citations(self) -> List[DataCitation]:
        """获取所有生成的引用"""
        return self.citations
    
    def export_bibtex(self, output_path: str) -> None:
        """导出为BibTeX文件"""
        bibtex_entries = [c.to_bibtex() for c in self.citations]
        
        content = [
            "% ===========================================",
            "% Data Source Citations",
            f"% Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "% ==========================================="
        ]
        content.extend(['', ''])
        content.append('\n\n'.join(bibtex_entries))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        logger.info(f"Exported {len(self.citations)} data citations to {output_path}")
    
    def export_json(self, output_path: str) -> None:
        """导出为JSON文件"""
        data = {
            'generated_at': datetime.now().isoformat(),
            'total_citations': len(self.citations),
            'citations': [c.to_dict() for c in self.citations]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(self.citations)} data citations to {output_path}")
    
    def clear(self) -> None:
        """清空引用列表"""
        self.citations = []


# 便捷函数
def generate_worldbank_citation(
    indicator: str,
    indicator_name: Optional[str] = None,
    countries: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> Dict[str, Any]:
    """便捷函数：生成World Bank数据引用"""
    generator = DataCitationGenerator()
    citation = generator.generate_worldbank_citation(
        indicator, indicator_name, countries, start_year, end_year
    )
    return citation.to_dict()


def generate_data_citation(
    source_type: str,
    dataset_name: str,
    url: str,
    **kwargs
) -> Dict[str, Any]:
    """便捷函数：生成通用数据源引用"""
    generator = DataCitationGenerator()
    citation = generator.generate_generic_citation(
        source_type, dataset_name, url, **kwargs
    )
    return citation.to_dict()


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    generator = DataCitationGenerator()
    
    # 测试World Bank引用
    wb_citation = generator.generate_worldbank_citation(
        indicator="NY.GDP.PCAP.CD",
        indicator_name="GDP per capita (current US$)",
        countries=["USA", "CHN", "DEU"],
        start_year=2015,
        end_year=2023
    )
    print("World Bank Citation:")
    print(wb_citation.to_bibtex())
    print()
    
    # 测试UN引用
    un_citation = generator.generate_un_citation(
        dataset_name="Population Indicators",
        note="Global population statistics"
    )
    print("UN Data Citation:")
    print(un_citation.to_bibtex())
    print()
    
    # 测试OECD引用
    oecd_citation = generator.generate_oecd_citation(
        dataset_name="Education Statistics",
        note="Education expenditure indicators"
    )
    print("OECD Data Citation:")
    print(oecd_citation.to_bibtex())
    print()
    
    # 打印所有引用
    print(f"\nTotal citations generated: {len(generator.get_all_citations())}")
