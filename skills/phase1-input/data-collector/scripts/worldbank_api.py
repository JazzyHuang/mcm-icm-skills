"""
World Bank API 数据获取
从世界银行开放数据获取经济、社会、环境指标
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# World Bank API 基础URL
BASE_URL = "https://api.worldbank.org/v2"

# 常用指标代码
COMMON_INDICATORS = {
    # 经济指标
    'gdp': 'NY.GDP.MKTP.CD',
    'gdp_per_capita': 'NY.GDP.PCAP.CD',
    'gdp_growth': 'NY.GDP.MKTP.KD.ZG',
    'inflation': 'FP.CPI.TOTL.ZG',
    'unemployment': 'SL.UEM.TOTL.ZS',
    
    # 人口指标
    'population': 'SP.POP.TOTL',
    'population_growth': 'SP.POP.GROW',
    'urban_population': 'SP.URB.TOTL.IN.ZS',
    'life_expectancy': 'SP.DYN.LE00.IN',
    
    # 环境指标
    'co2_emissions': 'EN.ATM.CO2E.PC',
    'renewable_energy': 'EG.FEC.RNEW.ZS',
    'forest_area': 'AG.LND.FRST.ZS',
    
    # 教育指标
    'literacy_rate': 'SE.ADT.LITR.ZS',
    'school_enrollment': 'SE.PRM.ENRR',
    
    # 健康指标
    'health_expenditure': 'SH.XPD.CHEX.PC.CD',
    'infant_mortality': 'SP.DYN.IMRT.IN',
}


def fetch_worldbank_data(
    indicator: str,
    countries: Union[str, List[str]] = 'all',
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    per_page: int = 1000
) -> pd.DataFrame:
    """
    从World Bank API获取数据
    
    Args:
        indicator: 指标代码或常用指标名称
        countries: 国家代码列表或'all'
        start_year: 起始年份
        end_year: 结束年份
        per_page: 每页数据量
        
    Returns:
        包含数据的DataFrame
    """
    # 处理常用指标别名
    if indicator in COMMON_INDICATORS:
        indicator = COMMON_INDICATORS[indicator]
        
    # 处理国家参数
    if isinstance(countries, list):
        countries_param = ';'.join(countries)
    else:
        countries_param = countries
        
    # 构建URL
    url = f"{BASE_URL}/country/{countries_param}/indicator/{indicator}"
    
    # 构建参数
    params = {
        'format': 'json',
        'per_page': per_page
    }
    
    if start_year:
        params['date'] = f"{start_year}:{end_year or datetime.now().year}"
        
    logger.info(f"Fetching World Bank data: {indicator}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # World Bank API返回格式: [metadata, data]
        if len(data) < 2 or data[1] is None:
            logger.warning(f"No data found for indicator: {indicator}")
            return pd.DataFrame()
            
        records = data[1]
        
        # 转换为DataFrame
        df = pd.DataFrame(records)
        
        # 选择需要的列
        columns_mapping = {
            'country': 'country',
            'countryiso3code': 'country_code',
            'date': 'year',
            'value': 'value',
            'indicator': 'indicator'
        }
        
        # 提取嵌套字段
        if 'country' in df.columns and isinstance(df['country'].iloc[0], dict):
            df['country_name'] = df['country'].apply(lambda x: x.get('value', ''))
            df['country_code'] = df['country'].apply(lambda x: x.get('id', ''))
            
        if 'indicator' in df.columns and isinstance(df['indicator'].iloc[0], dict):
            df['indicator_name'] = df['indicator'].apply(lambda x: x.get('value', ''))
            df['indicator_code'] = df['indicator'].apply(lambda x: x.get('id', ''))
            
        # 清理数据
        df['year'] = pd.to_numeric(df['date'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # 选择最终列
        final_columns = ['country_name', 'country_code', 'year', 'value', 
                        'indicator_name', 'indicator_code']
        df = df[[c for c in final_columns if c in df.columns]]
        
        logger.info(f"Fetched {len(df)} records")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching World Bank data: {e}")
        raise


def fetch_multiple_indicators(
    indicators: List[str],
    countries: Union[str, List[str]] = 'all',
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    获取多个指标的数据并合并
    
    Args:
        indicators: 指标列表
        countries: 国家列表
        start_year: 起始年份
        end_year: 结束年份
        
    Returns:
        合并后的DataFrame，每个指标一列
    """
    all_data = []
    
    for indicator in indicators:
        df = fetch_worldbank_data(indicator, countries, start_year, end_year)
        if not df.empty:
            # 使用指标代码作为列名
            indicator_code = df['indicator_code'].iloc[0] if 'indicator_code' in df.columns else indicator
            df = df.rename(columns={'value': indicator_code})
            all_data.append(df[['country_code', 'year', indicator_code]])
            
    if not all_data:
        return pd.DataFrame()
        
    # 合并所有数据
    result = all_data[0]
    for df in all_data[1:]:
        result = result.merge(df, on=['country_code', 'year'], how='outer')
        
    return result.sort_values(['country_code', 'year'])


def search_indicators(keyword: str, limit: int = 20) -> List[Dict]:
    """
    搜索指标
    
    Args:
        keyword: 搜索关键词
        limit: 返回结果数量限制
        
    Returns:
        匹配的指标列表
    """
    url = f"{BASE_URL}/indicator"
    params = {
        'format': 'json',
        'per_page': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if len(data) < 2 or data[1] is None:
            return []
            
        # 过滤包含关键词的指标
        keyword_lower = keyword.lower()
        results = []
        
        for indicator in data[1]:
            name = indicator.get('name', '').lower()
            source_note = indicator.get('sourceNote', '').lower()
            
            if keyword_lower in name or keyword_lower in source_note:
                results.append({
                    'id': indicator.get('id'),
                    'name': indicator.get('name'),
                    'source': indicator.get('source', {}).get('value', ''),
                    'note': indicator.get('sourceNote', '')[:200]
                })
                
        return results[:limit]
        
    except Exception as e:
        logger.error(f"Error searching indicators: {e}")
        return []


def get_country_list() -> List[Dict]:
    """获取所有国家列表"""
    url = f"{BASE_URL}/country"
    params = {
        'format': 'json',
        'per_page': 500
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if len(data) < 2 or data[1] is None:
            return []
            
        return [
            {
                'id': c.get('id'),
                'name': c.get('name'),
                'region': c.get('region', {}).get('value', ''),
                'income_level': c.get('incomeLevel', {}).get('value', '')
            }
            for c in data[1]
            if c.get('region', {}).get('id') != 'NA'  # 排除聚合区域
        ]
        
    except Exception as e:
        logger.error(f"Error fetching country list: {e}")
        return []


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试获取数据
    df = fetch_worldbank_data(
        indicator='gdp_per_capita',
        countries=['USA', 'CHN', 'DEU'],
        start_year=2015,
        end_year=2023
    )
    print("GDP per capita data:")
    print(df.head(10))
    
    # 测试搜索指标
    indicators = search_indicators('renewable energy')
    print("\nRenewable energy indicators:")
    for ind in indicators[:5]:
        print(f"  {ind['id']}: {ind['name']}")
