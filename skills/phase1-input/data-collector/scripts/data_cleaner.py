"""
数据清洗器
处理缺失值、异常值，进行数据标准化
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(
    df: pd.DataFrame,
    handle_missing: str = 'interpolate',
    handle_outliers: str = 'clip',
    outlier_method: str = 'iqr',
    outlier_threshold: float = 1.5,
    drop_threshold: float = 0.5
) -> Tuple[pd.DataFrame, Dict]:
    """
    清洗数据
    
    Args:
        df: 输入DataFrame
        handle_missing: 缺失值处理方式 ('drop', 'mean', 'median', 'interpolate', 'ffill')
        handle_outliers: 异常值处理方式 ('drop', 'clip', 'none')
        outlier_method: 异常值检测方法 ('iqr', 'zscore')
        outlier_threshold: 异常值阈值 (IQR倍数或Z-score)
        drop_threshold: 缺失比例超过此值的列将被删除
        
    Returns:
        清洗后的DataFrame和处理日志
    """
    log = {
        'original_shape': df.shape,
        'missing_before': df.isnull().sum().to_dict(),
        'columns_dropped': [],
        'missing_filled': 0,
        'outliers_handled': 0
    }
    
    df = df.copy()
    
    # 1. 删除缺失比例过高的列
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        log['columns_dropped'] = cols_to_drop
        logger.info(f"Dropped columns with >{drop_threshold*100}% missing: {cols_to_drop}")
        
    # 2. 处理缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if handle_missing == 'drop':
        original_len = len(df)
        df = df.dropna()
        log['rows_dropped'] = original_len - len(df)
        
    elif handle_missing == 'mean':
        for col in numeric_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(df[col].mean())
                log['missing_filled'] += missing_count
                
    elif handle_missing == 'median':
        for col in numeric_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(df[col].median())
                log['missing_filled'] += missing_count
                
    elif handle_missing == 'interpolate':
        for col in numeric_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                log['missing_filled'] += missing_count
                
    elif handle_missing == 'ffill':
        for col in numeric_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                log['missing_filled'] += missing_count
                
    # 3. 处理异常值
    if handle_outliers != 'none':
        for col in numeric_cols:
            if outlier_method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - outlier_threshold * IQR
                upper = Q3 + outlier_threshold * IQR
            else:  # zscore
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - outlier_threshold * std
                upper = mean + outlier_threshold * std
                
            outliers = (df[col] < lower) | (df[col] > upper)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                if handle_outliers == 'drop':
                    df = df[~outliers]
                elif handle_outliers == 'clip':
                    df[col] = df[col].clip(lower, upper)
                    
                log['outliers_handled'] += outlier_count
                
    log['final_shape'] = df.shape
    log['missing_after'] = df.isnull().sum().to_dict()
    
    logger.info(f"Data cleaned: {log['original_shape']} -> {log['final_shape']}")
    
    return df, log


def standardize_data(
    df: pd.DataFrame,
    method: str = 'standard',
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    标准化数据
    
    Args:
        df: 输入DataFrame
        method: 标准化方法 ('standard', 'minmax', 'robust')
        columns: 要标准化的列，默认所有数值列
        
    Returns:
        标准化后的DataFrame和缩放参数
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    params = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'standard':
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std if std > 0 else 0
            params[col] = {'method': 'standard', 'mean': mean, 'std': std}
            
        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            df[col] = (df[col] - min_val) / range_val if range_val > 0 else 0
            params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
            
        elif method == 'robust':
            median = df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = (df[col] - median) / iqr if iqr > 0 else 0
            params[col] = {'method': 'robust', 'median': median, 'iqr': iqr}
            
    return df, params


def detect_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 1.5,
    columns: Optional[List[str]] = None
) -> Dict[str, pd.Series]:
    """
    检测异常值
    
    Args:
        df: 输入DataFrame
        method: 检测方法 ('iqr', 'zscore')
        threshold: 阈值
        columns: 要检测的列
        
    Returns:
        每列的异常值掩码
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    outliers = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df[col] < Q1 - threshold * IQR) | (df[col] > Q3 + threshold * IQR)
        else:  # zscore
            z = (df[col] - df[col].mean()) / df[col].std()
            mask = np.abs(z) > threshold
            
        outliers[col] = mask
        
    return outliers


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    获取数据摘要
    
    Args:
        df: 输入DataFrame
        
    Returns:
        数据摘要字典
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'missing_percent': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }
    
    # 数值列统计
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
    # 分类列统计
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        summary['categorical_stats'] = {
            col: {
                'unique': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
            for col in cat_cols
        }
        
    return summary


def create_features(
    df: pd.DataFrame,
    date_column: Optional[str] = None,
    interaction_pairs: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    创建特征
    
    Args:
        df: 输入DataFrame
        date_column: 日期列名，用于提取时间特征
        interaction_pairs: 交互特征对
        
    Returns:
        添加特征后的DataFrame
    """
    df = df.copy()
    
    # 时间特征
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        if df[date_column].notna().any():
            df[f'{date_column}_year'] = df[date_column].dt.year
            df[f'{date_column}_month'] = df[date_column].dt.month
            df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
            df[f'{date_column}_quarter'] = df[date_column].dt.quarter
            
    # 交互特征
    if interaction_pairs:
        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                # 乘积
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                # 比率
                with np.errstate(divide='ignore', invalid='ignore'):
                    df[f'{col1}_div_{col2}'] = df[col1] / df[col2]
                    df[f'{col1}_div_{col2}'] = df[f'{col1}_div_{col2}'].replace([np.inf, -np.inf], np.nan)
                    
    return df


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100) * 10 + 50,
        'C': np.random.choice(['X', 'Y', 'Z'], 100)
    })
    
    # 添加一些缺失值和异常值
    df.loc[5:10, 'A'] = np.nan
    df.loc[95:99, 'B'] = 1000  # 异常值
    
    print("Original data:")
    print(get_data_summary(df))
    
    # 清洗数据
    cleaned_df, log = clean_data(df, handle_missing='mean', handle_outliers='clip')
    print("\nCleaned data:")
    print(get_data_summary(cleaned_df))
    print("\nCleaning log:")
    print(log)
