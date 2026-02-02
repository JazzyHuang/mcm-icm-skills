---
name: data-collector
description: 自动收集美赛建模所需的数据。支持从World Bank、OECD、UN Data等公开数据源获取数据，并进行自动预处理（缺失值处理、异常值检测、特征工程）。
---

# 数据收集器 (Data Collector)

## 功能概述

自动从多个公开数据源收集美赛建模所需的数据，并进行预处理。

## 支持的数据源

### 1. World Bank Open Data
- **URL**: https://api.worldbank.org/v2/
- **数据类型**: 经济、社会、环境指标
- **无需API密钥**

### 2. OECD Data
- **URL**: https://sdmx.oecd.org/public/rest/
- **数据类型**: 经济、教育、健康统计
- **无需API密钥**

### 3. UN Data
- **URL**: https://data.un.org/
- **数据类型**: 全球发展指标
- **无需API密钥**

### 4. 题目附带数据
- 支持CSV、Excel、JSON格式
- 自动识别数据结构

## 数据预处理流程

### 1. 数据清洗
```python
# 缺失值处理
- 识别缺失模式
- 选择填充策略(均值/中位数/插值)
- 记录处理日志

# 异常值检测
- IQR方法: Q1-1.5*IQR ~ Q3+1.5*IQR
- Z-score方法: |z| > 3
- 可选择删除或替换
```

### 2. 数据转换
```python
# 标准化/归一化
- StandardScaler: 均值0，标准差1
- MinMaxScaler: 缩放到[0,1]

# 编码
- One-Hot编码: 分类变量
- Label编码: 有序变量
```

### 3. 特征工程
```python
# 特征选择
- 相关性分析
- 互信息
- 特征重要性

# 特征创建
- 时间特征提取
- 交互特征
- 聚合特征
```

## 输出格式

```json
{
  "data_sources": [
    {
      "name": "World Bank GDP",
      "url": "...",
      "rows": 1000,
      "columns": 10
    }
  ],
  "processed_data": {
    "file_path": "output/data/processed_data.csv",
    "shape": [1000, 15],
    "columns": ["col1", "col2", ...],
    "dtypes": {"col1": "float64", ...}
  },
  "preprocessing_log": {
    "missing_values_filled": 50,
    "outliers_removed": 10,
    "features_created": 5
  },
  "data_quality_score": 0.95,
  "citations": [
    {
      "bibtex_key": "worldbank_gdp_2024",
      "bibtex": "@online{...}",
      "category": "data"
    }
  ]
}
```

## 数据源引用生成（新增）

### 自动引用功能

每次从数据源获取数据时，自动生成对应的BibTeX引用：

```python
from scripts.data_citation_generator import generate_data_citation

# 获取数据时自动生成引用
data, citation = fetch_worldbank_data_with_citation(
    indicator="NY.GDP.PCAP.CD",
    countries=["USA", "CHN"],
    start_year=2020,
    end_year=2024
)
```

### 引用格式示例

#### World Bank数据
```bibtex
@online{worldbank_gdp_2024,
  author = {{World Bank}},
  title = {{World Bank Open Data: GDP per capita (current US\$)}},
  year = {2024},
  url = {https://data.worldbank.org/indicator/NY.GDP.PCAP.CD},
  urldate = {2024-01-30},
  note = {Countries: USA, CHN, DEU; Period: 2020-2024}
}
```

#### UN Data
```bibtex
@online{undata_population_2024,
  author = {{United Nations}},
  title = {{UN Data: Population Indicators}},
  year = {2024},
  url = {https://data.un.org/},
  urldate = {2024-01-30},
  note = {Global population statistics}
}
```

#### OECD Data
```bibtex
@online{oecd_education_2024,
  author = {{OECD}},
  title = {{OECD Data: Education Statistics}},
  year = {2024},
  url = {https://data.oecd.org/},
  urldate = {2024-01-30},
  note = {Education expenditure indicators}
}
```

### 引用收集

所有生成的数据源引用会被收集到：
- `output/data/data_citations.bib` - BibTeX文件
- `output/data/data_citations.json` - 结构化JSON

## 使用脚本

```python
from scripts.worldbank_api import fetch_worldbank_data
from scripts.data_cleaner import clean_data

# 获取数据
raw_data = fetch_worldbank_data(
    indicator="NY.GDP.PCAP.CD",
    countries=["USA", "CHN", "DEU"],
    start_year=2010,
    end_year=2023
)

# 清洗数据
cleaned_data = clean_data(
    raw_data,
    handle_missing='interpolate',
    handle_outliers='clip'
)
```

## 错误处理

- **数据源不可用**: 自动切换备选数据源
- **数据格式错误**: 尝试自动修复或标记警告
- **数据量过大**: 分块处理

## 相关技能

- `problem-parser` - 识别数据需求
- `literature-searcher` - 获取数据来源建议
- `citation-diversity-validator` - 验证数据源引用多样性
- `citation-manager` - 管理所有引用
