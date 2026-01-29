---
name: citation-validator
description: 验证学术引用的真实性和准确性。通过DOI验证、作者信息核对、出版信息确认等方式，确保论文中的所有引用都是真实存在的，防止AI幻觉导致的虚假引用。
---

# 引用验证器 (Citation Validator)

## 功能概述

验证学术引用的真实性，确保论文中引用的所有文献都真实存在。这是防止AI幻觉的关键技能。

## 验证方法

### 1. DOI验证
```python
# 通过CrossRef验证DOI
def verify_doi(doi):
    response = requests.get(f"https://api.crossref.org/works/{doi}")
    return response.status_code == 200
```

### 2. 标题匹配验证
- 在Semantic Scholar搜索标题
- 检查返回结果是否匹配

### 3. 作者信息核对
- 验证作者姓名格式
- 检查作者机构信息

### 4. 出版信息确认
- 期刊/会议名称验证
- 出版年份合理性检查
- 卷号/页码格式检查

## 输出格式

```json
{
  "citations": [
    {
      "bibtex_key": "author2024title",
      "status": "verified",
      "verification_method": "doi",
      "doi_valid": true,
      "title_match": true,
      "confidence": 0.95
    },
    {
      "bibtex_key": "unknown2023paper",
      "status": "suspicious",
      "verification_method": "title_search",
      "doi_valid": false,
      "title_match": false,
      "confidence": 0.2,
      "warning": "无法验证此引用的真实性"
    }
  ],
  "summary": {
    "total": 20,
    "verified": 18,
    "suspicious": 2,
    "verification_rate": 0.9
  }
}
```

## 验证流程

1. **解析BibTeX** - 提取所有引用条目
2. **DOI优先验证** - 有DOI的直接验证
3. **标题搜索验证** - 无DOI的用标题搜索
4. **交叉验证** - 多源确认
5. **生成报告** - 标记可疑引用

## 可疑引用处理

对于无法验证的引用：
- 标记为可疑
- 尝试查找替代引用
- 生成警告报告

## 相关技能

- `literature-searcher` - 文献检索
- `hallucination-detector` - 幻觉检测
