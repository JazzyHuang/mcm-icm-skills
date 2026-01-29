---
name: hallucination-detector
description: 检测AI生成内容中的幻觉。基于HaluCheck技术逐句验证，检测虚假引用、编造数据、不存在的方法、错误的数学推导。零容忍策略。
---

# 幻觉检测器 (Hallucination Detector)

## 功能概述

检测论文中可能由AI生成的幻觉内容，确保内容真实可靠。

## 检测范围

### 1. 虚假引用
- 不存在的论文
- 错误的作者信息
- 虚假的DOI

### 2. 编造数据
- 不合理的数值
- 捏造的统计结果
- 虚假的实验数据

### 3. 不存在的方法
- 虚构的算法名称
- 不存在的工具/库
- 编造的技术术语

### 4. 错误推导
- 数学错误
- 逻辑谬误
- 因果倒置

## 检测方法

### HaluCheck流程
```python
def detect_hallucination(text):
    # 1. 分句
    sentences = split_sentences(text)
    
    # 2. 逐句验证
    results = []
    for sentence in sentences:
        # 提取声明
        claims = extract_claims(sentence)
        
        # 验证每个声明
        for claim in claims:
            verified = verify_claim(claim)
            results.append({
                'sentence': sentence,
                'claim': claim,
                'verified': verified,
                'confidence': verified['confidence']
            })
    
    return results
```

## 输出格式

```json
{
  "hallucination_check": {
    "status": "passed",
    "total_claims": 150,
    "verified": 148,
    "suspicious": 2,
    "details": [
      {
        "location": "Section 2.3, Line 45",
        "content": "According to Smith et al. (2023)...",
        "issue": "Citation not found in databases",
        "severity": "high",
        "action": "Verify or remove citation"
      }
    ],
    "confidence": 0.98
  }
}
```

## 零容忍策略

所有检测到的幻觉必须：
1. 标记并报告
2. 验证或删除
3. 不允许通过审核

## 相关技能

- `citation-validator` - 引用验证
- `fact-checker` - 事实核查
