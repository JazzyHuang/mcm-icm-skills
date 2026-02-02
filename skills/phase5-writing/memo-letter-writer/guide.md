---
name: memo-letter-writer
description: 生成专业的备忘录或信函。将技术内容转化为非技术语言，面向决策者和政策制定者，遵循商务信函格式。
---

# 备忘录/信函写作器 (Memo/Letter Writer)

## 功能概述

生成面向决策者的专业备忘录或信函，将复杂的技术分析转化为易于理解的建议。

## 格式要求

### 备忘录格式 (Memo)
```
MEMORANDUM

TO:      [收件人/组织]
FROM:    Team #XXXXX
DATE:    [日期]
RE:      [主题]

---

[Executive Summary - 1段]

[Background - 1段]

[Key Findings - 2-3段，可用bullet points]

[Recommendations - 具体可行的建议]

[Conclusion - 1段]
```

### 信函格式 (Letter)
```
[日期]

[收件人姓名]
[职位]
[组织名称]
[地址]

Dear [称呼]:

[开头段 - 说明目的]

[主体段落 - 关键发现和分析]

[建议段落 - 具体可行的建议]

[结尾段 - 总结和展望]

Sincerely,
Team #XXXXX
```

## 写作原则

### 1. 语言转换
- 技术术语 → 通俗解释
- 数学公式 → 直观描述
- 数据结果 → 实际意义

### 2. 重点突出
- 关键数字要突出
- 建议要具体可行
- 影响要量化说明

### 3. 行动导向
- 提供明确建议
- 说明实施步骤
- 预期效果量化

## 示例

```
MEMORANDUM

TO:      Solar Energy Solutions Inc., Board of Directors
FROM:    Team #2412345
DATE:    February 1, 2026
RE:      Optimization Strategy for Solar Panel Deployment

---

EXECUTIVE SUMMARY

Our analysis reveals that implementing the proposed optimization 
strategy can increase energy harvest efficiency by 23% while 
reducing installation costs by 15%. We recommend immediate 
adoption of the latitude-adjusted angle configuration for all 
new installations.

KEY FINDINGS

• Panel angle optimization alone can improve efficiency by 12%
• Geographic clustering of installations reduces maintenance 
  costs by $2.3M annually
• The break-even period for new optimized installations is 
  4.2 years (vs. 5.8 years for standard configuration)

RECOMMENDATIONS

1. Implement dynamic angle adjustment for panels in latitudes 
   30°-45° (highest impact zones)
2. Prioritize installation in the top 10 efficiency-rated 
   locations identified in our analysis
3. Invest in predictive maintenance systems to reduce 
   operational costs

CONCLUSION

The proposed optimization framework provides a clear path to 
improved performance and profitability. We recommend scheduling 
a technical briefing to discuss implementation details.
```

## 输出格式

```json
{
  "memo": {
    "type": "memorandum",
    "recipient": "Solar Energy Solutions Inc.",
    "subject": "Optimization Strategy",
    "sections": {
      "executive_summary": "...",
      "key_findings": ["...", "..."],
      "recommendations": ["...", "..."],
      "conclusion": "..."
    },
    "word_count": 450,
    "technical_terms_simplified": 8
  }
}
```

## 相关技能

- `section-writer` - 章节写作
- `ethical-analyzer` - 伦理分析
