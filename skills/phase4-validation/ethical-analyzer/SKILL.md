---
name: ethical-analyzer
description: 分析模型的伦理和社会影响。特别针对ICM E/F题，评估公平性、环境影响、社会效益等维度。
---

# 伦理分析器 (Ethical Analyzer)

## 功能概述

评估模型的伦理影响和社会后果，特别适用于ICM E/F题型。

## 评估维度

### 1. 公平性分析
- 利益相关者识别
- 分配公平性
- 程序公平性

### 2. 环境影响评估
- 资源消耗
- 污染排放
- 生态影响

### 3. 社会效益/成本
- 直接影响
- 间接影响
- 长期效果

### 4. 政策可行性
- 实施成本
- 政治接受度
- 技术可行性

### 5. 利益相关者影响
- 受益群体
- 受损群体
- 补偿机制

## 输出格式

```json
{
  "ethical_analysis": {
    "fairness": {
      "score": 0.75,
      "concerns": ["某些群体可能受益较少"],
      "mitigations": ["建议增加补偿机制"]
    },
    "environmental_impact": {
      "score": 0.85,
      "positive": ["减少碳排放"],
      "negative": ["初期建设影响"]
    },
    "social_impact": {
      "benefits": ["创造就业", "提高效率"],
      "costs": ["短期调整成本"],
      "net_assessment": "positive"
    },
    "recommendations": [
      "建议分阶段实施以减少社会冲击",
      "设立专项基金补偿受影响群体"
    ]
  }
}
```

## 相关技能

- `strengths-weaknesses` - 优缺点分析
- `memo-letter-writer` - 备忘录撰写
