---
name: infographic-generator
description: 科学信息图自动生成器，遵循Nature 2025设计原则。自动创建论文摘要图、关键发现可视化、方法流程图。支持视觉层次、配色策略、排版规范。
---

# 科学信息图生成器 (Scientific Infographic Generator)

## 功能概述

自动生成专业科学信息图：
1. **论文摘要图** - 一图概括研究亮点
2. **方法流程图** - 可视化研究方法
3. **关键发现图** - 突出主要结论
4. **比较分析图** - 多方案对比

## 设计原则 (Nature 2025)

### 视觉层次
```
标题 (最大, 14-18pt)
    ↓
核心图表 (占50%以上面积)
    ↓
关键数据 (统计数据、百分比)
    ↓
方法说明 (最小, 8-10pt)
```

### 配色策略
| 用途 | 推荐颜色 | 含义 |
|-----|---------|------|
| 科学/信任 | 蓝色系 | 数据、方法 |
| 生物/健康 | 绿色系 | 环境、可持续 |
| 警告/重要 | 红色系 | 关键发现 |
| 中性 | 灰色系 | 背景、次要 |

### 排版规范
- 标题比正文大2-3倍
- 最小字体14pt（确保可读性）
- 留白充足（不少于10%边距）
- 图文比例70:30

## 使用方法

### 摘要图生成

```python
from infographic_generator import InfographicGenerator

generator = InfographicGenerator(style='nature')

# 生成摘要信息图
fig = generator.create_summary_infographic(
    title="研究标题",
    key_findings=[
        {"text": "发现1", "value": "85%", "icon": "up"},
        {"text": "发现2", "value": "1000x", "icon": "speed"},
        {"text": "发现3", "value": "$2.5M", "icon": "money"}
    ],
    main_figure=result_plot,
    methods=["数据收集", "模型构建", "验证分析"],
    conclusion="主要结论摘要"
)

fig.savefig('summary_infographic.pdf', dpi=300)
```

### 方法流程图

```python
# 生成方法流程图
fig = generator.create_methods_flow(
    steps=[
        {"name": "数据预处理", "description": "清洗、标准化"},
        {"name": "特征工程", "description": "PCA、选择"},
        {"name": "模型训练", "description": "PINN + Transformer"},
        {"name": "验证分析", "description": "交叉验证、敏感性"}
    ],
    connections=[(0, 1), (1, 2), (2, 3)]
)
```

### 比较分析图

```python
# 生成方案对比图
fig = generator.create_comparison(
    scenarios=["基准方案", "优化方案A", "优化方案B"],
    metrics={
        "成本": [100, 85, 78],
        "效率": [70, 88, 92],
        "风险": [50, 35, 40]
    },
    highlight="优化方案B"
)
```

## 输出格式

```json
{
  "infographic": {
    "type": "summary",
    "style": "nature",
    "dimensions": {"width": 1200, "height": 800}
  },
  "elements": {
    "title": {"text": "...", "font_size": 18},
    "key_findings": 3,
    "figures_included": 2,
    "methods_steps": 4
  },
  "output_files": [
    "summary_infographic.pdf",
    "summary_infographic.png"
  ]
}
```

## O奖加分建议

- 在论文开头放置摘要信息图（"Graphical Abstract"）
- 保持视觉一致性（颜色、字体）
- 突出创新点和关键数据
- 确保即使打印为黑白也能理解

## 相关技能

- `chart-generator` - 基础图表
- `nature-style` - Nature样式规范
- `figure-validator` - 图表验证
