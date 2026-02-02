---
name: chart-generator
description: 生成出版级别的专业图表。支持折线图、柱状图、热力图、网络图等多种类型，遵循学术出版标准，使用色盲友好配色。
---

# 图表生成器 (Chart Generator)

## 功能概述

生成符合学术出版标准的专业图表。

## 图表类型矩阵

| 数据类型 | 推荐图表 | 备选方案 |
|---------|---------|---------|
| 时间趋势 | 带置信区间的折线图 | 面积图、阶梯图 |
| 分布比较 | 箱线图、小提琴图 | 直方图、KDE |
| 相关关系 | 散点图、热力图 | 气泡图、平行坐标 |
| 组成占比 | 堆叠柱状图 | 饼图(谨慎使用) |
| 网络结构 | 力导向图、邻接矩阵 | 圆形布局、层级 |
| 地理数据 | Choropleth地图 | 散点地图 |
| 3D关系 | 曲面图、等高线 | 热力切片 |

## 出版规范

### 分辨率
- 打印: 300+ DPI
- 屏幕: 150 DPI

### 格式
- 矢量图: PDF (首选)
- 位图: PNG (备选)

### 配色
- 使用色盲友好调色板
- 推荐: viridis, plasma, cividis
- 避免: 红绿对比

### 字体
- 大小: 8-14pt
- 类型: sans-serif (Arial, Helvetica)

### 图例
- 位置: 图内或图外统一
- 标签: 清晰、简洁

## 代码示例

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 设置学术风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# 创建图表
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, color='#1f77b4', linewidth=1.5, label='Model')
ax.fill_between(x, y_lower, y_upper, alpha=0.2)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Value (units)')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 保存
fig.savefig('figure.pdf', bbox_inches='tight', dpi=300)
```

## 输出格式

```json
{
  "figure": {
    "type": "line_chart",
    "file": "figures/trend_analysis.pdf",
    "dimensions": {"width": 6, "height": 4, "dpi": 300},
    "caption": "Figure 1: Temporal trend of...",
    "label": "fig:trend"
  }
}
```

## 相关技能

- `publication-scaler` - 出版级缩放
- `table-formatter` - 表格格式化
