---
name: publication-scaler
description: 自动缩放图表以适配论文布局。实现单栏/双栏自动适配、子图一致性调整、文字比例自动缩放。
---

# 出版级缩放器 (Publication Scaler)

## 功能概述

自动调整图表尺寸和比例，使其符合论文版面要求。

## 缩放策略

### 单栏布局
- 宽度: 3.5 inches (88mm)
- 适用: 单个图表

### 双栏布局
- 宽度: 7 inches (178mm)
- 适用: 大型图表、子图组

### 子图一致性
- 统一字体大小
- 统一线条粗细
- 统一图例样式

## 自动调整

```python
def scale_for_publication(fig, width='single'):
    """
    调整图表尺寸
    
    Args:
        fig: matplotlib Figure
        width: 'single' (3.5") or 'double' (7")
    """
    target_width = 3.5 if width == 'single' else 7.0
    current_width = fig.get_figwidth()
    scale = target_width / current_width
    
    fig.set_size_inches(target_width, fig.get_figheight() * scale)
    
    # 调整字体大小
    for ax in fig.axes:
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(10 * scale)
```

## 相关技能

- `chart-generator` - 图表生成
- `figure-validator` - 图表验证
