---
name: mcm-icm-automation
description: 美赛论文全自动化写作系统(2025-2026 SOTA版)。当用户上传美赛题目时自动触发，采用层级多智能体架构，完成从题目解析到O奖级别论文生成的完整流程。支持所有题型(A-F)，内置错误恢复、幻觉检测、和质量保证机制。使用时请提供美赛题目PDF或文本内容。
---

# MCM/ICM 全自动化论文写作系统

## 快速开始

当用户提供美赛题目时，系统自动执行完整流程：

1. **输入**: 题目PDF/文本 + 附加数据文件(可选)
2. **输出**: 完整的O奖级别LaTeX论文 + PDF

## 使用方式

```
用户: 这是2026年美赛A题，请帮我生成论文
[附加题目PDF或文本内容]
```

系统将自动：
1. 解析题目，识别问题类型
2. 收集数据和文献
3. 构建数学模型
4. 求解并验证
5. 生成完整论文

## 执行流程

### 阶段0: 初始化
- 解析用户输入
- 初始化状态管理器
- 创建工作目录

### 阶段1: 输入处理 (并行执行)
调用技能:
- `skills/phase1-input/problem-parser/` → 解析题目
- `skills/phase1-input/problem-type-classifier/` → 识别题型
- `skills/phase1-input/data-collector/` → 收集数据 (如需)
- `skills/phase1-input/literature-searcher/` → 检索文献
- `skills/phase1-input/citation-validator/` → 验证引用

### 阶段2: 问题分析 (顺序执行)
调用技能:
- `skills/phase2-analysis/problem-decomposer/` → 问题分解
- `skills/phase2-analysis/sub-problem-analyzer/` → 子问题分析
- `skills/phase2-analysis/assumption-generator/` → 生成假设
- `skills/phase2-analysis/variable-definer/` → 定义变量
- `skills/phase2-analysis/constraint-identifier/` → 识别约束

### 阶段3: 模型构建 (核心阶段)
调用技能:
- `skills/phase3-modeling/model-selector/` → 选择模型
- `skills/phase3-modeling/hybrid-model-designer/` → 设计混合模型
- `skills/phase3-modeling/model-builder/` → 构建模型
- `skills/phase3-modeling/model-solver/` → 求解模型
- `skills/phase3-modeling/code-verifier/` → 验证代码

**失败时**: 错误恢复 → 重试/降级方案

### 阶段4: 验证分析
调用技能:
- `skills/phase4-validation/sensitivity-analyzer/` → 敏感性分析
- `skills/phase4-validation/uncertainty-quantifier/` → 不确定性量化
- `skills/phase4-validation/model-validator/` → 模型验证
- `skills/phase4-validation/strengths-weaknesses/` → 优缺点分析
- `skills/phase4-validation/ethical-analyzer/` → 伦理分析 (ICM E/F)

### 阶段5: 内容生成
调用技能:
- `skills/phase5-writing/section-writer/` → 章节写作
- `skills/phase5-writing/fact-checker/` → 事实核查
- `skills/phase5-writing/abstract-generator/` → 生成摘要
- `skills/phase5-writing/abstract-iterative-optimizer/` → 迭代优化摘要 (12轮+)
- `skills/phase5-writing/memo-letter-writer/` → 备忘录

### 阶段6: 可视化
调用技能:
- `skills/phase6-visualization/chart-generator/` → 生成图表
- `skills/phase6-visualization/publication-scaler/` → 出版级缩放
- `skills/phase6-visualization/table-formatter/` → 格式化表格
- `skills/phase6-visualization/figure-validator/` → 验证图表

### 阶段7: 文档整合
调用技能:
- `skills/phase7-integration/latex-compiler/` → LaTeX编译
- `skills/phase7-integration/compilation-error-handler/` → 错误处理
- `skills/phase7-integration/citation-manager/` → 引用管理
- `skills/phase7-integration/format-checker/` → 格式检查
- `skills/phase7-integration/anonymization-checker/` → 匿名化检查

### 阶段8: 质量保证
调用技能:
- `skills/phase8-quality/quality-reviewer/` → 质量审查
- `skills/phase8-quality/hallucination-detector/` → 幻觉检测
- `skills/phase8-quality/grammar-checker/` → 语法检查
- `skills/phase8-quality/consistency-checker/` → 一致性检查

### 阶段9: 最终优化
调用技能:
- `skills/phase9-optimization/final-polisher/` → 最终润色
- `skills/phase9-optimization/academic-english-optimizer/` → 学术英语优化
- `skills/phase9-optimization/submission-preparer/` → 准备提交

### 阶段10: 提交验证
调用技能:
- `skills/phase10-submission/pre-submission-validator/` → 提交前验证
- `skills/phase10-submission/submission-checklist/` → 检查清单

## 错误恢复机制

系统内置三级错误恢复:

1. **自动重试**: 网络错误、API限流 (最多3次，指数退避)
2. **备选方案**: 求解器切换、数据源切换、简化模型
3. **人工干预**: 严重错误时暂停并请求指导

## 检查点恢复

每个阶段完成后自动保存检查点到 `output/checkpoints/`，支持从任意阶段恢复执行。

恢复命令: 指定 `--resume-from=phase3` 从阶段3恢复

## 配置文件

- `config/settings.yaml` - 全局配置
- `config/api_keys.yaml` - API密钥 (需从template复制并填写)
- `config/model_config.yaml` - 模型参数配置

## 输出目录

- `output/papers/` - 生成的论文
- `output/figures/` - 图表文件
- `output/data/` - 处理后的数据
- `output/logs/` - 运行日志

## 支持的题型

| 题型 | 名称 | 核心方法 |
|------|------|---------|
| A | 连续问题 | ODE/PDE、有限元 |
| B | 离散问题 | 图论、动态规划 |
| C | 数据问题 | 机器学习、时间序列 |
| D | 运筹学 | 整数规划、网络流 |
| E | 可持续性 | 多目标优化、系统动力学 |
| F | 政策 | 博弈论、仿真 |

## O奖质量保证

系统针对O奖标准进行了优化：

- **摘要**: 12轮以上迭代优化
- **模型**: 创新性混合模型设计
- **验证**: Sobol全局敏感性分析
- **可视化**: 出版级图表质量
- **语言**: LAT评分≥7.5
- **引用**: DOI验证确保真实性
- **幻觉**: 零容忍检测策略

## 详细文档

参见各子技能的 SKILL.md 文件获取详细说明。
