---
name: mcm-icm-automation
description: MCM/ICM美赛数学建模论文全自动化写作系统(2025-2026 SOTA版)。当用户提到美赛、MCM、ICM、数学建模、COMAP、Mathematical Contest in Modeling时自动触发。采用层级多智能体架构，完成从题目解析到O奖级别论文生成的完整流程。支持所有题型(A连续/B离散/C数据/D运筹/E可持续/F政策)，内置错误恢复、幻觉检测、质量门禁和阶段回退机制。使用时请提供美赛题目PDF或文本内容。
---

# MCM/ICM 全自动化论文写作系统 (O奖优化版)

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

**阶段1a: 题目解析与数据收集**
调用技能:
- `skills/phase1-input/problem-parser/` → 解析题目
- `skills/phase1-input/problem-type-classifier/` → 识别题型
- `skills/phase1-input/problem-reference-extractor/` → 提取题目引用
- `skills/phase1-input/data-collector/` → 收集数据并生成数据源引用

**阶段1b: 深度文献搜索**
调用技能:
- `skills/phase1-input/deep-reference-searcher/` → 多源深度文献搜索
  - Semantic Scholar + OpenAlex + arXiv + CrossRef + Google Scholar
  - 政府报告搜索 (World Bank, UN, OECD)
  - 数据源引用生成
- `skills/phase1-input/ai-deep-search-guide/` → AI深度搜索引导
- `skills/phase1-input/literature-searcher/` → 基础文献检索

**阶段1c: 引用验证与多样性检查**
调用技能:
- `skills/phase1-input/citation-validator/` → 验证引用真实性
- `skills/phase1-input/citation-diversity-validator/` → 验证引用多样性
  - 确保覆盖4+类别 (学术/政府/数据/题目/其他)
  - 多样性评分≥0.75
  - 总引用数8-15篇

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
- `skills/phase3-modeling/model-justification-generator/` → **模型选择论证** (新增)
  - 自动生成模型对比分析
  - 与至少2种备选方法对比
  - 量化说明选择依据
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
- `skills/phase4-validation/error-analyzer/` → **误差分析** (新增)
  - 多种误差指标计算
  - 与基准方法对比
  - 误差分布可视化
- `skills/phase4-validation/limitation-analyzer/` → **局限性分析** (新增)
  - 自动识别模型局限
  - 生成诚实的自我批评段落
  - 提出改进方向
- `skills/phase4-validation/strengths-weaknesses/` → 优缺点分析
- `skills/phase4-validation/ethical-analyzer/` → 伦理分析 (ICM E/F)

### 阶段5: 内容生成
调用技能:
- `skills/phase5-writing/section-writer/` → 章节写作
- `skills/phase5-writing/fact-checker/` → 事实核查
- `skills/phase5-writing/abstract-first-impression/` → **摘要第一印象** (新增)
  - 生成震撼开头(Hook句)
  - 三种Hook模式：问题重要性/数据冲击/方法创新
  - Hook质量评分
- `skills/phase5-writing/abstract-generator/` → 生成摘要
- `skills/phase5-writing/abstract-iterative-optimizer/` → 迭代优化摘要 (增强版)
  - 新增Hook质量评分维度
  - 新增量化密度评分维度
  - O奖基准对比
  - 12轮以上迭代
- `skills/phase5-writing/memo-letter-writer/` → 备忘录

### 阶段6: 可视化
调用技能:
- `skills/phase6-visualization/chart-generator/` → 生成图表 (增强版)
  - 新增交互效应热力图
  - 新增参数重要性排序图
  - 新增不确定性区间图
  - 敏感性分析综合仪表板
- `skills/phase6-visualization/figure-narrative-generator/` → **图表叙事** (新增)
  - 为每个图表生成Caption
  - 生成正文引用句
  - 深度解释（Level 3-4）
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
- `skills/phase8-quality/chinglish-detector/` → **中式英语检测** (新增)
  - 50+常见中式英语模式
  - 自动修正建议
  - Chinglish评分 ≤ 0.20
- `skills/phase8-quality/consistency-checker/` → 一致性检查
- `skills/phase8-quality/global-consistency-checker/` → **全局一致性** (新增)
  - 跨章节术语一致性
  - 数据引用一致性
  - 符号使用一致性
  - 图表编号一致性

### 阶段9: 最终优化
调用技能:
- `skills/phase9-optimization/final-polisher/` → 最终润色
- `skills/phase9-optimization/academic-english-optimizer/` → 学术英语优化
- `skills/phase9-optimization/submission-preparer/` → 准备提交

### 阶段10: 提交验证
调用技能:
- `skills/phase10-submission/pre-submission-validator/` → 提交前验证
- `skills/phase10-submission/submission-checklist/` → 检查清单

## 质量门禁机制 (新增)

系统在每个阶段后执行质量门禁检查:

```yaml
关键门禁:
  - 摘要评分 ≥ 0.85
  - Hook质量 ≥ 0.80
  - 量化密度 ≥ 0.75
  - LAT评分 ≥ 7.5
  - 中式英语评分 ≤ 0.20
  - 一致性评分 ≥ 0.90
  - 幻觉数量 = 0
```

门禁失败时自动触发阶段回退，最多重试2次。

配置文件: `config/quality_gates.yaml`

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
- `config/citation_sources.yaml` - 引用数据源配置
- `config/quality_gates.yaml` - **质量门禁配置** (新增)

## 知识库 (增强版)

- `knowledge_base/o_award_features.json` - O奖特征定义 (增强版)
  - 常见错误模式库
  - 降级风险因素清单
  - 质量阈值定义
- `knowledge_base/writing_guidelines.md` - 写作指南 (增强版)
  - 中式英语错误清单
  - 数学符号表达规范
  - Hook句设计指南
  - 量化表达指南
- `knowledge_base/academic_expressions.json` - **学术表达库** (新增)
  - 200+学术表达
  - 12个场景分类
  - 每个表达配例句

## 输出目录

- `output/papers/` - 生成的论文
- `output/figures/` - 图表文件
- `output/data/` - 处理后的数据
- `output/logs/` - 运行日志
- `output/quality_reports/` - **质量报告** (新增)

## 支持的题型

| 题型 | 名称 | 核心方法 |
|------|------|---------|
| A | 连续问题 | ODE/PDE、有限元、PINN |
| B | 离散问题 | 图论、动态规划、GNN |
| C | 数据问题 | 机器学习、时间序列、Transformer |
| D | 运筹学 | 整数规划、网络流、元启发式 |
| E | 可持续性 | 多目标优化、系统动力学、因果推断 |
| F | 政策 | 博弈论、仿真、多智能体 |

## O奖质量保证

系统针对O奖标准进行了全面优化：

### 摘要优化
- **Hook质量**: 三种模式自动生成，评分≥0.80
- **量化密度**: 4-8个具体数字，评分≥0.75
- **迭代优化**: 12轮以上，总分≥0.90

### 模型呈现
- **选择论证**: 自动与2+备选方法对比
- **创新声明**: 自动生成"Our Contribution"段落
- **混合设计**: 创新性评分≥0.70

### 验证分析
- **敏感性**: Sobol全局分析 + 交互效应热力图
- **误差分析**: 3-5种指标 + 基准对比 + 可视化
- **局限性**: 诚实的自我批评段落

### 语言质量
- **LAT评分**: ≥7.5
- **中式英语**: 50+模式检测，评分≤0.20
- **一致性**: 术语/数据/符号全文一致，评分≥0.90

### 可视化
- **数量**: 5-15个高质量图表
- **叙事**: 每个图表配Caption+引用+解释
- **格式**: Nature 2025出版规范

### 引用多样性
- **总数**: 8-15篇
- **类别**: 覆盖4+类别
- **多样性评分**: ≥0.75
- **验证**: DOI验证确保真实性

### 质量保证
- **幻觉检测**: 零容忍策略
- **质量门禁**: 每阶段检查
- **回退机制**: 失败自动重试

## 新增技能清单

| 技能 | 位置 | 功能 |
|------|------|------|
| abstract-first-impression | phase5-writing | 摘要Hook句生成 |
| model-justification-generator | phase3-modeling | 模型选择论证 |
| error-analyzer | phase4-validation | 误差分析生成 |
| limitation-analyzer | phase4-validation | 局限性分析 |
| figure-narrative-generator | phase6-visualization | 图表叙事 |
| chinglish-detector | phase8-quality | 中式英语检测 |
| global-consistency-checker | phase8-quality | 全局一致性 |

## 详细文档

参见各子技能的 guide.md 文件获取详细说明。
