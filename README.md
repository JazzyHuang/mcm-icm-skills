# MCM/ICM 全自动化论文写作系统

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

美赛(MCM/ICM)论文全自动化写作系统，采用2025-2026年最新SOTA技术，支持从题目解析到生成O奖级别LaTeX论文的完整流程。

## 特性

- **全自动化流程**: 10个阶段、30+技能覆盖论文写作全流程
- **SOTA技术栈**: AgentOrchestra编排、OR-LLM-Agent建模、XtraGPT写作
- **O奖级别质量**: 12轮摘要迭代、Sobol敏感性分析、出版级图表
- **智能错误恢复**: 三级恢复机制、检查点恢复
- **幻觉零容忍**: HaluCheck检测、DOI引用验证
- **全题型支持**: MCM A/B/C + ICM D/E/F

## 快速开始

### 安装

```bash
# 克隆仓库
cd /path/to/your/workspace

# 安装依赖
pip install -r scripts/requirements.txt

# 配置API密钥
cp config/api_keys.yaml.template config/api_keys.yaml
# 编辑 config/api_keys.yaml 填入你的API密钥
```

### 使用

在Cursor或Claude Code中，当提供美赛题目时，系统会自动触发：

```
用户: 这是2026年美赛A题: [题目内容]
请帮我生成完整的论文
```

系统将自动执行完整流程并生成论文。

## 目录结构

```
mcm-icm-skills/
├── SKILL.md                 # 主入口技能
├── README.md                # 本文件
├── config/                  # 配置文件
├── orchestrator/            # 编排层
├── skills/                  # 所有技能
│   ├── phase1-input/        # 输入处理
│   ├── phase2-analysis/     # 问题分析
│   ├── phase3-modeling/     # 模型构建
│   ├── phase4-validation/   # 验证分析
│   ├── phase5-writing/      # 内容生成
│   ├── phase6-visualization/# 可视化
│   ├── phase7-integration/  # 文档整合
│   ├── phase8-quality/      # 质量保证
│   ├── phase9-optimization/ # 最终优化
│   └── phase10-submission/  # 提交验证
├── templates/               # LaTeX模板
├── knowledge_base/          # 知识库
├── scripts/                 # Python脚本
├── tests/                   # 测试文件
└── output/                  # 输出目录
```

## 核心技术

### 多智能体编排
- **AgentOrchestra**: 层级式任务分解 (GAIA基准83.39%)
- **Gradientsys**: ReAct动态规划
- **MARCO**: 94.48%任务执行准确率

### 数学建模
- **OR-LLM-Agent**: 自然语言到数学模型 (100%通过率)
- **AgentMath**: AIME24 90.6%准确率
- **SymCode**: 可验证代码生成

### 学术写作
- **XtraGPT**: 上下文感知论文修订
- **ScholarCopilot**: 40.1% top-1引用准确率
- **AJE Grammar Check**: 学术英语LAT评分

### 质量保证
- **HaluCheck**: 幻觉检测
- **VerifiAgent**: 代码验证
- **SALib**: 敏感性分析

## 配置说明

### settings.yaml

主要配置项：

```yaml
execution:
  max_retries: 3              # 最大重试次数
  timeout_per_skill: 300      # 单技能超时(秒)

quality_thresholds:
  abstract_min_iterations: 12 # 摘要最少迭代次数
  grammar_score_min: 7.5      # LAT评分最低要求
  hallucination_tolerance: 0  # 幻觉零容忍
```

### API密钥

必需的API密钥：
- Semantic Scholar API Key (免费申请)

可选的API密钥：
- Gurobi License (学术免费)
- OpenAI/Anthropic API

## 题型支持

| 题型 | 特点 | 核心方法 | 适合背景 |
|------|------|---------|---------|
| A-连续 | 微分方程、物理建模 | ODE/PDE、有限元 | 理工科 |
| B-离散 | 算法、组合优化 | 图论、DP | CS |
| C-数据 | 数据挖掘、预测 | ML、时间序列 | 数据科学 |
| D-运筹 | 网络优化、调度 | MILP、网络流 | 运筹学 |
| E-可持续 | 环境、生态 | 多目标、系统动力学 | 跨学科 |
| F-政策 | 社会科学、决策 | 博弈论、仿真 | 社科 |

## O奖标准

系统针对以下评分标准优化：

| 维度 | 权重 | 实现方式 |
|------|------|---------|
| 模型合理性 | 30% | 多模型对比、假设充分论证 |
| 算法创新性 | 25% | 混合模型、跨领域迁移 |
| 可视化强度 | 20% | 出版级图表、色盲友好 |
| 敏感性分析 | 15% | Sobol+Morris全局分析 |
| 伦理审查 | 10% | 公平性、社会影响评估 |

## 测试

```bash
# 运行单元测试
pytest tests/unit/

# 运行集成测试
pytest tests/integration/

# 使用历史题目测试
python scripts/test_runner.py --problem 2024A
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request!

## 致谢

- COMAP MCM/ICM 竞赛组织方
- Semantic Scholar 提供文献API
- 所有开源项目贡献者
