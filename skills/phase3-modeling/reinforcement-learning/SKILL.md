---
name: reinforcement-learning
description: 强化学习模块，支持DQN和PPO算法实现动态决策优化。适用于B题离散决策、D题调度优化、F题多智能体博弈。创新性评分0.88。
---

# 强化学习模块 (Reinforcement Learning)

## 功能概述

提供强化学习算法用于动态决策问题：
1. DQN - 离散动作空间
2. PPO - 连续动作空间  
3. Multi-Agent RL - 多智能体博弈

## 创新性评分：0.88/1.0

## 适用场景

| 算法 | 动作空间 | 应用场景 | MCM题型 |
|-----|---------|---------|---------|
| DQN | 离散 | 路径选择、资源分配 | B, D |
| PPO | 连续 | 参数优化、控制 | D |
| MARL | 混合 | 博弈、多主体交互 | F |

## 使用方法

### DQN (离散决策)

```python
from rl_module import DQNAgent, Environment

# 创建环境和智能体
env = Environment(state_dim=10, action_dim=5)
agent = DQNAgent(
    state_dim=10,
    action_dim=5,
    hidden_dim=64,
    learning_rate=1e-3
)

# 训练
rewards = agent.train(env, episodes=1000)

# 评估
agent.evaluate(env, episodes=100)
agent.plot_learning_curve(save_path='figures/dqn_rewards.pdf')
```

### PPO (连续控制)

```python
from rl_module import PPOAgent

agent = PPOAgent(
    state_dim=10,
    action_dim=3,
    continuous=True
)

# 训练
agent.train(env, total_timesteps=100000)

# 获取最优策略
optimal_action = agent.get_action(state)
```

### 多智能体强化学习

```python
from rl_module import MultiAgentRL

marl = MultiAgentRL(
    num_agents=3,
    state_dim=20,
    action_dim=5
)

# 训练博弈均衡
marl.train(env, episodes=5000)

# 分析均衡策略
equilibrium = marl.analyze_equilibrium()
```

## 输出格式

```json
{
  "algorithm": "DQN",
  "training": {
    "episodes": 1000,
    "final_reward": 95.5,
    "convergence_episode": 650
  },
  "policy": {
    "type": "epsilon_greedy",
    "best_actions": [2, 1, 4, 0, 3]
  },
  "figures": ["learning_curve.pdf", "q_values.pdf"]
}
```

## O奖加分建议

- 与传统优化方法（如动态规划）对比
- 展示学习曲线和收敛性
- 分析学到的策略的可解释性
- 对于博弈问题，分析纳什均衡

## 相关技能

- `model-solver` - 传统优化求解器
- `game-theory` - 博弈论分析
- `sensitivity-analyzer` - 参数敏感性
