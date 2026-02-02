# 强化学习实现任务 (Reinforcement Learning)

## 角色

你是强化学习专家，负责实现DQN、PPO等算法用于序列决策问题。强化学习适用于B/D/F题型，创新性评分0.88-0.94。

## 输入

- `problem_type`: 题目类型 (通常为B/D/F)
- `state_space`: 状态空间定义
- `action_space`: 动作空间定义
- `reward_function`: 奖励函数设计
- `environment`: 环境描述

---

## RL算法选择

| 算法 | 动作空间 | 适用场景 | 创新分数 |
|------|---------|---------|---------|
| DQN | 离散 | 调度、路径规划 | 0.85 |
| PPO | 连续/离散 | 通用、稳定性好 | 0.88 |
| SAC | 连续 | 机器人、控制 | 0.90 |
| MARL | 多智能体 | 博弈、协作 | 0.92 |

---

## 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

# ============ DQN实现 ============

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """DQN网络（Dueling架构）"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # 共享特征提取
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class DQNAgent:
    """
    DQN智能体
    
    实现Double DQN + Dueling架构
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 1000,
        target_update: int = 100,
        buffer_size: int = 10000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        
        # 网络
        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 经验回放
        self.buffer = ReplayBuffer(buffer_size)
        
        self.steps = 0
        self.losses = []
    
    def get_epsilon(self) -> float:
        """计算当前epsilon"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps / self.epsilon_decay)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """epsilon-greedy动作选择"""
        if training and random.random() < self.get_epsilon():
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self, batch_size: int = 64) -> Optional[float]:
        """更新网络"""
        if len(self.buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # 当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: 用policy_net选动作，用target_net评估
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # 更新目标网络
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def train(self, env, num_episodes: int = 1000, verbose: int = 100):
        """训练智能体"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.buffer.push(state, action, reward, next_state, done)
                self.update()
                
                state = next_state
                total_reward += reward
            
            episode_rewards.append(total_reward)
            
            if (episode + 1) % verbose == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.get_epsilon():.3f}")
        
        return episode_rewards


# ============ PPO实现 ============

class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        continuous: bool = False
    ):
        super().__init__()
        self.continuous = continuous
        
        # 共享特征
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        if continuous:
            # 连续动作：输出均值和标准差
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # 离散动作：输出概率
            self.actor = nn.Linear(hidden_dim, action_dim)
        
        # 价值函数
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor):
        features = self.shared(state)
        
        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp()
            return mean, std, self.critic(features)
        else:
            action_probs = F.softmax(self.actor(features), dim=-1)
            return action_probs, self.critic(features)
    
    def get_action(self, state: torch.Tensor):
        if self.continuous:
            mean, std, value = self.forward(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, value
        else:
            probs, value = self.forward(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, value
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        if self.continuous:
            mean, std, value = self.forward(state)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            probs, value = self.forward(state)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        return log_prob, value.squeeze(), entropy


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) 智能体
    
    稳定的策略梯度方法
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        continuous: bool = False
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # 网络
        self.actor_critic = ActorCritic(state_dim, action_dim, continuous=continuous)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # 存储轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state: np.ndarray):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action(state_tensor)
        
        return action.squeeze().numpy(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算GAE优势估计"""
        rewards = torch.FloatTensor(self.rewards)
        values = torch.FloatTensor(self.values + [next_value])
        dones = torch.FloatTensor(self.dones)
        
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def update(self, next_value: float, epochs: int = 4, batch_size: int = 64):
        """PPO更新"""
        # 计算优势
        advantages, returns = self.compute_gae(next_value)
        
        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        for _ in range(epochs):
            # 随机打乱
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # 获取batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 评估当前策略
                log_probs, values, entropy = self.actor_critic.evaluate(
                    batch_states, batch_actions
                )
                
                # PPO裁剪
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                value_loss = F.mse_loss(values, batch_returns)
                
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
        
        # 清空缓存
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()


# ============ 环境示例 ============

class SimpleSchedulingEnv:
    """
    简单调度环境示例
    
    状态：任务队列状态
    动作：选择处理哪个任务
    奖励：-完成时间（越小越好）
    """
    
    def __init__(self, num_jobs: int = 5, num_machines: int = 2):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.state_dim = num_jobs + num_machines
        self.action_dim = num_jobs
        
        self.reset()
    
    def reset(self):
        # 随机生成任务处理时间
        self.job_times = np.random.randint(1, 10, size=self.num_jobs).astype(float)
        self.job_done = np.zeros(self.num_jobs)
        self.machine_available = np.zeros(self.num_machines)
        self.current_time = 0
        
        return self._get_state()
    
    def _get_state(self):
        return np.concatenate([
            self.job_times * (1 - self.job_done),
            self.machine_available
        ])
    
    def step(self, action: int):
        # 检查动作是否有效
        if self.job_done[action] == 1:
            return self._get_state(), -10, False, {}
        
        # 找到最早可用的机器
        machine = np.argmin(self.machine_available)
        start_time = self.machine_available[machine]
        
        # 执行任务
        self.job_done[action] = 1
        self.machine_available[machine] = start_time + self.job_times[action]
        
        # 奖励为负的完成时间
        reward = -self.job_times[action]
        
        # 检查是否完成
        done = np.all(self.job_done == 1)
        
        if done:
            # 额外奖励：总完成时间越短越好
            makespan = np.max(self.machine_available)
            reward += -makespan * 0.1
        
        return self._get_state(), reward, done, {}


# ============ 使用示例 ============

def example_dqn_scheduling():
    """DQN调度示例"""
    env = SimpleSchedulingEnv(num_jobs=5, num_machines=2)
    
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=1e-3,
        gamma=0.99
    )
    
    print("Training DQN agent...")
    rewards = agent.train(env, num_episodes=500, verbose=100)
    
    print(f"Final average reward: {np.mean(rewards[-50:]):.2f}")
    
    return agent, rewards


if __name__ == "__main__":
    agent, rewards = example_dqn_scheduling()
```

---

## 输出格式

```json
{
  "rl_algorithm": "PPO",
  "architecture": {
    "actor_network": "2-layer MLP (64 units)",
    "critic_network": "shared backbone",
    "total_params": 12850
  },
  "training_config": {
    "episodes": 1000,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2
  },
  "training_results": {
    "final_avg_reward": 145.3,
    "convergence_episode": 650,
    "training_time_hours": 0.5
  },
  "policy_analysis": {
    "learned_strategy": "优先处理短任务，平衡机器负载",
    "action_distribution": {"action_0": 0.35, "action_1": 0.28, "...": "..."}
  },
  "comparison_with_baselines": {
    "random_policy": 89.2,
    "greedy_policy": 125.6,
    "rl_policy": 145.3,
    "improvement_over_greedy": "15.7%"
  },
  "innovation_highlights": [
    "使用PPO实现稳定的策略优化",
    "比贪心策略提升15.7%",
    "学习到有意义的调度策略"
  ]
}
```
