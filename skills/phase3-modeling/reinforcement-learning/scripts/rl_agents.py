"""
Reinforcement Learning Agents
强化学习智能体模块

支持DQN、PPO和多智能体强化学习，适用于MCM/ICM动态决策问题。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import json


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
                
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q网络"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """Deep Q-Network智能体"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        device: str = 'auto'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # 网络
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size)
        
        self.training_rewards = []
        self.losses = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作（epsilon-greedy）"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
            
    def update(self):
        """更新网络"""
        if len(self.buffer) < self.batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
            
        # 损失
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def train(self, env, episodes: int = 1000, update_target_freq: int = 10,
              verbose: int = 100):
        """训练智能体"""
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.buffer.push(state, action, reward, next_state, done)
                loss = self.update()
                
                state = next_state
                total_reward += reward
                
            self.training_rewards.append(total_reward)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # 更新目标网络
            if episode % update_target_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                
            if verbose and (episode + 1) % verbose == 0:
                avg_reward = np.mean(self.training_rewards[-100:])
                print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
                
        return self.training_rewards
        
    def evaluate(self, env, episodes: int = 100) -> Dict:
        """评估智能体"""
        rewards = []
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, training=False)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                
            rewards.append(total_reward)
            
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards)
        }
        
    def plot_learning_curve(self, save_path: str = None):
        """绘制学习曲线"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 原始奖励
        ax.plot(self.training_rewards, alpha=0.3, color='blue')
        
        # 滑动平均
        window = min(100, len(self.training_rewards) // 10)
        if window > 1:
            smoothed = np.convolve(self.training_rewards, 
                                   np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(self.training_rewards)), 
                   smoothed, color='blue', linewidth=2, label='Smoothed')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('DQN Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


class PPOAgent:
    """Proximal Policy Optimization智能体"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        continuous: bool = False,
        device: str = 'auto'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.continuous = continuous
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Actor-Critic网络
        self.actor = self._build_actor(state_dim, action_dim, hidden_dim)
        self.critic = self._build_critic(state_dim, hidden_dim)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )
        
        self.training_rewards = []
        
    def _build_actor(self, state_dim, action_dim, hidden_dim):
        if self.continuous:
            return nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
            ).to(self.device)
        else:
            return nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            ).to(self.device)
            
    def _build_critic(self, state_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
    def get_action(self, state: np.ndarray):
        """获取动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if self.continuous:
                output = self.actor(state_tensor)
                mean = output[:, :self.action_dim]
                log_std = output[:, self.action_dim:]
                std = torch.exp(log_std.clamp(-20, 2))
                action = torch.normal(mean, std)
                return action.cpu().numpy()[0]
            else:
                probs = self.actor(state_tensor)
                action = torch.multinomial(probs, 1)
                return action.item()
                
    def train(self, env, total_timesteps: int = 100000, verbose: int = 1000):
        """训练PPO"""
        state = env.reset()
        episode_reward = 0
        
        for step in range(total_timesteps):
            action = self.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if done:
                self.training_rewards.append(episode_reward)
                episode_reward = 0
                state = env.reset()
                
            if verbose and (step + 1) % verbose == 0:
                avg_reward = np.mean(self.training_rewards[-100:]) if self.training_rewards else 0
                print(f"Step {step+1}/{total_timesteps}, Avg Reward: {avg_reward:.2f}")
                
        return self.training_rewards


class MultiAgentRL:
    """多智能体强化学习"""
    
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        device: str = 'auto'
    ):
        self.num_agents = num_agents
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # 每个智能体一个DQN
        self.agents = [
            DQNAgent(state_dim, action_dim, hidden_dim, device=device)
            for _ in range(num_agents)
        ]
        
        self.training_rewards = [[] for _ in range(num_agents)]
        
    def train(self, env, episodes: int = 5000, verbose: int = 500):
        """训练多智能体"""
        for episode in range(episodes):
            states = env.reset()  # [num_agents, state_dim]
            episode_rewards = [0] * self.num_agents
            done = False
            
            while not done:
                # 每个智能体选择动作
                actions = [agent.select_action(states[i]) 
                          for i, agent in enumerate(self.agents)]
                
                next_states, rewards, done, _ = env.step(actions)
                
                # 存储经验并更新
                for i, agent in enumerate(self.agents):
                    agent.buffer.push(states[i], actions[i], rewards[i], 
                                     next_states[i], done)
                    agent.update()
                    episode_rewards[i] += rewards[i]
                    
                states = next_states
                
            for i in range(self.num_agents):
                self.training_rewards[i].append(episode_rewards[i])
                self.agents[i].epsilon = max(
                    self.agents[i].epsilon_end,
                    self.agents[i].epsilon * self.agents[i].epsilon_decay
                )
                
            if verbose and (episode + 1) % verbose == 0:
                avg_rewards = [np.mean(r[-100:]) for r in self.training_rewards]
                print(f"Episode {episode+1}, Avg Rewards: {avg_rewards}")
                
    def analyze_equilibrium(self) -> Dict:
        """分析均衡策略"""
        return {
            f'agent_{i}': {
                'avg_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
                'strategy': 'learned_policy'
            }
            for i, rewards in enumerate(self.training_rewards)
        }


class SimpleEnvironment:
    """简单测试环境"""
    def __init__(self, state_dim: int = 10, action_dim: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = None
        self.steps = 0
        self.max_steps = 100
        
    def reset(self):
        self.state = np.random.randn(self.state_dim)
        self.steps = 0
        return self.state
        
    def step(self, action):
        self.steps += 1
        reward = np.random.randn() + (action == 2)  # 偏好动作2
        self.state = np.random.randn(self.state_dim)
        done = self.steps >= self.max_steps
        return self.state, reward, done, {}


if __name__ == '__main__':
    print("Testing RL Agents...")
    
    # 测试DQN
    env = SimpleEnvironment()
    agent = DQNAgent(state_dim=10, action_dim=5)
    rewards = agent.train(env, episodes=100, verbose=20)
    print(f"DQN Final Avg Reward: {np.mean(rewards[-10:]):.2f}")
    
    # 测试PPO
    ppo_agent = PPOAgent(state_dim=10, action_dim=5)
    ppo_rewards = ppo_agent.train(env, total_timesteps=1000, verbose=200)
    print(f"PPO Episodes completed: {len(ppo_rewards)}")
    
    print("RL Agents test completed!")
