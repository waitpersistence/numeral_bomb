import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from collections import deque
import random
from env import NumberBombEnv
import numpy as np
# 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # 5040 -> 2048 -> 1024 -> 5040
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=50000)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999

    def act(self, state):
        if random.random() < self.epsilon:
            # 探索期：为了效率，优先从“存活”的候选集中盲猜
            valid_actions = np.where(state == 1)[0]
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values).item())

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 当前 Q 值
        current_q = self.q_net(states).gather(1, actions)
        
        # 目标 Q 值 (Double DQN 逻辑可在此扩展，此处演示标准 DQN)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
            
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

if __name__ == "__main__":
    env = NumberBombEnv()
    agent = DQNAgent(state_dim=env.state_space, action_dim=env.action_space)
    
    episodes = 1000
    update_freq = 10
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            total_reward += reward
            
            # 设置一个兜底步数，防止死循环
            if env.step_count > 20: 
                break
                
        # 衰减 epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        # 定期更新目标网络
        if ep % update_freq == 0:
            agent.update_target_network()
            
        if ep % 100 == 0:
            print(f"Episode: {ep}, Steps to win: {env.step_count}, Epsilon: {agent.epsilon:.3f}")
        # 训练结束后保存模型权重
    torch.save(agent.q_net.state_dict(), "number_bomb_agent.pth")
    print("训练完成，模型已保存为 number_bomb_agent.pth")