import numpy as np
import itertools
import torch

class NumberBombEnv:
    def __init__(self):
        # 1. 生成所有 5040 个合法的四位互不相同的数字
        self.all_numbers = list(itertools.permutations(range(10), 4))
        self.num_to_idx = {num: i for i, num in enumerate(self.all_numbers)}
        self.action_space = len(self.all_numbers) # 5040
        self.state_space = len(self.all_numbers)  # 5040维二值向量
        
        # 2. 预计算反馈矩阵 (5040, 5040)
        # M[i, j] 表示如果猜测是 i，目标是 j，会得到几个位置相同的反馈
        print("正在预计算反馈矩阵...")
        self.M = np.zeros((self.action_space, self.action_space), dtype=np.int8)
        for i, guess in enumerate(self.all_numbers):
            for j, target in enumerate(self.all_numbers):
                # 计算位置和数字均相同的个数
                self.M[i, j] = sum(g == t for g, t in zip(guess, target))
                
        self.reset()

    def reset(self):
        # 随机选择一个目标数字的索引
        self.target_idx = np.random.randint(self.action_space)
        # 初始状态：所有 5040 个数字都有可能是答案 (全1向量)
        self.state = np.ones(self.state_space, dtype=np.float32)
        self.step_count = 0
        return self.state.copy()

    def step(self, action_idx):
        self.step_count += 1
        
        # 1. 从环境中获取真实反馈 (对比动作和目标)
        feedback = self.M[action_idx, self.target_idx]
        
        # 2. 判断是否获胜 (4个全部正确)
        done = bool(feedback == 4)
        
        # 3. 状态转移（核心降维打击）
        # 假设当前猜测为 action_idx，得到了 feedback。
        # 那么只有那些与 action_idx 对比也能产生相同 feedback 的候选数字，才可能存活。
        # 利用预计算矩阵，一行代码完成过滤！
        mask = (self.M[action_idx, :] == feedback)
        self.state = self.state * mask  # 逻辑与操作，排除不可能的选项
        
        # 4. 奖励设计
        if done:
            reward = 10.0  # 获胜奖励
        else:
            reward = -1.0  # 步数惩罚
            
            # 可选的高阶奖励：基于信息增益（熵减）
            # remaining = np.sum(self.state)
            # reward = -1.0 + 0.01 * (5040 - remaining) 
            
        return self.state.copy(), reward, done, {}