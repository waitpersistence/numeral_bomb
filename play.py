import torch
import numpy as np
from env import NumberBombEnv
from train import QNetwork  # 确保能导入之前的网络结构

def human_vs_ai():
    env = NumberBombEnv()
    # 1. 加载 AI
    input_dim = env.state_space
    output_dim = env.action_space
    model = QNetwork(input_dim, output_dim)
    model.load_state_dict(torch.load("number_bomb_agent.pth"))
    model.eval() # 开启预测模式

    print("--- 欢迎来到数字炸弹对战！ ---")
    print("请在脑海里想一个四位各不相同的数字（0-9）。")
    
    state = env.reset()
    done = False
    
    while not done:
        # AI 思考
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
            # 这里的逻辑：AI 会选择 Q 值最高的动作
            q_values_numpy = q_values.squeeze().numpy()
            # 将那些已经被排除掉的数字（state 为 0 的位置）的 Q 值设为一个极小的负数
            # 这样 argmax 永远不会选到它们
            masked_q_values = q_values_numpy + (state - 1) * 1e9 

            action_idx = np.argmax(masked_q_values)
        
        guess = env.all_numbers[action_idx]
        print(f"\nAI 猜你的数字是: {''.join(map(str, guess))}")
        
        # 用户输入反馈（位置和数字都对的个数）
        try:
            feedback = int(input("请输入匹配个数 (0-4): "))
        except ValueError:
            print("请输入数字！")
            continue

        if feedback == 4:
            print(f"AI 赢了！一共用了 {env.step_count + 1} 步。")
            done = True
        else:
            # 更新 AI 的状态（排除法）
            mask = (env.M[action_idx, :] == feedback)
            state = state * mask
            env.step_count += 1
            
            # 检查是否有逻辑矛盾
            if np.sum(state) == 0:
                print("你输入的反馈似乎有误，候选集已经空了！")
                break

if __name__ == "__main__":
    human_vs_ai()