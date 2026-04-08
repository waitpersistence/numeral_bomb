import torch
import numpy as np
import random
from env import NumberBombEnv
from train import QNetwork

def calculate_feedback(guess_tuple, target_tuple):
    """计算两个数字元组之间位置和数值完全匹配的个数"""
    return sum(g == t for g, t in zip(guess_tuple, target_tuple))

def play_battle():
    env = NumberBombEnv()
    
    # 1. 初始化 AI 的进攻端 (加载模型)
    model = QNetwork(env.state_space, env.action_space)
    try:
        model.load_state_dict(torch.load("number_bomb_agent.pth"))
        model.eval()
        print("成功加载 AI 大脑，它已经准备好进攻了！")
    except:
        print("未找到训练好的模型，AI 将使用随机策略对战。")

    # 2. 初始化 AI 的防御端 (AI 选一个秘密数字)
    ai_target_idx = random.randint(0, env.action_space - 1)
    ai_secret_num = env.all_numbers[ai_target_idx]
    
    # 3. 初始化玩家的进攻端状态 (AI 对玩家数字的追踪)
    ai_tracking_state = env.reset() # 这里的 state 用来追踪玩家的数字
    
    print("\n" + "="*30)
    print("--- 真正的对决：人类 vs 强化学习 AI ---")
    print("规则：双方各选一个四位各不相同的数字，轮流猜测。")
    print("="*30)
    print("AI 已经选好了它的数字，你也准备好你的数字了吗？")
    
    turn = 1
    while True:
        print(f"\n--- 第 {turn} 轮 ---")
        
        # --- 玩家回合 (Player's Turn) ---
        user_guess_str = input("你的回合 -> 猜猜 AI 的数字 (如1234): ")
        if len(user_guess_str) != 4 or not user_guess_str.isdigit():
            print("输入无效，请输入4位数字！")
            continue
        
        user_guess = tuple(int(d) for d in user_guess_str)
        player_feedback = calculate_feedback(user_guess, ai_secret_num)
        print(f"AI 反馈: 匹配个数为 {player_feedback}")
        
        if player_feedback == 4:
            print("\n恭喜你！你先猜出了 AI 的数字，人类守住了尊严！")
            break

        # --- AI 回回合 (AI's Turn) ---
        print("\nAI 的回合 -> 正在思考...")
        
        # 使用模型预测 + 动作掩码
        state_tensor = torch.FloatTensor(ai_tracking_state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor).squeeze().numpy()
            # 动作掩码：只在活着的候选集中选
            masked_q_values = q_values + (ai_tracking_state - 1) * 1e9
            ai_action_idx = np.argmax(masked_q_values)
        
        ai_guess = env.all_numbers[ai_action_idx]
        ai_guess_str = "".join(map(str, ai_guess))
        print(f"AI 猜你的数字是: {ai_guess_str}")
        
        # 获取玩家的反馈
        try:
            user_to_ai_feedback = int(input("给 AI 的反馈 (0-4): "))
        except ValueError:
            print("输入错误，AI 这一轮算白猜了...")
            continue
            
        if user_to_ai_feedback == 4:
            print(f"\nAI 赢了！它识破了你的数字 {ai_guess_str}。")
            break
        
        # AI 根据玩家反馈更新它的逻辑库
        mask = (env.M[ai_action_idx, :] == user_to_ai_feedback)
        ai_tracking_state *= mask
        
        remaining = int(np.sum(ai_tracking_state))
        print(f"(AI 暗自分析：你的数字可能还有 {remaining} 种可能)")
        
        if remaining == 0:
            print("警告：你给的反馈有逻辑矛盾！AI 的逻辑崩溃了。")
            break
            
        turn += 1

if __name__ == "__main__":
    play_battle()