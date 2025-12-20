import gymnasium as gym
import gymnasium_robotics#加载robotics环境，虽然程序中没用到这个库，但是必须有这行
import time

def main():
    # 1. 创建环境
    # render_mode="human" 用于弹窗显示动画，如果是在服务器无头模式跑训练则设为 None
    env = gym.make('FetchReach-v4', render_mode="human")

    # 2. 重置环境，获取初始观测值
    # seed 用于固定随机数种子，保证实验可复现（可选）
    observation, info = env.reset(seed=42)

    print("environment setup!")
    print(f"action_space:{env.action_space}")  # 查看动作维度 (通常是 Box(4,))
    print(f"obs_space:{env.observation_space}") # 查看观测结构 (Dict)

    # 3. 循环执行动作
    # 这里设置运行 1000 步
    for step in range(1000):
        # 随机采样一个动作
        # FetchReach 的动作通常是 4 维：[x移动, y移动, z移动, 夹爪开闭]
        action = env.action_space.sample()

        # 执行一步交互
        # terminated: 任务是否完成 (FetchReach中通常是指特定步数没完成算结束，或者到达目标)
        # truncated: 是否超时 (达到最大步数)
        observation, reward, terminated, truncated, info = env.step(action)

        # 为了方便人眼观察，稍微加一点延时 (实际训练时不要加)
        time.sleep(0.05)

        # 检查是否结束
        if terminated or truncated:
            print(f"Episode {step + 1} end, reset environment...")
            print(f"obs:{observation}")
            print(f"reward:{reward}")
            print(f"terminated:{terminated}")
            print(f"truncated:{truncated}")
            print(f"info:{info}")
            observation, info = env.reset()

    # 4. 关闭环境，释放资源
    env.close()
    print("show over")

if __name__ == "__main__":
    main()