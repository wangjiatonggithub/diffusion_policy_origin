import gymnasium as gym
import gymnasium_robotics#虽然这个库没有显式用到，但它起到了隐式的作用，向程序提供了环境的相关信息，删掉会报错

# 创建环境 (开启 'human' 模式，让你能看到画面)
env = gym.make('FetchReach-v4', render_mode='human')

print("动作空间 (Action Space):", env.action_space)
print("观测空间 (Observation Space):", env.observation_space)
# 初始化环境
obs, info = env.reset()

print("\n初始状态字典 keys:", obs.keys())
print("当前机械臂末端位置 (achieved_goal):", obs['achieved_goal'])
print("目标红点位置 (desired_goal):", obs['desired_goal'])

# 开始循环 (例如跑 500 步)
for step in range(500):
    # 1. 随机选择一个动作 (Sampling)
    action = env.action_space.sample()

    # 2. 执行动作 (Stepping)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: Reward = {reward}, Obs={obs}")
    #obs中包含obs（末端执行器位置（3维）、速度（3维）、夹爪状态（2维）包括两个夹爪之间的距离和它们的开合速度物体相对位置），achieved_goal（末端执行器当前位置3维），desired_goal（末端执行器目标位置3维）

    # 3. 如果回合结束 (例如超时)，重置环境
    if terminated or truncated:
        obs, info = env.reset()

# 关闭环境
env.close()

