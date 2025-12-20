import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)#将transitions元组列表中每个元组的对应项提取出来组成一个新元组
        return np.array(state), action, reward, np.array(next_state), done#np.array函数将普通元组转换为numpy列表，更易于pytorch处理

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

class Qnet(torch.nn.Module):#定义父类torch.nn.Module子类Qnet
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()#确保Qnet继承了父类torch.nn.Module的所有基本功能和属性
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)#定义输入层为全连接层
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)#定义输出层为全连接层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)#若随机数小于epsilon，则要在动作维度中随机选一个执行
        else:#否则选取值函数最大的动作
            state = torch.tensor([state], dtype=torch.float).to(self.device)#将状态数据转换成tensor格式
            action = self.q_net(state).argmax().item()#动作输出为神经网络输出最大值的索引，再取整
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值，先计算所有states在所有动作下的q值，在利用gather函数在actions矩阵中根据实际采用的动作获取实际(s,a)对的q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)#max(1)[0]的作用是按列选取每一行中的最大值，并只保留元组中的第一个元素，即最大值而抛弃最大值索引
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标，其中done=1表示回合结束，未来奖励被清零
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  #计算梯度
        self.optimizer.step()#更新参数

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络，其中load_state_dict和state_dict是torch网络中自带的方法，分别用于设置模型参数和获取模型参数
        self.count += 1

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

#创建环境，为gym平台中的倒立摆
env_name = 'CartPole-v1'
# env = gym.make(env_name, render_mode='human')#实时显示仿真
env = gym.make(env_name, render_mode="rgb_array")#不实时显示仿真
#随机种子设置
random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]#获取环境状态维度
action_dim = env.action_space.n#获取环境动作维度
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)#rl=learning rate

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            if 50*i+i_episode==150:
                pass
            episode_return = 0
            state,info = env.reset(seed=0)#重置环境
            done = False#False表示回合尚未结束
            num=0
            while not done:
                action = agent.take_action(state)#选择动作
                next_state, reward, terminated, truncated, _ = env.step(action)#与环境交互，’_‘表示忽略额外信息
                if num>200:
                    truncated=True
                done = terminated or truncated
                replay_buffer.add(state, action, reward, next_state, done)#存储经验
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
                num += 1
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])#显示最近10回合的平均奖励
                })
            pbar.update(1)

# 1. 打印提示信息
print("\n" + "=" * 50)
print("finish!")
print("=" * 50)

user_input = input("press y to continue: ")

if user_input.lower() == 'y':
    print("start to show the demo")
# 2. 创建一个新的环境，开启实时显示模式 ("human")
# 注意：我们使用新的变量名 render_env
    render_env = gym.make(env_name, render_mode='human')

    # 3. 让 Q 网络进入评估模式 (关闭 dropout 等，确保结果稳定)
    agent.q_net.eval()

    # 4. 演示 N 个回合 (例如 5 个回合)
    num_demo_episodes = 5
    for episode in range(num_demo_episodes):
        episode_return = 0
        state, info = render_env.reset()
        done = False

        print(f"episode: {episode + 1}/{num_demo_episodes} ---")

        while not done:
            # 在演示阶段，我们必须使用贪婪策略 (Greedy, 100% 选 Q 值最大的动作)
            # 不再使用 epsilon-greedy，因为不需探索

            # 4a. 关闭梯度计算 (因为只是推理，不训练)
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float).to(agent.device)
                # 动作输出为 Qnet 输出的最大值索引
                action = agent.q_net(state_tensor).argmax().item()

            # 4b. 执行动作
            next_state, reward, terminated, truncated, info = render_env.step(action)
            state = next_state
            done = terminated or truncated
            episode_return += reward

            # 可以在这里加一个 time.sleep(0.01) 减慢速度，但 CartPole 通常不需要

        print(f"episode_return: {episode_return:.2f}")

# 5. 关闭渲染环境
render_env.close()

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)#从第9个数开始，计算前9个数的平均值，作为新的值，用于平滑列表
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()