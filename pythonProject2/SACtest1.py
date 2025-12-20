# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# from cleanrl_utils.buffers import ReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True # 让pytorch运算可复现
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False # 是否上传到WandB云端
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False # 是否录制机器人动作视频
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v5"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1 # 同时开的环境个数
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005 # 参数软更新系数
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True # 是否自动调整alpha
    """automatic tuning of the entropy coefficient"""

# 不显示仿真环境
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        # 如果需要录像且是第0个环境，就加个录像插件
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id) # 创建普通环境
        env = gym.wrappers.RecordEpisodeStatistics(env) # 记录每一局的总分和步数，方便画图
        env.action_space.seed(seed) # 固定动作空间的随机种子
        return env

    return thunk

# 实时显示仿真环境
# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         # 只要是第 0 个环境（通常你就跑这一个），就开启 human 模式
#         if idx == 0 and not capture_video:
#             env = gym.make(env_id, render_mode="human")
#         elif capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env.action_space.seed(seed)
#         return env
#
#     return thunk

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # 定义三层神经网络结构
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1) # 拼接状态x和动作a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # 前两层输出经过relu激活函数，第三层不用
        return x

# 定义最大最小的动作范围
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env): # 利用环境作为输入是为了获取状态和动作空间的维数
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256) # prod表示乘积，用于计算神经元的个数
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        # 将神经网络tanh激活函数输出的[-1,1]之间的数映射到真实的动作输出范围上，register_buffer规定了scale和bias这两个参数不能被训练，作为模型状态能被自动保存和加载，同时能跟随模型训练设备改变自动在CPU/GPU中移动
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        # 将tanh函数的输出拉伸平移到[LOG_STD_MIN,LOG_STD_MAX]范围，tanh函数的作用是防止神经网络输出的对数方差太大或太小，拉伸平移的作用是让对数方差映射到我们更关注的范围上
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x) # 利用self实例调用了call函数，call函数能自动调用forward函数并具备一定的检查功能
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))重参数化采样
        # 对新求出的动作进行tanh函数和拉伸平移映射，确保输出到执行器真实范围
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # 计算动作x_t的对数概率
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound消除函数tanh变换对概率的影响，在原来概率基础上除以tanh函数的导数，放到log函数上就是减去
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        # 将动作矩阵所有维度的概率相加
        log_prob = log_prob.sum(1, keepdim=True)
        # 对动作均值也进行tanh函数和拉伸平移映射
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":

    args = tyro.cli(Args) # 读取Agrs中的默认参数
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}" # 生成实验名字
    # 若track=True，设置wanb
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # 设置tensorboard记录器
    writer = SummaryWriter(f"runs/{run_name}")
    # 将参数写进tensorboard中
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding设置所有随机数种子，保证实验可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup创建环境（同步向量环境）
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    # 检查环境中的动作空间（single_action_space）是不是连续的（BOX）
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict()) # 模型状态自动加载，这时由register_buffer注册的数据会被自动加载
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item() # to(device)的作用是将Tensor搬到GPU上，item()作用是将tensor变量转换为普通的浮点数
        # 定义时变的参数alpha并定义优化器，并且优化的是log_alpha，这样能保证alpha始终为正
        log_alpha = torch.zeros(1, requires_grad=True, device=device) # 定义了一个初始值为0的标量，'requires_grad=True'表明这个变量可以被优化，需要存储其梯度
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32 # 将从环境中采集的变量都强制为浮点数，因为神经网络默认处理的数据是float32

    # 初始化ReplayBuffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        # handle_timeout_termination=False,
    )
    start_time = time.time() # 计时器，为了计算后面的SPS

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed) # 重置环境

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            # 若还没到学习时间，就随机采样动作
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device)) # obs代表环境采集的状态
            actions = actions.detach().cpu().numpy() # 转成numpy给环境用

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # 若有一个回合结束了，就打印这一局得分，并写入Tensorboard
        # if "final_info" in infos:# final_info是字典中的一个健，回合没结束时这个健不存在
        #     for info in infos["final_info"]:
        #         if info is not None:
        #             # 打印这一局的得分
        #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #             # 写入Tensorboard
        #             writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #             writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #             break
        if "episode" in infos:
            env_dones = infos["_episode"] # "_episode”中存储着回合是否结束的信息
            for idx, done in enumerate(env_dones):
                if done:
                    # 4. 只有 done=True 的环境，其 episode 数据才是这一局刚结算的
                    ret = infos["episode"]["r"][idx]  # 获取得分 Return
                    len_ = infos["episode"]["l"][idx]  # 获取步数 Length

                    print(f"global_step={global_step}, episodic_return={ret}, length={len_}")

                    # 写入 TensorBoard
                    writer.add_scalar("charts/episodic_return", ret, global_step)
                    writer.add_scalar("charts/episodic_length", len_, global_step)
                    break
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # 处理由于时间到了时状态从最后一帧直接变到原点的问题
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            # 若是截断，即时间到了，next_obs应该还是最后一个观测值
            if trunc:
                if "final_observation" in infos:
                    real_next_obs[idx] = infos["final_observation"][idx]
        # rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        # 我们手动传一个空字典列表给它，骗过 SB3 的检查
        # 加入replay buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, [{}])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # 从buffer中抽取数据
            data = rb.sample(args.batch_size)

            # 更新 Critic 网络
            with torch.no_grad():
                # 用 Actor 算出下一步的动作和熵
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)#这里只计算了当前动作的对数概率来表示熵，用采样代替期望
                # 两个 Target Q 网络打分
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                # 取最小值，并加上熵奖励 (alpha * log_pi)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # Bellman 公式计算目标 Q 值
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)#(1 - data.dones.flatten())为终止条件，若回合结束了后面就没有奖励了
                #.flatten()和.view(-1)作用相同，但flatten常用于numpy数据，更稳妥不会报错；view常用于pytorch数据，更简洁高效
            # 计算当前 Q 网络的预测值
            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            # 计算MSE损失
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model 反向传播更新 Q 网络
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # 更新 Actor网络，延迟更新（每两步更新一次）
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(# actor在延迟更新时要把之前没更新的都补上
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)# 更新策略网络时计算动作价值用的是新策略
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()# actor网络损失：期望

                    # 更新actor网络
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # 更新alpha熵系数，目的是让当前的熵更接近目标熵
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            # 软更新target网络
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    # envs.close()
    # writer.close()

    # 1. 只要训练结束，先立刻保存模型！(防止手滑关掉程序导致白跑)
    # 确保 runs 文件夹存在
    if not os.path.exists(f"runs/{run_name}"):
        os.makedirs(f"runs/{run_name}")

    model_path = f"runs/{run_name}/sac_actor.pth"
    torch.save(actor.state_dict(), model_path)
    print(f"model haven been saved in: {model_path}")

    # 2. 关闭训练环境
    envs.close()
    writer.close()

    # 3. 【关键步骤】在这里设置路障，等待你的命令
    print("\n" + "=" * 50)
    print("finish！")
    print(f"model path: {model_path}")
    print("=" * 50)

    # 程序会在这里“卡住”，直到你在控制台输入内容并回车
    user_input = input("press y to continue: ")

    # 4. 根据你的输入决定是否播放
    if user_input.lower() == 'y':
        print("start to show the demo")

        # 创建可视化环境
        eval_env = gym.make(args.env_id, render_mode="human")
        obs, _ = eval_env.reset()

        step_num=500

        for _ in range(step_num):
            # 维度修正 (unsqueeze)
            obs_tensor = torch.Tensor(obs).to(device).unsqueeze(0)

            with torch.no_grad():
                # 获取动作 (使用均值 mean 表现更好)
                _, _, mean = actor.get_action(obs_tensor)
                action = torch.tanh(mean) * actor.action_scale + actor.action_bias

            # 维度拆解 ([0])
            action = action.cpu().numpy()[0]

            # 执行动作
            next_obs, _, terminated, truncated, _ = eval_env.step(action)
            obs = next_obs

            if terminated or truncated:
                obs, _ = eval_env.reset()

    else:
        print("exit the program")
