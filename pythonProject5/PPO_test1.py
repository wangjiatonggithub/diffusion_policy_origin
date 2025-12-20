# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")] # 以本文件为例生成PPO_test1
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True # pytorch开启确定性模式，以确保代码可复现性
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False # 是否使用Weights and Biases（WandB）进行实验跟踪，将实时数据上传到云端以便多端查看
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000 # 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048 # 规定了一个环境的batch_size
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True # 使用学习率退火，随着时间逐步衰减
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99 # 折扣因子
    """the discount factor gamma"""
    gae_lambda: float = 0.95 # GAE的平滑参数
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32 # PPO通常分成多个mini-batch进行优化
    """the number of mini-batches"""
    update_epochs: int = 10 # 每个batch训练的次数
    """the K epochs to update the policy"""
    norm_adv: bool = True # 是否对GAE进行归一化
    """Toggles advantages normalization"""
    clip_coef: float = 0.2 # 策略比率的裁剪范围
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True # 价值函数是否裁剪
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0 # 熵正则化系数
    """coefficient of the entropy"""
    vf_coef: float = 0.5 # 价值函数正则化系数
    """coefficient of the value function"""
    max_grad_norm: float = 0.5 # 梯度裁剪的最大范数
    """the maximum norm for the gradient clipping"""
    target_kl: float = None # 目标KL散度阈值，本算法中没有
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env) # 限制动作在环境允许的物理范围内
        env = gym.wrappers.NormalizeObservation(env) # 将环境中的数据标准化到一个正态分布内，方便神经网络训练
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10)) # 强制将环境中的异常数据限制在[-10，10]
        env = gym.wrappers.NormalizeReward(env, gamma=gamma) # 将奖励标准化，使程序不用为每个环境都重新调整learning_rate
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10)) # 奖励值截断
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0): # 对神经网络的参数进行初始化
    torch.nn.init.orthogonal_(layer.weight, std) # 生成一个正交矩阵作为权重，并乘上std来保持输入和输出方差稳定
    torch.nn.init.constant_(layer.bias, bias_const) # 生成一个常数作为偏置
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__() # 初始化pytorch的父类nn.Module
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(), # 激活函数为tanh
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01), # 方差取得很小，防止训练不稳定
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape))) # 更新agent网络参数时也顺便更新方差参数

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None): # 若有输入得动作，则直接生成该动作得方差，若没有输入得动作，就根据当前策略生成新动作
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean) # 这里是将方差的batch维度由1扩展到相应维度
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args) # 读取实验参数
    args.batch_size = int(args.num_envs * args.num_steps) # 给batch_size赋值
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # 每个mini-batch的数据量
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}" # 实验名
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported" # PPO是连续动作！

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5) # eps作用是防止分母为0

    # ALGO Logic: Storage setup 预先分配指定大小的pytorch张量数据，并移动到GPU上
    # 为一个episode在所有环境中的采样数据分配空间
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device) # 最终生成维度为(args.num_steps, args.num_envs, envs.single_observation_space.shape)的tensor
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr: # 学习率随训练时间不断衰减
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps): # 根据当前的策略采样全部num_steps
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad(): # 关闭梯度计算，以免占用显存
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten() # .flatten()将多维tensor展开成一维tensor，方便神经网络计算
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1) # view(-1)将tensor展开成一维tensor
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos: # 一个episode结束后打印return
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad(): # 计算GAE
            next_value = agent.get_value(next_obs).reshape(1, -1) # reshape(1,-1)的作用还是使多维tensor展开成一维tensor
            advantages = torch.zeros_like(rewards).to(device) # 用zeros_like函数的目的是指定新建的tensor维度和rewards相同
            lastgaelam = 0 # 初始化GAE累积值
            for t in reversed(range(args.num_steps)): # 从后往前计算GAE
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t] # TD误差
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam # 计算advantage
            returns = advantages + values # A=R-V

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = [] # 存放策略比率超出clip范围的比例，用于监控
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds) # 打乱batch中的数据
            for start in range(0, args.batch_size, args.minibatch_size): # 把训练分成一个个小批次，以降低显存压力
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds] # 新策略和老策略的概率对数差
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean() # 旧的计算logratio的方法，用于监控和调试
                    approx_kl = ((ratio - 1) - logratio).mean() # PPO中实际使用的计算logratio的方法
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()] # 判断ratio是否超出clip范围

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv: # 是否对advantage进行归一化
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss: # 价值函数也有相应的裁剪策略
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # 价值网络和策略网络同时更新，可以提高训练效率并保持梯度稳定
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) # 梯度裁剪，防止梯度爆炸
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl: # 如果策略的kl散度超出了目标值，就停止本次训练
                break

        # 对比预测价值和真实回报之间的方差，评估价值函数的预测效果
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    # envs.close()
    # writer.close()

    # 1. 只要训练结束，先立刻保存模型！(防止手滑关掉程序导致白跑)
    # 确保 runs 文件夹存在
    if not os.path.exists(f"runs/{run_name}"):
        os.makedirs(f"runs/{run_name}")

    model_path = f"runs/{run_name}/PPO_test1.pth"
    torch.save(agent.actor_mean.state_dict(), model_path)
    print(f"model haven been saved in: {model_path}")

    # 2. 关闭训练环境
    envs.close()
    writer.close()


    # 程序会在这里“卡住”，直到你在控制台输入内容并回车
    user_input = input("press y to continue: ")

    # 3. 根据你的输入决定是否播放
    if user_input.lower() == 'y':
        print("start to show the demo")

        # 创建可视化环境
        eval_env = gym.make(args.env_id, render_mode="human")

        obs, info = eval_env.reset()
        # done = False

        # 让 agent 进入 eval 模式（关闭 dropout 等）
        agent.eval()

        episode_reward = 0
        num_step=500
        for _ in range(num_step):
            # 转成 torch tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)

            action = action.cpu().numpy()[0]

            # 转 numpy
            # action = action_mean.cpu().numpy()[0]

            # 送入环境执行
            obs, reward, terminated, truncated, info = eval_env.step(action)
            # done = terminated or truncated

            episode_reward += reward

        print(f"Simulation finished, total reward = {episode_reward}")
        eval_env.close()


    else:
        print("exit the program")