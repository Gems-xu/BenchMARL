#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
from gemsmarl.algorithms import MappoConfig
from gemsmarl.environments import VmasTask
from gemsmarl.experiment import Experiment, ExperimentConfig
from gemsmarl.models import MlpConfig, PinnConfig, SafePinnPPOConfig

# 设置随机种子以确保可重复性
# torch.manual_seed(0)

# 1. 配置算法 (MAPPO)
algorithm_config = MappoConfig.get_from_yaml()

# 可选：调整 scale_mapping 以获得更稳定的初始值
# algorithm_config.scale_mapping = "exp"  # 备选方案

# 定义场景名称，用于自动选择任务和配置 PINN
scenario_name = "navigation_obs"

# 2. 配置 Actor 模型 (使用 PINN 或 Safe_PINN_PPO)
use_safe_pinn = True  # Set to True to use Safe-PINN (PPO version)

if use_safe_pinn:
    # Configure the Safe-PINN-PPO model (optimized for on-policy algorithms)
    # 
    # Barrier Schedule (prevents late-training instability):
    # - [0, 200 steps]: Warmup from 0 to 0.1
    # - [200, 300 steps]: Hold at 0.1 (plateau)
    # - [300+ steps]: Decay to 0.025 (allow task to dominate for fine-tuning)
    #
    # Key improvements over SafePinn (for MASAC):
    # 1. Uses log-barrier (smoother gradients)
    # 2. Barrier gradient normalization prevents large updates
    # 3. Adaptive barrier scaling based on task gradient magnitude
    # 4. Lower f_max (1.0 vs 10.0) for on-policy stability
    actor_model_config = SafePinnPPOConfig(
        num_cells=[64, 64],
        layer_class=nn.Linear,
        activation_class=nn.Tanh,
        scenario_name=scenario_name,
        r_communication=0.45,
        r_collision=0.2,             # 2x agent_radius (0.1) for proper collision distance
        barrier_epsilon=0.05,        # Larger epsilon for smoother gradients
        f_max=1.0,                   # Lower force saturation for PPO stability
        task_weight=1.0,             # Keep task gradient strong
        barrier_weight=0.05,         # Final barrier weight after decay
        barrier_weight_max=0.1,      # Maximum barrier weight during plateau
        use_log_barrier=True,        # Log barrier has smoother gradient profile
        barrier_warmup_steps=200,    # Gradual barrier activation
        barrier_decay_start=300,     # Start reducing barrier after this
        barrier_decay_rate=0.5,      # Decay to 50% of barrier_weight
    )
else:
    # Configure the standard PINN model
    # Configure the PINN model for the Actor
    # We use the custom PinnConfig we defined
    # Note: The PINN model (LEMURS) requires specific parameters like scenario_name and r_communication
    actor_model_config = PinnConfig(
        num_cells=[64, 64],
        layer_class=nn.Linear,
        activation_class=nn.Tanh,  # Tanh is often used in physics-informed networks
        scenario_name=scenario_name,
        r_communication=0.45,
    )

# 3. 配置 Critic 模型 (使用普通 MLP)
# MAPPO 使用中心化 Critic，所以可以使用更大的网络
critic_model_config = MlpConfig(
    num_cells=[256, 256],
    layer_class=nn.Linear,
    activation_class=nn.ReLU,
)

# 4. 配置实验
experiment_config = ExperimentConfig.get_from_yaml()
experiment_config.evaluation = True

# 速度优化配置
experiment_config.train_device = "cuda:0"      # 训练放 GPU，加速反向传播
experiment_config.sampling_device = "cuda:0"   # 采样设备（如环境支持 GPU）
experiment_config.buffer_device = "cuda:0"     # 缓冲区放 GPU（如内存允许）

# On-policy 算法的并行度和批次配置
experiment_config.on_policy_n_envs_per_worker = 40     # 增加并行环境数
experiment_config.on_policy_collected_frames_per_batch = 6000  # 每批收集的帧数
experiment_config.on_policy_n_minibatch_iters = 45     # minibatch 迭代次数
experiment_config.on_policy_minibatch_size = 400       # minibatch 大小

# 数值稳定性配置 - 更严格的梯度裁剪防止后期不稳定
experiment_config.clip_grad_norm = True
experiment_config.clip_grad_val = 0.5  # 更严格的梯度裁剪（之前是1.0，默认是5.0）

# 降低学习率以提高稳定性
experiment_config.lr = 2e-5  # 降低学习率（之前是3e-5，默认是5e-5）

experiment_config.save_folder = "outputs"
# experiment_config.checkpoint_at_end = True
# experiment_config.checkpoint_interval = 60000  # Save every 10 iterations (6000 * 10)
experiment_config.render = True  # Enable video rendering during evaluation

# 仅使用 wandb logger，视频只会同步到 WandB，不会保存到本地
experiment_config.loggers = ["wandb"]

# 5. 创建实验
experiment = Experiment(
    algorithm_config=algorithm_config,
    task=getattr(VmasTask, scenario_name.upper()).get_from_yaml(),  # 自动根据 scenario_name 选择任务
    seed=0,
    config=experiment_config,
    model_config=actor_model_config,        # Actor 使用 PINN
    critic_model_config=critic_model_config,  # Critic 使用 MLP
)

# 6. 运行实验
if __name__ == "__main__":
    experiment.run()
