#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
from gemsmarl.algorithms import MasacConfig
from gemsmarl.environments import VmasTask
from gemsmarl.experiment import Experiment, ExperimentConfig
from gemsmarl.models import MlpConfig, PinnConfig, SafePinnConfig

# 1. 配置算法 (MASAC)
algorithm_config = MasacConfig.get_from_yaml()

# 定义场景名称，用于自动选择任务和配置 PINN
# 可选: "navigation" (无障碍物) 或 "navigation_obs" (带圆形障碍物)
scenario_name = "navigation_obs"

# 2. 配置 Actor 模型 (使用 PINN 或 Safe_PINN)
use_safe_pinn = True  # Set to True to use Safe-PINN

if use_safe_pinn:
    # Configure the Safe-PINN model
    actor_model_config = SafePinnConfig(
        num_cells=[64, 64],
        layer_class=nn.Linear,
        activation_class=nn.Tanh,
        scenario_name=scenario_name,
        r_communication=0.45,
        r_collision=0.05,  # Collision radius for barrier
        barrier_epsilon=1e-3,
        f_max=10.0,        # Force saturation
    )
else:
    # Configure the standard PINN model
    actor_model_config = PinnConfig(
        num_cells=[64, 64],
        layer_class=nn.Linear,
        activation_class=nn.Tanh,
        scenario_name=scenario_name,
        r_communication=0.45,
    )

# 3. 配置 Critic 模型 (使用普通 MLP)
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

# 提高并行度和批次大小
experiment_config.off_policy_n_envs_per_worker = 40   # 增加并行环境数
experiment_config.off_policy_train_batch_size = 512   # 增大批次，提高 GPU 利用率
experiment_config.off_policy_n_optimizer_steps = 500  # 减少优化步数（如果采样是瓶颈）

experiment_config.save_folder = "outputs"
# experiment_config.checkpoint_at_end = True
# experiment_config.checkpoint_interval = 60000  # Save every 10 iterations (6000 * 10)
experiment_config.render = True  # Enable video rendering during evaluation

# 仅使用 wandb logger，视频只会同步到 WandB，不会保存到本地
experiment_config.loggers = ["wandb"]

# 5. 创建实验
experiment = Experiment(
    algorithm_config=algorithm_config,
    task=getattr(VmasTask, scenario_name.upper()).get_from_yaml(), # 自动根据 scenario_name 选择任务
    seed=0,
    config=experiment_config,
    model_config=actor_model_config,       # Actor 使用 PINN
    critic_model_config=critic_model_config, # Critic 使用 MLP
)

# 6. 运行实验
if __name__ == "__main__":
    experiment.run()
