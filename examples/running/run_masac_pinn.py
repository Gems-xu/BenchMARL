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
from gemsmarl.models import MlpConfig, PinnConfig

# 1. 配置算法 (MASAC)
algorithm_config = MasacConfig.get_from_yaml()

# 定义场景名称，用于自动选择任务和配置 PINN
scenario_name = "navigation"

# 2. 配置 Actor 模型 (使用 PINN)
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
critic_model_config = MlpConfig(
    num_cells=[256, 256],
    layer_class=nn.Linear,
    activation_class=nn.ReLU,
)

# 4. 配置实验
experiment_config = ExperimentConfig.get_from_yaml()
experiment_config.evaluation = True
experiment_config.train_device = "cuda:1"
experiment_config.sampling_device = "cuda:1"
experiment_config.save_folder = "outputs"
experiment_config.checkpoint_at_end = True
experiment_config.checkpoint_interval = 60000  # Save every 10 iterations (6000 * 10)
experiment_config.render = True  # Enable video rendering during evaluation

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
