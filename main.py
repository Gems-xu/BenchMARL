"""Entry point for running Safe-PINN MAPPO or MASAC experiments."""

import argparse
from typing import Literal

from torch import nn

from gemsmarl.algorithms import MappoConfig, MasacConfig
from gemsmarl.environments import PettingZooTask, VmasTask
from gemsmarl.experiment import Experiment, ExperimentConfig
from gemsmarl.models import MlpConfig, PinnConfig, SafePinnConfig


AlgorithmName = Literal["masac", "mappo"]
EnvironmentName = Literal["vmas", "pettingzoo"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Safe-PINN MARL benchmarks")
    parser.add_argument(
        "--algorithm",
        choices=["masac", "mappo"],
        default="masac",
        help="Algorithm to run (safe-PINN MASAC or MAPPO)",
    )
    parser.add_argument(
        "--env",
        choices=["vmas", "pettingzoo"],
        default="vmas",
        help="Environment backend",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Task name inside the selected environment (e.g., navigation_obs, flocking, simple_spread)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for training/sampling/buffer (e.g., cuda:0 or cpu)",
    )
    parser.add_argument("--save-folder", default="outputs", help="Directory for logs and checkpoints")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable video rendering during evaluation (enabled by default)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging (enabled by default)",
    )
    parser.add_argument(
        "--use-safe-pinn",
        action="store_true",
        help="Force Safe-PINN actor (defaults to True for both algorithms)",
    )
    parser.add_argument(
        "--no-safe-pinn",
        action="store_true",
        help="Force standard PINN actor",
    )
    parser.add_argument(
        "--r-communication",
        type=float,
        default=0.45,
        help="Communication radius for PINN/Safe-PINN",
    )
    parser.add_argument(
        "--r-collision",
        type=float,
        default=0.05,
        help="Collision radius for Safe-PINN barrier",
    )
    parser.add_argument(
        "--barrier-epsilon",
        type=float,
        default=1e-3,
        help="Barrier epsilon for Safe-PINN",
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=10.0,
        help="Force saturation for Safe-PINN",
    )
    # MASAC-specific shortcuts
    parser.add_argument("--off-policy-n-envs-per-worker", type=int, default=40)
    parser.add_argument("--off-policy-train-batch-size", type=int, default=512)
    parser.add_argument("--off-policy-n-optimizer-steps", type=int, default=500)
    # MAPPO-specific shortcuts
    parser.add_argument("--on-policy-n-envs-per-worker", type=int, default=40)
    parser.add_argument("--on-policy-collected-frames-per-batch", type=int, default=6000)
    parser.add_argument("--on-policy-n-minibatch-iters", type=int, default=45)
    parser.add_argument("--on-policy-minibatch-size", type=int, default=400)
    parser.add_argument("--clip-grad-val", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for MAPPO")
    # Evaluation settings
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=None,
        help="Evaluation interval in frames (default: 300000 = 50 batches)",
    )
    parser.add_argument("--evaluation-episodes", type=int, default=10, help="Number of evaluation episodes")
    return parser.parse_args()


def build_algorithm_config(name: AlgorithmName):
    if name == "masac":
        return MasacConfig.get_from_yaml()
    if name == "mappo":
        return MappoConfig.get_from_yaml()
    raise ValueError(f"Unsupported algorithm: {name}")


def build_task(env: EnvironmentName, scenario: str):
    scenario_name = scenario.upper()
    if env == "vmas":
        task_enum = getattr(VmasTask, scenario_name, None)
    elif env == "pettingzoo":
        task_enum = getattr(PettingZooTask, scenario_name, None)
    else:
        raise ValueError(f"Unsupported environment: {env}")
    if task_enum is None:
        raise ValueError(f"Unknown scenario '{scenario}' for environment '{env}'")
    
    task = task_enum.get_from_yaml()
    
    # Safe-PINN/PINN models currently only support VMAS environments
    # because they assume specific observation structure (position in first 2 dims)
    if env == "pettingzoo":
        raise ValueError(
            f"Safe-PINN and PINN models currently only support VMAS environments. "
            f"PettingZoo environments have different observation structures that are not compatible. "
            f"Please use --env vmas with scenarios like 'navigation', 'navigation_obs', 'flocking', 'transport', etc."
        )
    
    return task


def build_actor_model_config(
    scenario: str,
    use_safe_pinn: bool,
    r_communication: float,
    r_collision: float,
    barrier_epsilon: float,
    f_max: float,
):
    if use_safe_pinn:
        return SafePinnConfig(
            num_cells=[64, 64],
            layer_class=nn.Linear,
            activation_class=nn.Tanh,
            scenario_name=scenario,
            r_communication=r_communication,
            r_collision=r_collision,
            barrier_epsilon=barrier_epsilon,
            f_max=f_max,
        )
    return PinnConfig(
        num_cells=[64, 64],
        layer_class=nn.Linear,
        activation_class=nn.Tanh,
        scenario_name=scenario,
        r_communication=r_communication,
    )


def build_critic_model_config():
    return MlpConfig(num_cells=[256, 256], layer_class=nn.Linear, activation_class=nn.ReLU)


def build_experiment_config(args: argparse.Namespace, algorithm: AlgorithmName):
    config = ExperimentConfig.get_from_yaml()
    config.evaluation = True
    config.train_device = args.device
    config.sampling_device = args.device
    config.buffer_device = args.device
    config.save_folder = args.save_folder
    config.render = not args.no_render  # Default to True unless --no-render is specified
    config.loggers = [] if args.no_wandb else ["wandb"]
    
    # Set evaluation interval - use shorter default for quick testing
    if args.evaluation_interval is not None:
        config.evaluation_interval = args.evaluation_interval
    else:
        # Default to 300000 for evaluation every 50 batches (50 * 6000 frames per batch)
        config.evaluation_interval = 300000
    
    config.evaluation_episodes = args.evaluation_episodes
    
    if algorithm == "masac":
        config.off_policy_n_envs_per_worker = args.off_policy_n_envs_per_worker
        config.off_policy_train_batch_size = args.off_policy_train_batch_size
        config.off_policy_n_optimizer_steps = args.off_policy_n_optimizer_steps
    elif algorithm == "mappo":
        config.on_policy_n_envs_per_worker = args.on_policy_n_envs_per_worker
        config.on_policy_collected_frames_per_batch = args.on_policy_collected_frames_per_batch
        config.on_policy_n_minibatch_iters = args.on_policy_n_minibatch_iters
        config.on_policy_minibatch_size = args.on_policy_minibatch_size
        config.clip_grad_norm = True
        config.clip_grad_val = args.clip_grad_val
        config.lr = args.lr
    return config


def main():
    args = parse_args()
    # Default scenarios if not provided
    default_scenarios = {"vmas": "navigation_obs", "pettingzoo": "simple_tag"}
    scenario = args.scenario or default_scenarios[args.env]
    use_safe_pinn = True
    if args.no_safe_pinn:
        use_safe_pinn = False
    elif args.use_safe_pinn:
        use_safe_pinn = True

    algorithm_config = build_algorithm_config(args.algorithm)
    actor_model_config = build_actor_model_config(
        scenario=scenario,
        use_safe_pinn=use_safe_pinn,
        r_communication=args.r_communication,
        r_collision=args.r_collision,
        barrier_epsilon=args.barrier_epsilon,
        f_max=args.f_max,
    )
    critic_model_config = build_critic_model_config()
    experiment_config = build_experiment_config(args, args.algorithm)
    task = build_task(args.env, scenario)

    experiment = Experiment(
        algorithm_config=algorithm_config,
        task=task,
        seed=args.seed,
        config=experiment_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
    )
    experiment.run()


if __name__ == "__main__":
    main()
