import typing
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class NavigationObsScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 4)
        self.collisions = kwargs.pop("collisions", True)

        self.world_spawning_x = kwargs.pop(
            "world_spawning_x", 1
        )  # X-coordinate limit for entities spawning
        self.world_spawning_y = kwargs.pop(
            "world_spawning_y", 1
        )  # Y-coordinate limit for entities spawning
        self.enforce_bounds = kwargs.pop(
            "enforce_bounds", False
        )  # If False, the world is unlimited; else, constrained by world_spawning_x and world_spawning_y.

        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.split_goals = kwargs.pop("split_goals", False)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)

        self.lidar_range = kwargs.pop("lidar_range", 0.35)
        self.agent_radius = kwargs.pop("agent_radius", 0.1)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)

        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 0.01)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
        
        # 添加障碍物参数
        self.n_obstacles = kwargs.pop("n_obstacles", 3)
        self.obstacle_radius = kwargs.pop("obstacle_radius", 0.1)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.005

        if self.enforce_bounds:
            self.x_semidim = self.world_spawning_x
            self.y_semidim = self.world_spawning_y
        else:
            self.x_semidim = None
            self.y_semidim = None

        assert 1 <= self.agents_with_same_goal <= self.n_agents
        if self.agents_with_same_goal > 1:
            assert (
                not self.collisions
            ), "If agents share goals they cannot be collidables"
        # agents_with_same_goal == n_agents: all agent same goal
        # agents_with_same_goal = x: the first x agents share the goal
        # agents_with_same_goal = 1: all independent goals
        if self.split_goals:
            assert (
                self.n_agents % 2 == 0
                and self.agents_with_same_goal == self.n_agents // 2
            ), "Splitting the goals is allowed when the agents are even and half the team has the same goal"

        # Make world
        world = World(
            batch_dim,
            device,
            substeps=2,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
        )

        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )
        # 修改实体过滤器：让Lidar能检测智能体和可碰撞的障碍物
        entity_filter_agents: Callable[[Entity], bool] = lambda e: (
            isinstance(e, Agent) or 
            (isinstance(e, Landmark) and e.collide)  # 包含可碰撞的障碍物
        )

        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=self.n_lidar_rays,
                            max_range=self.lidar_range,
                            entity_filter=entity_filter_agents,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
            )
            # Add constraint attribute for each agent
            agent.constraint_val = torch.zeros(batch_dim, device=device)
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            # Add goals
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=color,
            )
            world.add_landmark(goal)
            agent.goal = goal

        # 添加障碍物
        for i in range(self.n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=self.obstacle_radius),
                color=Color.GRAY,
            )
            world.add_landmark(obstacle)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_spawning_x, self.world_spawning_x),
            (-self.world_spawning_y, self.world_spawning_y),
        )

        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        # 为障碍物分配随机位置
        obstacles = [landmark for landmark in self.world.landmarks if landmark.name.startswith("obstacle_")]
        for obstacle in obstacles:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-self.world_spawning_x, self.world_spawning_x),
                y_bounds=(-self.world_spawning_y, self.world_spawning_y),
            )
            obstacle.set_pos(position.squeeze(1), batch_index=env_index)
            occupied_positions = torch.cat([occupied_positions, position], dim=1)

        goal_poses = []
        for _ in self.world.agents:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-self.world_spawning_x, self.world_spawning_x),
                y_bounds=(-self.world_spawning_y, self.world_spawning_y),
            )
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)

        for i, agent in enumerate(self.world.agents):
            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i

            agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)

            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos,
                        dim=1,
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )

            self.final_rew[self.all_goal_reached] = self.final_reward

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty

            # Calculate constraint values for each policy agent
            # self.constraint()

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def constraint(self):
        """
        Calculate constraint values for all agents based on minimum distances
        Return shape: [num_agents] containing each agent's constraint value
        Each agent's constraint = normalized minimum distance to all other entities
        
        Constraint value mapping:
        +1.0: Very safe (distance > safe_threshold)  
        0.0: Neutral (distance = safe_threshold)
        -1.0: Very dangerous (distance <= safe_threshold)
        """
        # Define safety thresholds
        safe_threshold = self.agent_radius * 1.5  
        scalar = 20.0  # Scaling factor for tanh or clamp input
     
        for current_agent in self.world.policy_agents:
            min_distances_all_envs = []
                
            # Calculate distances to other policy agents in this environment
            for other_agent in self.world.policy_agents:
                if other_agent != current_agent:
                    # Get distance for this specific environment
                    all_distances = self.world.get_distance(current_agent, other_agent)
                    min_distances_all_envs.append(all_distances)
                
            # Calculate distances to obstacles in this environment
            for obstacle in self.world.landmarks:
                if obstacle.collide:  # Only check collidable obstacles
                    all_distances = self.world.get_distance(current_agent, obstacle)
                    min_distances_all_envs.append(all_distances)
                
            # Find minimum distance for this agent in this environment
            min_distance = torch.min(torch.stack(min_distances_all_envs, dim=0), 
                                     dim=0).values  # [B,]
            
            # Apply tanh or clamp to ensure smooth mapping to [-1, 1]
            current_agent.constraint_val = torch.clamp((min_distance-safe_threshold)*scalar, 
                                                       min=-1.0, max=1.0)  # [B,]
        
    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    def observation(self, agent: Agent):
        goal_poses = []
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ]
            + goal_poses
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.collisions
                else []
            ),
            dim=-1,
        )

    def done(self):
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )
                < agent.shape.radius
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(-1)

    def info(self, agent: Agent):
        # Find the constraint for the current agent
        if agent in self.world.policy_agents:
            agent_constraint = agent.constraint_val
        else:
            # For target agent, use zero constraint
            agent_constraint = torch.zeros(self.world.batch_dim, device=self.world.device)
        
        return {"constraints": agent_constraint}

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, clf_epsilon=0.2, clf_slack=100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf_epsilon = clf_epsilon  # Exponential CLF convergence rate
        self.clf_slack = clf_slack  # weights on CLF-QP slack variable

    def compute_action(self, observation: Tensor, u_range: Tensor) -> Tensor:
        """
        QP inputs:
        These values need to computed apriri based on observation before passing into QP

        V: Lyapunov function value
        lfV: Lie derivative of Lyapunov function
        lgV: Lie derivative of Lyapunov function
        CLF_slack: CLF constraint slack variable

        QP outputs:
        u: action
        CLF_slack: CLF constraint slack variable, 0 if CLF constraint is satisfied
        """
        # Install it with: pip install cvxpylayers
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer

        self.n_env = observation.shape[0]
        self.device = observation.device
        agent_pos = observation[:, :2]
        agent_vel = observation[:, 2:4]
        goal_pos = (-1.0) * (observation[:, 4:6] - agent_pos)

        # Pre-compute tensors for the CLF and CBF constraints,
        # Lyapunov Function from: https://arxiv.org/pdf/1903.03692.pdf

        # Laypunov function
        V_value = (
            (agent_pos[:, X] - goal_pos[:, X]) ** 2
            + 0.5 * (agent_pos[:, X] - goal_pos[:, X]) * agent_vel[:, X]
            + agent_vel[:, X] ** 2
            + (agent_pos[:, Y] - goal_pos[:, Y]) ** 2
            + 0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) * agent_vel[:, Y]
            + agent_vel[:, Y] ** 2
        )

        LfV_val = (2 * (agent_pos[:, X] - goal_pos[:, X]) + agent_vel[:, X]) * (
            agent_vel[:, X]
        ) + (2 * (agent_pos[:, Y] - goal_pos[:, Y]) + agent_vel[:, Y]) * (
            agent_vel[:, Y]
        )
        LgV_vals = torch.stack(
            [
                0.5 * (agent_pos[:, X] - goal_pos[:, X]) + 2 * agent_vel[:, X],
                0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) + 2 * agent_vel[:, Y],
            ],
            dim=1,
        )
        # Define Quadratic Program (QP) based controller
        u = cp.Variable(2)
        V_param = cp.Parameter(1)  # Lyapunov Function: V(x): x -> R, dim: (1,1)
        lfV_param = cp.Parameter(1)
        lgV_params = cp.Parameter(
            2
        )  # Lie derivative of Lyapunov Function, dim: (1, action_dim)
        clf_slack = cp.Variable(1)  # CLF constraint slack variable, dim: (1,1)

        constraints = []

        # QP Cost F = u^T @ u + clf_slack**2
        qp_objective = cp.Minimize(cp.sum_squares(u) + self.clf_slack * clf_slack**2)

        # control bounds between u_range
        constraints += [u <= u_range]
        constraints += [u >= -u_range]
        # CLF constraint
        constraints += [
            lfV_param + lgV_params @ u + self.clf_epsilon * V_param + clf_slack <= 0
        ]

        QP_problem = cp.Problem(qp_objective, constraints)

        # Initialize CVXPY layers
        QP_controller = CvxpyLayer(
            QP_problem,
            parameters=[V_param, lfV_param, lgV_params],
            variables=[u],
        )

        # Solve QP
        CVXpylayer_parameters = [
            V_value.unsqueeze(1),
            LfV_val.unsqueeze(1),
            LgV_vals,
        ]
        action = QP_controller(*CVXpylayer_parameters, solver_args={"max_iters": 500})[
            0
        ]

        return action


if __name__ == "__main__":
    import vmas
    scenario = NavigationObsScenario()
    env = vmas.make_env(
        scenario=scenario,
        num_envs=1,
        device="cuda:0",
        continuous_actions=True
    )
    obs = env.reset()
    for step in range(500):
        # Generate actions in correct range [-0.5, 0.5]
        actions = []
        for i in range(len(env.action_space)):
            action = torch.rand(env.num_envs, env.action_space[i].shape[0]) - 0.5
            actions.append(action)
            
        obs, rews, dones, infos = env.step(actions)
        env.render(mode="rgb_array")
        
        # Print constraint values every 50 steps
        # if step % 50 == 0:
        #     if hasattr(scenario, 'constraint'):
        #         agent_constraints = scenario.constraint()
        #         print(f"Step {step:3d}: Agent constraints = {[f'{c:.4f}' for c in agent_constraints]}")
        
        if dones.any():
            print(f"Episode finished at step {step}")
            break
    
    print("Simulation completed!")