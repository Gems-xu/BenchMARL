import typing
from dataclasses import dataclass, MISSING
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    collisions: bool = MISSING
    agents_with_same_goal: int = MISSING
    observe_all_goals: bool = MISSING
    shared_rew: bool = MISSING
    split_goals: bool = MISSING
    lidar_range: float = MISSING
    agent_radius: float = MISSING
    n_obstacles: int = MISSING
    obstacle_radius: float = MISSING
    max_linear_velocity: float = MISSING
    max_angular_velocity: float = MISSING


class NavigationObsUnicycleScenario(BaseScenario):
    """Navigation scenario with obstacles using Unicycle dynamics model.
    
    Unicycle dynamics:
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = omega
    
    Action space: [v, omega] where:
        v: linear velocity (forward/backward)
        omega: angular velocity (turning rate)
    
    State: [x, y, vx, vy, theta] where:
        (x, y): position
        (vx, vy): linear velocities computed from v and theta
        theta: heading angle
    """
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 4)
        self.collisions = kwargs.pop("collisions", True)

        self.world_spawning_x = kwargs.pop("world_spawning_x", 1)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1)
        self.enforce_bounds = kwargs.pop("enforce_bounds", False)

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
        
        # Obstacle parameters
        self.n_obstacles = kwargs.pop("n_obstacles", 3)
        self.obstacle_radius = kwargs.pop("obstacle_radius", 0.1)
        
        # Unicycle dynamics parameters
        self.max_linear_velocity = kwargs.pop("max_linear_velocity", 0.8)
        self.max_angular_velocity = kwargs.pop("max_angular_velocity", 2.0)
        
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
            assert not self.collisions, "If agents share goals they cannot be collidables"
        if self.split_goals:
            assert (
                self.n_agents % 2 == 0
                and self.agents_with_same_goal == self.n_agents // 2
            ), "Splitting the goals is allowed when the agents are even and half the team has the same goal"

        # Make world with DiffDrive dynamics
        world = World(
            batch_dim,
            device,
            substeps=2,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
        )
        
        # Store reference for DiffDrive dynamics
        self._world_ref = world

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
        
        # Entity filter for Lidar: detect agents and collidable obstacles
        entity_filter_agents: Callable[[Entity], bool] = lambda e: (
            isinstance(e, Agent) or 
            (isinstance(e, Landmark) and e.collide)
        )

        # Add agents with unicycle dynamics
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

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
                mass=1.0,
                max_speed=None,  # Handled by unicycle model
                # Use DiffDrive dynamics for unicycle kinematics
                dynamics=DiffDrive(world=world, integration="rk4"),
                # Enable rotation for unicycle
                rotatable=True,
            )
            
            # Other agent attributes (no need for manual orientation tracking)
            agent.constraint_val = torch.zeros(batch_dim, device=device)
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            # Add goals with larger radius for easier reaching
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=color,
                shape=Sphere(radius=self.agent_radius),  # Same size as agent for visibility
            )
            world.add_landmark(goal)
            agent.goal = goal

        # Add obstacles
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

        # Initialize random orientations and zero velocity for unicycle agents
        for agent in self.world.agents:
            if env_index is None:
                # Random orientation between -pi and pi (use agent.state.rot)
                random_rot = (torch.rand(self.world.batch_dim, 1, device=self.world.device) * 2 * torch.pi - torch.pi)
                agent.set_rot(random_rot, batch_index=None)
                agent.state.vel[:] = 0  # Start with zero velocity
                agent.state.ang_vel[:] = 0
            else:
                random_rot = torch.rand(1, device=self.world.device).item() * 2 * torch.pi - torch.pi
                agent.set_rot(torch.tensor([[random_rot]], device=self.world.device), batch_index=env_index)
                agent.state.vel[env_index] = 0
                agent.state.ang_vel[env_index] = 0

        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        # Spawn obstacles randomly
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

        # Spawn goals randomly
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

    def process_action(self, agent: Agent):
        """
        Process actions for unicycle dynamics using DiffDrive.
        Action space: normalized [-1, 1]^2 representing [v, omega]
        - action[0]: normalized linear velocity (mapped to [-max_linear_velocity, max_linear_velocity])
        - action[1]: normalized angular velocity (mapped to [-max_angular_velocity, max_angular_velocity])
        
        The DiffDrive dynamics will handle the unicycle kinematics:
            dx/dt = v * cos(theta)
            dy/dt = v * sin(theta)
            dtheta/dt = omega
        """
        # Scale normalized actions [-1, 1] to actual velocity ranges
        # DiffDrive expects: action[0] = forward velocity, action[1] = angular velocity
        agent.action.u[:, 0] = agent.action.u[:, 0] * self.max_linear_velocity
        agent.action.u[:, 1] = agent.action.u[:, 1] * self.max_angular_velocity
        
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

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def constraint(self):
        """Calculate constraint values for all agents based on minimum distances."""
        safe_threshold = self.agent_radius * 1.5  
        scalar = 20.0
     
        for current_agent in self.world.policy_agents:
            min_distances_all_envs = []
                
            # Calculate distances to other policy agents
            for other_agent in self.world.policy_agents:
                if other_agent != current_agent:
                    all_distances = self.world.get_distance(current_agent, other_agent)
                    min_distances_all_envs.append(all_distances)
                
            # Calculate distances to obstacles
            for obstacle in self.world.landmarks:
                if obstacle.collide:
                    all_distances = self.world.get_distance(current_agent, obstacle)
                    min_distances_all_envs.append(all_distances)
                
            # Find minimum distance
            if len(min_distances_all_envs) > 0:
                min_distance = torch.min(torch.stack(min_distances_all_envs, dim=0), 
                                         dim=0).values
                # Map to constraint value in [-1, 1]
                current_agent.constraint_val = torch.clamp((min_distance-safe_threshold)*scalar, 
                                                           min=-1.0, max=1.0)
            else:
                current_agent.constraint_val = torch.ones(self.world.batch_dim, device=self.world.device)
        
    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        
        # Add heading reward: encourage agent to face the goal
        theta = agent.state.rot.squeeze(-1)
        goal_dir = agent.goal.state.pos - agent.state.pos
        goal_angle = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])
        angle_error = torch.abs(torch.atan2(
            torch.sin(goal_angle - theta), 
            torch.cos(goal_angle - theta)
        ))
        # Heading bonus: max 0.1 when perfectly aligned, 0 when perpendicular
        heading_bonus = 0.05 * (1.0 - angle_error / torch.pi) * (1.0 - agent.on_goal.float())
        agent.pos_rew = agent.pos_rew + heading_bonus
        
        return agent.pos_rew

    def observation(self, agent: Agent):
        """
        Observation includes:
        - agent position (2D)
        - agent velocity (2D)  
        - agent orientation cos(theta), sin(theta) (2D) - for unicycle
        - goal direction in agent's local frame cos(goal_angle - theta), sin(goal_angle - theta) (2D)
        - distance to goal (1D)
        - goal position relative to agent (2D or more if observe_all_goals)
        - lidar readings (if collisions enabled)
        
        The local goal direction helps the policy directly learn what action to take:
        - If goal is in front (local_goal_dir ≈ [1, 0]): go forward
        - If goal is to the left (local_goal_dir ≈ [0, 1]): turn left first
        - If goal is behind (local_goal_dir ≈ [-1, 0]): turn around
        """
        goal_poses = []
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)
        
        # Agent orientation as cos(theta), sin(theta)
        theta = agent.state.rot.squeeze(-1)
        orientation = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        
        # Goal direction in world frame
        goal_rel = agent.goal.state.pos - agent.state.pos
        goal_angle = torch.atan2(goal_rel[:, 1], goal_rel[:, 0])
        
        # Goal direction in agent's local frame (very useful for unicycle control)
        local_goal_angle = goal_angle - theta
        local_goal_dir = torch.stack([
            torch.cos(local_goal_angle), 
            torch.sin(local_goal_angle)
        ], dim=-1)
        
        # Distance to goal (normalized by world size)
        dist_to_goal = torch.linalg.vector_norm(goal_rel, dim=-1, keepdim=True) / (2 * self.world_spawning_x)
        
        return torch.cat(
            [
                agent.state.pos,       # (2,) position
                agent.state.vel,       # (2,) velocity
                orientation,           # (2,) cos(theta), sin(theta)
                local_goal_dir,        # (2,) goal direction in local frame
                dist_to_goal,          # (1,) normalized distance to goal
            ]
            + goal_poses            # (2,) or more - goal relative position
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]  # (n_rays,)
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
        if agent in self.world.policy_agents:
            agent_constraint = agent.constraint_val
        else:
            agent_constraint = torch.zeros(self.world.batch_dim, device=self.world.device)
        
        return {"constraints": agent_constraint}

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Draw communication lines
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
        
        # Draw orientation arrows for unicycle agents
        for agent in self.world.agents:
            # Use agent.state.rot instead of agent.orientation
            theta = agent.state.rot[env_index, 0]
            pos = agent.state.pos[env_index]
            
            # Arrow length proportional to agent radius
            arrow_length = self.agent_radius * 1.5
            end_x = pos[0] + arrow_length * torch.cos(theta)
            end_y = pos[1] + arrow_length * torch.sin(theta)
            
            end_pos = torch.tensor([end_x.item(), end_y.item()], device=self.world.device)
            
            arrow = rendering.Line(
                pos,
                end_pos,
                width=2,
            )
            arrow.set_color(*Color.BLACK.value)
            geoms.append(arrow)

        return geoms
    
    def post_step(self):
        """
        Called after physics step.
        DiffDrive handles orientation updates internally, so nothing needed here.
        """
        pass


class HeuristicPolicy(BaseHeuristicPolicy):
    """Simple proportional controller for unicycle model."""
    
    def __init__(self, k_v=1.0, k_omega=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_v = k_v  # Linear velocity gain
        self.k_omega = k_omega  # Angular velocity gain

    def compute_action(self, observation: Tensor, u_range: Tensor) -> Tensor:
        """
        Simple proportional controller for unicycle model.
        Uses local goal direction from observation for direct control.
        
        Observation format:
            [pos(2), vel(2), orientation(2), local_goal_dir(2), dist_to_goal(1), goal_rel(2), ...]
        
        Args:
            observation: observation tensor
            u_range: [max_v, max_omega]
        """
        self.n_env = observation.shape[0]
        self.device = observation.device
        
        # Local goal direction (already in agent's frame) at index 6:8
        local_goal_cos = observation[:, 6]  # cos(goal_angle - theta)
        local_goal_sin = observation[:, 7]  # sin(goal_angle - theta)
        
        # Normalized distance to goal at index 8
        dist_normalized = observation[:, 8]
        
        # Angular error is simply atan2(sin, cos) of local goal direction
        theta_error = torch.atan2(local_goal_sin, local_goal_cos)
        
        # Proportional control
        # Linear velocity: go forward when facing goal, slow when need to turn
        v = self.k_v * dist_normalized * torch.cos(theta_error)
        omega = self.k_omega * theta_error
        
        # Clamp to action limits
        v = torch.clamp(v, -u_range[0], u_range[0])
        omega = torch.clamp(omega, -u_range[1], u_range[1])
        
        action = torch.stack([v, omega], dim=-1)
        
        return action


if __name__ == "__main__":
    import vmas
    scenario = NavigationObsUnicycleScenario()
    env = vmas.make_env(
        scenario=scenario,
        num_envs=4,
        device="cpu",
        continuous_actions=True,
        n_agents=4,
        n_obstacles=3,
    )
    obs = env.reset()
    
    print("=" * 60)
    print("Navigation with Obstacles - Unicycle Dynamics Test")
    print("=" * 60)
    print(f"Number of environments: {env.num_envs}")
    print(f"Number of agents: {len(env.agents)}")
    print(f"Observation shape: {obs[0].shape}")
    print(f"Action space: {env.action_space[0]}")
    print(f"Observation breakdown:")
    print(f"  - Position: [0:2]")
    print(f"  - Velocity: [2:4]")
    print(f"  - Orientation (cos, sin): [4:6]")
    print(f"  - Local goal direction (cos, sin): [6:8]")
    print(f"  - Distance to goal (normalized): [8:9]")
    print(f"  - Goal relative position: [9:11]")
    if scenario.collisions:
        print(f"  - Lidar readings: [11:{11+scenario.n_lidar_rays}]")
    print("=" * 60)
    
    for step in range(200):
        # Generate random actions [v, omega] within proper range
        actions = []
        for i in range(len(env.action_space)):
            # Action space is [-1, 1]^2, will be scaled by max velocities in process_action
            # But VMAS expects actions in [-1, 1], so we normalize:
            # v in [- max_linear_velocity, max_linear_velocity] -> [-1, 1]
            # omega in [-max_angular_velocity, max_angular_velocity] -> [-1, 1]
            v_normalized = torch.rand(env.num_envs, 1) * 2.0 - 1.0  # [-1, 1]
            omega_normalized = torch.rand(env.num_envs, 1) * 2.0 - 1.0  # [-1, 1]
            action = torch.cat([v_normalized, omega_normalized], dim=-1)
            actions.append(action)
            
        obs, rews, dones, infos = env.step(actions)
        
        # Print state info every 50 steps
        if step % 50 == 0:
            agent_pos = obs[0][0, :2]
            agent_vel = obs[0][0, 2:4]
            cos_theta = obs[0][0, 4]
            sin_theta = obs[0][0, 5]
            theta = torch.atan2(sin_theta, cos_theta).item()
            print(f"Step {step:3d}: pos=({agent_pos[0]:.3f},{agent_pos[1]:.3f}), " +
                  f"vel=({agent_vel[0]:.3f},{agent_vel[1]:.3f}), theta={theta:6.3f}rad ({theta*180/3.14159:.1f}°)")
        
        if dones.any():
            print(f"\nEpisode finished at step {step}")
            print(f"Environments done: {dones.sum().item()}/{env.num_envs}")
            break
    
    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    print("=" * 60)
