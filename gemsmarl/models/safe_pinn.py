#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

import torch
from tensordict import TensorDictBase
from torch import nn
from torch.autograd import Variable

from gemsmarl.models.common import Model, ModelConfig
from gemsmarl.models.pinn import MLP, MLP2, Attention_LEMURS, Att_R, Att_J, Att_H

class BarrierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        
        # Encoders for state
        self.mlp_shared = MLP(input_dim, [hidden_dim, hidden_dim]).to(device)
        
        # MLP for k_ij
        # Input: z_i (hidden) + z_j (hidden)
        self.mlp_k = MLP(2 * hidden_dim, [hidden_dim, 1]).to(device)
        self.softplus = nn.Softplus()
        
        # Learnable smoothness parameter
        self.log_smoothness = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, x, adj):
        # x: (batch, n_agents, input_dim)
        # adj: (batch, n_agents, n_agents) - adjacency/laplacian mask
        
        b, n, d = x.shape
        
        # Encode states
        z = self.mlp_shared(x) # (b, n, h)
        
        # Prepare pairs
        z_i = z.unsqueeze(2).expand(-1, -1, n, -1) # (b, n, n, h)
        z_j = z.unsqueeze(1).expand(-1, n, -1, -1) # (b, n, n, h)
        
        z_combined = torch.cat([z_i, z_j], dim=-1) # (b, n, n, 2h)
        
        k_ij_raw = self.mlp_k(z_combined).squeeze(-1) # (b, n, n)
        
        # Clamp raw values to prevent extreme outputs
        k_ij_raw = torch.clamp(k_ij_raw, min=-5.0, max=2.0)
        
        # Apply softplus with learnable smoothness scaling
        smoothness = self.softplus(self.log_smoothness) + 0.1
        k_ij = self.softplus(k_ij_raw) * smoothness
        
        # Clamp k_ij to prevent numerical instability - VERY LOW for MASAC
        k_ij = torch.clamp(k_ij, min=0.0, max=1.0)
        
        # Mask with adjacency/interaction range
        k_ij = k_ij * adj
        
        return k_ij

class SafePinn(Model):
    """Safe Physics-Informed Neural Network (Safe-PINN) model based on Barrier Hamiltonian.
    
    Optimized for off-policy algorithms (MASAC) with:
    - Softer barrier parameters for stable Q-learning
    - Progressive barrier warmup to allow initial exploration
    - Balanced task/barrier gradient weighting
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.num_feature_dims = kwargs.pop("num_feature_dims", 1)
        self.scenario_name = kwargs.pop("scenario_name", "grassland_vmas")
        self.r_communication = kwargs.pop("r_communication", 0.45)
        self.r_collision = kwargs.pop("r_collision", 0.18)  # Slightly conservative collision distance
        self.barrier_epsilon = kwargs.pop("barrier_epsilon", 0.08)  # Larger epsilon for smoother gradients
        self.f_max = kwargs.pop("f_max", 1.0)  # Lower force saturation for stability
        
        # Lidar-based obstacle avoidance parameters
        self.use_lidar_barrier = kwargs.pop("use_lidar_barrier", True)  # Use lidar for obstacle avoidance
        self.lidar_start_idx = kwargs.pop("lidar_start_idx", 6)  # Lidar data starts at index 6
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)  # Number of lidar rays
        self.lidar_max_range = kwargs.pop("lidar_max_range", 0.35)  # Max lidar range
        self.obstacle_barrier_weight = kwargs.pop("obstacle_barrier_weight", 0.01)  # Very low weight
        
        # Off-policy specific parameters - LOW barrier weights for MASAC
        self.task_weight = kwargs.pop("task_weight", 1.0)
        self.barrier_weight = kwargs.pop("barrier_weight", 0.01)  # Low: prioritize goal reaching
        self.barrier_weight_max = kwargs.pop("barrier_weight_max", 0.02)  # Low max weight
        self.use_log_barrier = kwargs.pop("use_log_barrier", True)  # Smoother barrier
        self.barrier_warmup_steps = kwargs.pop("barrier_warmup_steps", 500)  # Moderate warmup
        self.barrier_decay_start = kwargs.pop("barrier_decay_start", 800)  # Moderate decay start
        self.barrier_decay_rate = kwargs.pop("barrier_decay_rate", 0.3)  # Decay to reduce barrier
        
        # Goal attraction strength - same as PPO for stability
        self.goal_attraction_strength = kwargs.pop("goal_attraction_strength", 10.0)
        
        # Multi-agent scaling
        self.neighbor_normalized_barrier = kwargs.pop("neighbor_normalized_barrier", True)
        self.auto_scale_by_agents = kwargs.pop("auto_scale_by_agents", True)
        
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        self.input_features = sum(
            [
                torch.prod(torch.tensor(spec.shape[-self.num_feature_dims :])).item()
                for spec in self.input_spec.values(True, True)
            ]
        )
        # Output features should be 2 * action_dim (mean and log_std)
        self.output_features = self.output_leaf_spec.shape[-1]
        self.action_dim_per_agent = int(self.output_features // 2)
        self.observation_dim_per_agent = int(self.input_features)

        self.drag = 0.25
        self.log_std_min = -5
        self.log_std_max = 2
        
        # Multi-agent scaling: reduce barrier influence for more agents
        if self.auto_scale_by_agents and self.n_agents > 4:
            reference_pairs = 6.0  # 4 agents reference
            actual_pairs = self.n_agents * (self.n_agents - 1) / 2.0
            scale_factor = reference_pairs / actual_pairs
            self.barrier_weight = self.barrier_weight * scale_factor
            self.barrier_weight_max = self.barrier_weight_max * scale_factor
        
        # Track training steps for barrier warmup
        self.register_buffer('_training_steps', torch.tensor(0, dtype=torch.long))

        # Dynamics Heads
        self.R_mean = Att_R(self.observation_dim_per_agent, 16, 8, self.observation_dim_per_agent, self.scenario_name, self.device).to(self.device)
        self.J_mean = Att_J(self.observation_dim_per_agent, 16, 8, self.observation_dim_per_agent, self.scenario_name, self.device).to(self.device)
        
        # Task Potential Head (H_task)
        self.H_task = Att_H(self.observation_dim_per_agent, 25, 8, self.observation_dim_per_agent, self.device).to(self.device)
        
        # Barrier Potential Head (H_barrier)
        self.H_barrier_head = BarrierHead(self.observation_dim_per_agent, 16, self.device).to(self.device)

        self.std_net = Attention_LEMURS(self.observation_dim_per_agent + self.action_dim_per_agent,
                                 self.action_dim_per_agent,
                                 self.observation_dim_per_agent,
                                 self.n_agents,
                                 self.device).to(self.device)
        
        # Pre-compute system matrices
        self.F_sys_pinv = torch.cat((torch.zeros(self.action_dim_per_agent * self.n_agents,
                                                 self.action_dim_per_agent * self.n_agents,
                                                 device=self.device),
                                 torch.eye(self.action_dim_per_agent * self.n_agents, device=self.device)), dim=1)

        self.J_sys = torch.cat((torch.cat((torch.zeros(self.action_dim_per_agent * self.n_agents,
                                                       self.action_dim_per_agent * self.n_agents,
                                                       device=self.device),
                                 torch.eye(self.action_dim_per_agent * self.n_agents, device=self.device)), dim=1),
                                torch.cat((-torch.eye(self.action_dim_per_agent * self.n_agents, device=self.device),
                                torch.zeros(self.action_dim_per_agent * self.n_agents,
                                            self.action_dim_per_agent * self.n_agents, device=self.device)), dim=1)
                                ), dim=0)
        self.R_sys = torch.cat((torch.cat((torch.zeros(self.action_dim_per_agent * self.n_agents,
                                                       self.action_dim_per_agent * self.n_agents,
                                                       device=self.device),
                                 torch.zeros(self.action_dim_per_agent * self.n_agents,
                                             self.action_dim_per_agent * self.n_agents,
                                             device=self.device)), dim=1),
                                torch.cat((torch.zeros(self.action_dim_per_agent * self.n_agents,
                                                       self.action_dim_per_agent * self.n_agents,
                                                       device=self.device),
                                self.drag*torch.eye(self.action_dim_per_agent * self.n_agents, device=self.device)), dim=1)
                                ), dim=0)

    def laplacian(self, q_agents):
        # Optimized pairwise distance calculation
        Q = torch.cdist(q_agents, q_agents, p=2)
        L = Q.le(self.r_communication).float()
        L = L * torch.sigmoid(-(2.0) * (Q - self.r_communication))
        return L

    def _perform_checks(self):
        super()._perform_checks()
        if not self.input_has_agent_dim:
             raise ValueError("SafePINN model requires input with agent dimension")

    def _get_barrier_weight(self):
        """Progressive barrier activation with warmup and decay.
        
        Schedule (optimized for off-policy with larger replay buffer):
        1. [0, warmup_steps]: Cosine warmup from 0 to barrier_weight_max
        2. [warmup_steps, decay_start]: Hold at barrier_weight_max
        3. [decay_start, inf]: Gradual decay to barrier_weight * decay_rate
        """
        steps = self._training_steps.float()
        
        if steps < self.barrier_warmup_steps:
            # Warmup phase: cosine warmup for smoother gradient changes
            progress = steps / max(1.0, self.barrier_warmup_steps)
            cosine_progress = 0.5 * (1.0 - torch.cos(progress * 3.14159))
            return self.barrier_weight_max * cosine_progress
        elif steps < self.barrier_decay_start:
            # Plateau phase: hold at max
            return self.barrier_weight_max
        else:
            # Decay phase: gradual cosine decay
            decay_duration = 1000.0  # Longer decay for off-policy
            decay_steps = steps - self.barrier_decay_start
            decay_progress = torch.clamp(decay_steps / decay_duration, max=1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(decay_progress * 3.14159))
            target_weight = self.barrier_weight * self.barrier_decay_rate
            return target_weight + (self.barrier_weight_max - target_weight) * cosine_decay

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Increment training step counter if training
        if self.training:
            self._training_steps += 1
            
        # Gather in_key and flatten the last self.num_feature_dims dimensions
        # Input shape: (batch, n_agents, obs_dim)
        x = torch.cat(
            [
                torch.flatten(tensordict.get(in_key), start_dim=-self.num_feature_dims)
                for in_key in self.in_keys
            ],
            dim=-1,
        )
        
        batch_size = x.shape[0]
        
        # Use pre-computed system matrices
        # Expand them to match batch size
        F_sys_pinv = self.F_sys_pinv.unsqueeze(0).expand(batch_size, -1, -1)
        J_sys = self.J_sys.unsqueeze(0).expand(batch_size, -1, -1)
        R_sys = self.R_sys.unsqueeze(0).expand(batch_size, -1, -1)

        state = x
        state_h_mean = torch.clone(state).reshape(-1, self.observation_dim_per_agent)

        # Laplacian
        # Assuming first 2 dims are position
        q_pos = state[:, :, 0:2]
        laplacian_base = self.laplacian(q_pos)
        # Optimized: replace torch.kron with repeat_interleave
        laplacian = laplacian_base.unsqueeze(-1).repeat_interleave(self.observation_dim_per_agent, dim=-1)
        laplacian = laplacian.reshape(-1, self.n_agents, self.observation_dim_per_agent)

        # Reshape and normalize inputs - use expand instead of repeat where possible
        state_expanded = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1).reshape(-1, self.n_agents, self.observation_dim_per_agent)
        state_masked = laplacian * state_expanded

        # Use detach for std_input since it doesn't need gradients from state computation
        std_input = state_masked.detach().clone()

        R_mean = self.R_mean.forward(state_masked.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        J_mean = self.J_mean.forward(state_masked.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        
        with torch.enable_grad():
            state_h_mean = Variable(state_h_mean.data, requires_grad=True)
            
            # Reconstruct batch structure for H calculation
            state_batch = state_h_mean.reshape(batch_size, self.n_agents, -1)
            
            # 1. H_task (Attractive)
            # Att_H expects (batch*n_agents, obs_dim) if we look at pinn.py, but let's check Att_H.forward
            # Att_H.forward(x, na) -> x is (batch*na, dim).
            # It returns (batch*na, 1).
            H_task_val = self.H_task.forward(state_h_mean.to(torch.float32), self.n_agents)
            H_task_sum = H_task_val.sum()
            
            # 2. H_barrier (Repulsive)
            # Need positions q from state_batch
            q_batch = state_batch[:, :, 0:2] # (b, n, 2)
            
            # Calculate pairwise distances
            # (b, n, 1, 2) - (b, 1, n, 2) -> (b, n, n, 2)
            diff = q_batch.unsqueeze(2) - q_batch.unsqueeze(1)
            dist_sq = torch.sum(diff**2, dim=-1) # (b, n, n)
            dist = torch.sqrt(dist_sq + 1e-6)
            
            # Get stiffness k_ij
            k_ij = self.H_barrier_head(state_batch, laplacian_base) # (b, n, n)
            
            # Calculate Barrier Potential using softplus-based log barrier (smoother gradients)
            gap = dist - self.r_collision
            # Ensure gap is positive with larger minimum for stability
            safe_gap = torch.clamp(gap, min=max(self.barrier_epsilon, 0.02))
            
            # Softplus-based log barrier (smoother than 1/x barrier)
            ratio = safe_gap / (self.r_collision + self.barrier_epsilon + 1e-6)
            ratio = torch.clamp(ratio, min=0.01, max=100.0)
            log_term = torch.log(ratio + 1e-6)
            softplus_input = torch.clamp(-log_term, min=-20.0, max=20.0)
            
            # Mask: only neighbors (laplacian_base) and not self (eye)
            eye_mask = torch.eye(self.n_agents, device=self.device).unsqueeze(0)
            mask = laplacian_base * (1 - eye_mask)
            
            H_barrier_ij = k_ij * torch.nn.functional.softplus(softplus_input, beta=2.0) * mask
            
            # Tighter clamp for off-policy stability (prevent extreme Q-values)
            H_barrier_ij = torch.clamp(H_barrier_ij, min=0.0, max=20.0)
            H_barrier_ij = torch.nan_to_num(H_barrier_ij, nan=0.0, posinf=20.0, neginf=0.0)
            
            # Normalize by neighbor count to prevent gradient accumulation
            if self.neighbor_normalized_barrier:
                neighbor_count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                H_barrier_per_agent = H_barrier_ij.sum(dim=-1) / neighbor_count.squeeze(-1)
                H_barrier_agent_sum = H_barrier_per_agent.sum()
            else:
                H_barrier_agent_sum = H_barrier_ij.sum()
            
            # 2b. H_barrier_obstacle (Lidar-based obstacle avoidance)
            # Lidar data format: lidar_obs = max_range - measured_distance
            # So lidar_obs close to max_range means obstacle is very close
            # lidar_obs close to 0 means no obstacle detected
            H_barrier_obs_sum = torch.tensor(0.0, device=self.device)
            if self.use_lidar_barrier and self.observation_dim_per_agent > self.lidar_start_idx:
                lidar_end_idx = min(self.lidar_start_idx + self.n_lidar_rays, self.observation_dim_per_agent)
                if lidar_end_idx > self.lidar_start_idx:
                    # Extract lidar data: (batch, n_agents, n_lidar_rays)
                    lidar_data = state_batch[:, :, self.lidar_start_idx:lidar_end_idx]
                    
                    # Convert to actual distances: distance = max_range - lidar_obs
                    # When obstacle is close: lidar_obs is high, distance is low
                    obstacle_dist = self.lidar_max_range - lidar_data  # (b, n, n_rays)
                    
                    # Clamp distances to avoid numerical issues
                    obstacle_dist = torch.clamp(obstacle_dist, min=0.01, max=self.lidar_max_range)
                    
                    # Barrier potential: high when obstacle is close
                    # Use log-barrier similar to agent-agent collision
                    safe_dist = torch.clamp(obstacle_dist - self.r_collision, min=self.barrier_epsilon)
                    ratio_obs = safe_dist / (self.r_collision + self.barrier_epsilon)
                    ratio_obs = torch.clamp(ratio_obs, min=0.01, max=10.0)
                    log_term_obs = torch.log(ratio_obs + 1e-6)
                    softplus_input_obs = torch.clamp(-log_term_obs, min=-20.0, max=20.0)
                    
                    # Apply softplus barrier
                    H_barrier_obs_per_ray = torch.nn.functional.softplus(softplus_input_obs, beta=3.0)
                    
                    # Sum over all rays and agents, weight by obstacle_barrier_weight
                    H_barrier_obs_sum = H_barrier_obs_per_ray.sum() * self.obstacle_barrier_weight
                    H_barrier_obs_sum = torch.clamp(H_barrier_obs_sum, min=0.0, max=50.0)
            
            # Total barrier = agent barrier + obstacle barrier
            H_barrier_sum = H_barrier_agent_sum + H_barrier_obs_sum
            
            # 3. H_kin (Kinetic energy)
            # Velocity is at indices 2:4 in observation
            v_batch = state_batch[:, :, 2:4]
            H_kin_sum = 0.5 * torch.sum(v_batch**2)
            
            # 4. H_goal (Goal attraction potential) - POSITION-BASED potential
            # 
            # Observation format (18D): 
            #   - pos = indices 0:2 (agent position, q)
            #   - vel = indices 2:4 (velocity, p)
            #   - goal_offset = indices 4:6 (agent.pos - goal.pos)
            # 
            # From goal_offset = pos - goal_pos, we can compute:
            #   goal_pos = pos - goal_offset
            #
            # We want H_goal to be a function of POSITION (q) so that
            # dH_goal/dq gives us the force direction.
            #
            # H_goal(q) = 0.5 * ||q - goal_pos||^2
            #           = 0.5 * ||goal_offset||^2 (if goal_pos is constant)
            #
            # BUT goal_offset in the state is not constant - it changes with q!
            # So we need to compute H_goal using q directly.
            #
            # Since goal_offset = q - goal_pos, and goal_pos is the same for each agent per env:
            #   goal_pos = q - goal_offset (using current observation values)
            #
            # But we need the gradient dH/dq, and goal_pos is constant, so:
            #   dH/dq = d/dq [0.5 * ||q - goal_pos||^2] = (q - goal_pos) = goal_offset_detached
            #
            # The trick: compute goal_pos with detached values, then use q to compute potential
            q_pos = state_batch[:, :, 0:2]  # Current position (has gradients)
            goal_offset_obs = state_batch[:, :, 4:6]  # goal_offset from observation
            
            # Reconstruct goal position (detached - treat as constant)
            goal_pos = (q_pos - goal_offset_obs).detach()
            
            # Now compute H_goal as function of q_pos
            goal_diff = q_pos - goal_pos  # This is effectively goal_offset, but now gradient flows through q_pos!
            dist_to_goal_sq = torch.sum(goal_diff**2, dim=-1)  # (b, n)
            
            # Strong quadratic attractive potential
            # dH_goal/dq = goal_diff, points AWAY from goal
            # Force = -dH_goal/dq = -goal_diff, points TOWARDS goal (correct!)
            H_goal_sum = 0.5 * dist_to_goal_sq.sum() * self.goal_attraction_strength  # Configurable goal attraction
            
            # Compute gradients - include explicit goal potential
            grad_H_task = torch.autograd.grad(
                H_task_sum + H_kin_sum + H_goal_sum, 
                state_h_mean, 
                only_inputs=True, 
                retain_graph=True, 
                create_graph=self.training
            )[0]
            grad_H_barrier = torch.autograd.grad(
                H_barrier_sum, 
                state_h_mean, 
                only_inputs=True, 
                create_graph=self.training
            )[0]
            
            # Replace NaN/Inf in gradients for numerical stability
            grad_H_task = torch.nan_to_num(grad_H_task, nan=0.0, posinf=1.0, neginf=-1.0)
            grad_H_barrier = torch.nan_to_num(grad_H_barrier, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Gradient clipping for barrier (use smaller f_max for stability)
            grad_H_barrier_clipped = torch.clamp(grad_H_barrier, -self.f_max, self.f_max)
            
            # Get current barrier weight (dynamic scheduling)
            current_barrier_weight = self._get_barrier_weight()
            
            # Combine: task gradient is primary, barrier is secondary
            # Use dynamic weight from warmup/decay schedule
            dH_mean_combined = (
                self.task_weight * grad_H_task + 
                current_barrier_weight * grad_H_barrier_clipped
            )
            
            # Moderate safety clamp (allow larger gradients for motion)
            dH_mean_combined = torch.clamp(dH_mean_combined, min=-10.0, max=10.0)
            
        # Extract position and velocity gradients
        # For Hamiltonian dynamics: dq/dt = dH/dp, dp/dt = -dH/dq
        # Position is at indices 0:2, velocity is at indices 2:4
        dHq_mean = dH_mean_combined[:, :self.action_dim_per_agent].reshape(-1,
                                                                   self.n_agents * self.action_dim_per_agent)
        dHp_mean = dH_mean_combined[:, self.action_dim_per_agent:2 * self.action_dim_per_agent].reshape(-1,
                                                                     self.n_agents * self.action_dim_per_agent)
        dHdx_mean = torch.cat((dHq_mean, dHp_mean), dim=1)

        # Closed-loop dynamics
        dx_mean = torch.bmm(J_mean.to(torch.float32) - R_mean.to(torch.float32), dHdx_mean.unsqueeze(2)).squeeze(2)

        # Controller dynamics
        # F_sys_pinv, R_sys, J_sys are already expanded above

        dHdx_sys_mean = torch.cat((torch.zeros(dx_mean.shape[0], int(dx_mean.shape[1]/2), device=self.device).unsqueeze(dim=2),
                                   dx_mean[:, :self.action_dim_per_agent * self.n_agents].unsqueeze(dim=2)), dim=1)

        u_mean = torch.bmm(F_sys_pinv, dx_mean.unsqueeze(dim=2) - torch.bmm(J_sys - R_sys, dHdx_sys_mean)).squeeze(dim=2).reshape(batch_size, self.n_agents, -1)

        # Optimize: use expand instead of repeat, and avoid unnecessary reshape
        u_mean_expanded = u_mean.reshape(-1, 1, u_mean.shape[2]).expand(-1, self.n_agents, -1)
        u_log_std = self.std_net(torch.cat((std_input, u_mean_expanded), dim=2))
        
        res = torch.cat([u_mean, u_log_std], dim=-1)
        
        # Clamp output to prevent NaN propagation during training
        res = torch.clamp(res, min=-10.0, max=10.0)
        
        # Replace any NaN values with zeros for stability
        res = torch.nan_to_num(res, nan=0.0, posinf=10.0, neginf=-10.0)

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class SafePinnConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.SafePinn`.
    
    Optimized for off-policy algorithms (MASAC) with balanced goal-reaching and collision avoidance.
    """

    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    activation_class: Type[nn.Module] = MISSING
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    num_feature_dims: int = 1
    
    # PINN specific
    scenario_name: str = "navigation_obs"  # vmas
    r_communication: float = 0.45
    
    # Safe PINN specific (optimized for MASAC goal-reaching + collision avoidance)
    r_collision: float = 0.18          # Slightly conservative collision distance
    barrier_epsilon: float = 0.08      # Larger epsilon for smoother gradients
    f_max: float = 1.0                 # Lower force saturation for stability
    
    # Lidar-based obstacle avoidance
    use_lidar_barrier: bool = True     # Use lidar for obstacle avoidance
    lidar_start_idx: int = 6           # Lidar data starts at index 6 in observation
    n_lidar_rays: int = 12             # Number of lidar rays
    lidar_max_range: float = 0.35      # Max lidar range
    obstacle_barrier_weight: float = 0.01  # Very low weight
    
    # Barrier weight parameters (LOW for MASAC to prioritize goal-reaching)
    task_weight: float = 1.0           # Weight on task gradient
    barrier_weight: float = 0.01       # Low: prioritize goal reaching
    barrier_weight_max: float = 0.02   # Low max weight during plateau
    use_log_barrier: bool = True       # Use log barrier for smoother gradients
    barrier_warmup_steps: int = 500    # Moderate warmup for stability
    barrier_decay_start: int = 800     # Moderate decay start
    barrier_decay_rate: float = 0.3    # Decay to reduce barrier
    
    # Goal attraction (same as PPO for stability)
    goal_attraction_strength: float = 10.0  # Standard goal attraction
    
    # Multi-agent scaling
    neighbor_normalized_barrier: bool = True  # Normalize by neighbor count
    auto_scale_by_agents: bool = True         # Auto-scale params for many agents

    @staticmethod
    def associated_class():
        return SafePinn
