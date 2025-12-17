#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""
Safe Physics-Informed Neural Network optimized for PPO algorithms.

Key differences from SafePinn (optimized for off-policy like SAC):
1. Softer barrier function to reduce gradient variance
2. Separate barrier/task gradient scaling to prevent barrier domination
3. Lower clipping thresholds for on-policy stability
4. Optional detached barrier for exploration phase
"""

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

import torch
from tensordict import TensorDictBase
from torch import nn
from torch.autograd import Variable

from gemsmarl.models.common import Model, ModelConfig
from gemsmarl.models.pinn import MLP, MLP2, Attention_LEMURS, Att_R, Att_J, Att_H


class SoftBarrierHead(nn.Module):
    """Softer barrier head with learnable smoothness for PPO compatibility."""
    
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        
        # Encoders for state
        self.mlp_shared = MLP(input_dim, [hidden_dim, hidden_dim]).to(device)
        
        # MLP for k_ij (stiffness)
        self.mlp_k = MLP(2 * hidden_dim, [hidden_dim, 1]).to(device)
        
        # Learnable smoothness parameter (initialized to produce softer barriers)
        self.log_smoothness = nn.Parameter(torch.tensor(0.0, device=device))
        
        self.softplus = nn.Softplus(beta=1.0)  # Softer activation

    def forward(self, x, adj):
        # x: (batch, n_agents, input_dim)
        # adj: (batch, n_agents, n_agents) - adjacency/laplacian mask
        
        b, n, d = x.shape
        
        # Encode states
        z = self.mlp_shared(x)  # (b, n, h)
        
        # Prepare pairs
        z_i = z.unsqueeze(2).expand(-1, -1, n, -1)  # (b, n, n, h)
        z_j = z.unsqueeze(1).expand(-1, n, -1, -1)  # (b, n, n, h)
        
        z_combined = torch.cat([z_i, z_j], dim=-1)  # (b, n, n, 2h)
        
        k_ij_raw = self.mlp_k(z_combined).squeeze(-1)  # (b, n, n)
        
        # Apply softplus with learnable smoothness scaling
        smoothness = self.softplus(self.log_smoothness) + 0.1  # Ensure minimum smoothness
        k_ij = self.softplus(k_ij_raw) * smoothness
        
        # Mask with adjacency/interaction range
        k_ij = k_ij * adj
        
        return k_ij


class SafePinnPPO(Model):
    """Safe Physics-Informed Neural Network optimized for PPO algorithms.
    
    Key optimizations for on-policy methods:
    1. Log-barrier function instead of 1/x barrier (smoother gradients)
    2. Gradient weight balancing between task and safety
    3. Progressive barrier activation during training
    4. Softer gradient clipping
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.num_feature_dims = kwargs.pop("num_feature_dims", 1)
        self.scenario_name = kwargs.pop("scenario_name", "grassland_vmas")
        self.r_communication = kwargs.pop("r_communication", 0.45)
        self.r_collision = kwargs.pop("r_collision", 0.2)  # Default: 2x agent radius
        self.barrier_epsilon = kwargs.pop("barrier_epsilon", 0.05)  # Larger epsilon for stability
        self.f_max = kwargs.pop("f_max", 1.0)  # Lower force saturation for PPO
        
        # PPO-specific parameters
        self.task_weight = kwargs.pop("task_weight", 1.0)
        self.barrier_weight = kwargs.pop("barrier_weight", 0.05)  # Even lower weight on barrier
        self.barrier_weight_max = kwargs.pop("barrier_weight_max", 0.1)  # Maximum barrier weight
        self.use_log_barrier = kwargs.pop("use_log_barrier", True)  # Use log barrier by default
        self.barrier_warmup_steps = kwargs.pop("barrier_warmup_steps", 200)  # Longer warmup
        self.barrier_decay_start = kwargs.pop("barrier_decay_start", 300)  # Start decay after this
        self.barrier_decay_rate = kwargs.pop("barrier_decay_rate", 0.5)  # Decay to this fraction
        
        # Multi-agent scaling parameters (NEW)
        self.neighbor_normalized_barrier = kwargs.pop("neighbor_normalized_barrier", True)  # Normalize by neighbor count
        self.per_agent_grad_clip = kwargs.pop("per_agent_grad_clip", True)  # Clip gradients per-agent
        self.auto_scale_by_agents = kwargs.pop("auto_scale_by_agents", True)  # Auto-scale params by n_agents
        
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
        # With N agents, there are N*(N-1)/2 pairs, so barrier accumulates quadratically
        if self.auto_scale_by_agents and self.n_agents > 4:
            scale_factor = 4.0 / self.n_agents  # Reduce barrier for >4 agents
            self.barrier_weight = self.barrier_weight * scale_factor
            self.barrier_weight_max = self.barrier_weight_max * scale_factor
            self.f_max = self.f_max * scale_factor  # Also reduce force saturation

        # Track training steps for barrier warmup
        self.register_buffer('_training_steps', torch.tensor(0, dtype=torch.long))

        # Dynamics Heads
        self.R_mean = Att_R(self.observation_dim_per_agent, 16, 8, self.observation_dim_per_agent, self.scenario_name, self.device).to(self.device)
        self.J_mean = Att_J(self.observation_dim_per_agent, 16, 8, self.observation_dim_per_agent, self.scenario_name, self.device).to(self.device)
        
        # Task Potential Head (H_task)
        self.H_task = Att_H(self.observation_dim_per_agent, 25, 8, self.observation_dim_per_agent, self.device).to(self.device)
        
        # Soft Barrier Head (optimized for PPO)
        self.H_barrier_head = SoftBarrierHead(self.observation_dim_per_agent, 16, self.device).to(self.device)

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
             raise ValueError("SafePinnPPO model requires input with agent dimension")

    def _compute_log_barrier(self, dist, k_ij, mask):
        """Compute log-barrier potential (smoother than 1/x barrier)."""
        # Log barrier: B(d) = -k * log((d - r_coll) / r_coll)
        # When d -> r_coll, B -> +inf
        # Gradient: dB/dd = -k / (d - r_coll) 
        
        gap = dist - self.r_collision
        # Ensure gap is positive and bounded
        safe_gap = torch.clamp(gap, min=self.barrier_epsilon)
        
        # Log barrier (smoother gradient profile)
        H_barrier_ij = -k_ij * torch.log(safe_gap / (self.r_collision + self.barrier_epsilon)) * mask
        
        # Clamp potential to prevent extreme values
        H_barrier_ij = torch.clamp(H_barrier_ij, min=0.0, max=100.0)
        
        return H_barrier_ij

    def _compute_quadratic_barrier(self, dist, k_ij, mask):
        """Compute quadratic barrier (original, but with larger epsilon)."""
        gap = dist - self.r_collision
        denom = (gap**2 + self.barrier_epsilon)
        H_barrier_ij = (k_ij / denom) * mask
        return H_barrier_ij

    def _get_barrier_weight(self):
        """Progressive barrier activation with warmup and decay.
        
        Schedule:
        1. [0, warmup_steps]: Linear warmup from 0 to barrier_weight_max
        2. [warmup_steps, decay_start]: Hold at barrier_weight_max
        3. [decay_start, inf]: Decay to barrier_weight * decay_rate
        """
        steps = self._training_steps.float()
        
        if steps < self.barrier_warmup_steps:
            # Warmup phase: linear increase
            progress = steps / max(1.0, self.barrier_warmup_steps)
            return self.barrier_weight_max * progress
        elif steps < self.barrier_decay_start:
            # Plateau phase: hold at max
            return self.barrier_weight_max
        else:
            # Decay phase: reduce barrier influence for fine-tuning
            decay_steps = steps - self.barrier_decay_start
            decay_progress = min(1.0, decay_steps / 100.0)  # Decay over 100 steps
            target_weight = self.barrier_weight * self.barrier_decay_rate
            return self.barrier_weight_max - (self.barrier_weight_max - target_weight) * decay_progress

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
            
            # 1. H_task (Attractive potential towards goal)
            H_task_val = self.H_task.forward(state_h_mean.to(torch.float32), self.n_agents)
            H_task_sum = H_task_val.sum()
            
            # 2. H_barrier (Repulsive potential from obstacles)
            q_batch = state_batch[:, :, 0:2]  # (b, n, 2)
            
            # Calculate pairwise distances
            diff = q_batch.unsqueeze(2) - q_batch.unsqueeze(1)
            dist_sq = torch.sum(diff**2, dim=-1)  # (b, n, n)
            dist = torch.sqrt(dist_sq + 1e-6)
            
            # Get learned stiffness k_ij
            k_ij = self.H_barrier_head(state_batch, laplacian_base)  # (b, n, n)
            
            # Mask: only neighbors and not self
            eye_mask = torch.eye(self.n_agents, device=self.device).unsqueeze(0)
            mask = laplacian_base * (1 - eye_mask)
            
            # Count neighbors per agent for normalization
            neighbor_count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (b, n, 1)
            
            # Compute barrier potential
            if self.use_log_barrier:
                H_barrier_ij = self._compute_log_barrier(dist, k_ij, mask)
            else:
                H_barrier_ij = self._compute_quadratic_barrier(dist, k_ij, mask)
            
            # Neighbor-normalized barrier: average instead of sum per agent
            if self.neighbor_normalized_barrier:
                # Normalize by neighbor count to prevent gradient accumulation
                H_barrier_per_agent = H_barrier_ij.sum(dim=-1) / neighbor_count.squeeze(-1).clamp(min=1.0)
                H_barrier_sum = H_barrier_per_agent.sum()
            else:
                H_barrier_sum = H_barrier_ij.sum()
            
            # 3. H_kin (Kinetic energy)
            v_batch = state_batch[:, :, 2:4]
            H_kin_sum = 0.5 * torch.sum(v_batch**2)
            
            # Compute gradients separately for proper weighting
            grad_H_task_kin = torch.autograd.grad(
                H_task_sum + H_kin_sum, 
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
            
            # Per-agent gradient processing (reshape to per-agent)
            grad_H_barrier_reshaped = grad_H_barrier.reshape(batch_size, self.n_agents, -1)
            
            if self.per_agent_grad_clip:
                # Clip gradients per-agent to prevent any single agent from dominating
                per_agent_norm = torch.norm(grad_H_barrier_reshaped, dim=-1, keepdim=True).clamp(min=1e-6)
                # Normalize each agent's gradient independently
                grad_H_barrier_normalized = grad_H_barrier_reshaped / per_agent_norm.clamp(min=1.0)
                # Clip per-agent
                grad_H_barrier_clipped = torch.clamp(grad_H_barrier_normalized, -self.f_max, self.f_max)
            else:
                # Global normalization (original behavior)
                barrier_grad_norm = torch.norm(grad_H_barrier, dim=-1, keepdim=True).clamp(min=1e-6)
                grad_H_barrier_normalized = grad_H_barrier / barrier_grad_norm.clamp(min=1.0)
                grad_H_barrier_clipped = torch.clamp(grad_H_barrier_normalized, -self.f_max, self.f_max)
                grad_H_barrier_clipped = grad_H_barrier_clipped.reshape(batch_size, self.n_agents, -1)
            
            # Reshape task gradient to match
            grad_H_task_kin_reshaped = grad_H_task_kin.reshape(batch_size, self.n_agents, -1)
            
            # Get current barrier weight (may be ramping up during warmup or decaying)
            current_barrier_weight = self._get_barrier_weight()
            
            # Adaptive barrier weight based on task gradient magnitude (per-agent)
            task_grad_norm_per_agent = torch.norm(grad_H_task_kin_reshaped, dim=-1, keepdim=True)
            # Scale barrier weight inversely with task gradient (when task is strong, reduce barrier)
            adaptive_scale = 1.0 / (1.0 + task_grad_norm_per_agent.clamp(max=5.0))
            effective_barrier_weight = current_barrier_weight * adaptive_scale
            
            # Weighted combination of gradients (per-agent)
            dH_mean_combined_reshaped = (
                self.task_weight * grad_H_task_kin_reshaped + 
                effective_barrier_weight * grad_H_barrier_clipped
            )
            
            # Flatten back
            dH_mean_combined = dH_mean_combined_reshaped.reshape(-1, self.observation_dim_per_agent)
            
        dHq_mean = dH_mean_combined[:, :self.action_dim_per_agent].reshape(-1,
                                                                   self.n_agents * self.action_dim_per_agent)
        dHp_mean = dH_mean_combined[:, self.action_dim_per_agent:2 * self.action_dim_per_agent].reshape(-1,
                                                                     self.n_agents * self.action_dim_per_agent)
        dHdx_mean = torch.cat((dHq_mean, dHp_mean), dim=1)

        # Closed-loop dynamics
        dx_mean = torch.bmm(J_mean.to(torch.float32) - R_mean.to(torch.float32), dHdx_mean.unsqueeze(2)).squeeze(2)

        # Controller dynamics
        dHdx_sys_mean = torch.cat((torch.zeros(dx_mean.shape[0], int(dx_mean.shape[1]/2), device=self.device).unsqueeze(dim=2),
                                   dx_mean[:, :self.action_dim_per_agent * self.n_agents].unsqueeze(dim=2)), dim=1)

        u_mean = torch.bmm(F_sys_pinv, dx_mean.unsqueeze(dim=2) - torch.bmm(J_sys - R_sys, dHdx_sys_mean)).squeeze(dim=2).reshape(batch_size, self.n_agents, -1)

        # Compute std
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
class SafePinnPPOConfig(ModelConfig):
    """Dataclass config for a :class:`~gemsmarl.models.SafePinnPPO`.
    
    Optimized for on-policy algorithms like MAPPO.
    
    Barrier Schedule:
    - [0, warmup_steps]: Linear warmup from 0 to barrier_weight_max
    - [warmup_steps, decay_start]: Hold at barrier_weight_max  
    - [decay_start, inf]: Decay to barrier_weight * decay_rate
    
    Multi-Agent Scaling:
    - neighbor_normalized_barrier: Average barrier per agent instead of sum
    - per_agent_grad_clip: Clip barrier gradients per-agent independently
    - auto_scale_by_agents: Automatically reduce barrier for >4 agents
    """

    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    activation_class: Type[nn.Module] = MISSING
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    num_feature_dims: int = 1
    
    # PINN specific
    scenario_name: str = "navigation_obs"
    r_communication: float = 0.45
    
    # Safe PINN specific (with PPO-optimized defaults)
    r_collision: float = 0.2        # 2x agent radius for proper collision distance
    barrier_epsilon: float = 0.05   # Larger epsilon for smoother gradients
    f_max: float = 1.0              # Lower force saturation for on-policy stability
    
    # PPO-specific parameters
    task_weight: float = 1.0        # Weight on task gradient
    barrier_weight: float = 0.05    # Final barrier weight after decay
    barrier_weight_max: float = 0.1 # Maximum barrier weight during plateau
    use_log_barrier: bool = True    # Use log barrier for smoother gradients
    barrier_warmup_steps: int = 200 # Steps for barrier warmup
    barrier_decay_start: int = 300  # Start decay after this many steps
    barrier_decay_rate: float = 0.5 # Decay to this fraction of barrier_weight
    
    # Multi-agent scaling parameters (for >4 agents)
    neighbor_normalized_barrier: bool = True   # Average barrier instead of sum
    per_agent_grad_clip: bool = True           # Clip gradients per-agent
    auto_scale_by_agents: bool = True          # Auto-scale params for many agents

    @staticmethod
    def associated_class():
        return SafePinnPPO

