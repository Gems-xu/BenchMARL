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
        k_ij = self.softplus(k_ij_raw)
        
        # Mask with adjacency/interaction range
        k_ij = k_ij * adj
        
        return k_ij

class SafePinn(Model):
    """Safe Physics-Informed Neural Network (Safe-PINN) model based on Barrier Hamiltonian.
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.num_feature_dims = kwargs.pop("num_feature_dims", 1)
        self.scenario_name = kwargs.pop("scenario_name", "grassland_vmas")
        self.r_communication = kwargs.pop("r_communication", 0.45)
        self.r_collision = kwargs.pop("r_collision", 0.05) # Default collision radius
        self.barrier_epsilon = kwargs.pop("barrier_epsilon", 1e-3)
        self.f_max = kwargs.pop("f_max", 10.0) # Force saturation
        
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

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
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
            # Pass state_masked or original state? 
            # BarrierHead expects (b, n, dim) and adj
            k_ij = self.H_barrier_head(state_batch, laplacian_base) # (b, n, n)
            
            # Calculate Barrier Potential
            # B(d_ij) = k_ij / ((d_ij - d_safe)^2 + epsilon)
            # Only consider neighbors (laplacian_base > 0) and avoid self-loops (dist > 0)
            # Also usually barrier is only active when d_ij < d_safe? 
            # The doc says: H_barrier = k_ij / ((d_ij - r_coll)^2)
            # And "when h(x) -> 0 (collision), H -> inf".
            # So denominator is distance to collision.
            # If we define r_coll as collision radius (sum of radii), then d_ij - r_coll is the gap.
            # We want barrier when gap is small.
            
            gap = dist - self.r_collision
            # We might want to only apply barrier if gap is positive (outside collision) but small?
            # Or just the formula. The formula has singularity at gap=0.
            
            # Avoid division by zero and negative gaps (penetration)
            # If gap < 0, potential should be very high.
            # Using epsilon in denominator: (gap^2 + eps)
            
            denom = (gap**2 + self.barrier_epsilon)
            
            # Mask: only neighbors (laplacian_base) and not self (eye)
            mask = laplacian_base * (1 - torch.eye(self.n_agents, device=self.device).unsqueeze(0))
            
            H_barrier_ij = (k_ij / denom) * mask
            H_barrier_sum = H_barrier_ij.sum()
            
            # 3. H_kin (Kinetic)
            # H_kin = 0.5 * p^T M^-1 p
            # Assuming p is velocity, and M=I.
            # p is usually state[:, :, 2:4] if 2D.
            # Let's assume state structure [q_x, q_y, v_x, v_y, ...]
            v_batch = state_batch[:, :, 2:4]
            H_kin_sum = 0.5 * torch.sum(v_batch**2)
            
            # Total Energy
            H_total = H_task_sum + H_barrier_sum + H_kin_sum
            
            # Compute Gradients
            Hgrad = torch.autograd.grad(H_total, state_h_mean, only_inputs=True, create_graph=self.training)
            dH_mean = Hgrad[0]
            
            # Separate gradients for task and barrier for clipping?
            # The doc says: u_control = [J - R] ( grad H_task + Clip(grad H_barrier) )
            # But here we computed grad H_total directly.
            # To implement clipping, we should compute gradients separately.
            
            grad_H_task = torch.autograd.grad(H_task_sum + H_kin_sum, state_h_mean, only_inputs=True, retain_graph=True, create_graph=self.training)[0]
            grad_H_barrier = torch.autograd.grad(H_barrier_sum, state_h_mean, only_inputs=True, create_graph=self.training)[0]
            
            # Clip barrier gradient
            grad_H_barrier_clipped = torch.clamp(grad_H_barrier, -self.f_max, self.f_max)
            
            dH_mean_combined = grad_H_task + grad_H_barrier_clipped
            
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
    """Dataclass config for a :class:`~benchmarl.models.SafePinn`."""

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
    
    # Safe PINN specific
    r_collision: float = 0.05
    barrier_epsilon: float = 1e-3
    f_max: float = 10.0

    @staticmethod
    def associated_class():
        return SafePinn
