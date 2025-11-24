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


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers = [nn.Linear(self.in_channels, self.hidden_channels[0]), nn.SiLU()]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i + 1]))
            if i < len(self.hidden_channels) - 2:
                layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class MLP2(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers = [nn.Linear(self.in_channels, self.hidden_channels[0]), nn.SiLU()]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i + 1]))
            layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Attention_LEMURS(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, na, device):
        super().__init__()

        self.device = device
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_swish = nn.SiLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        # Initialized to avoid unstable training
        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(
            input_dim,
            [2 * hidden_dim]
        ).to(device)

        self.mlp_hidden_4 = MLP(
            2 * hidden_dim,
            [hidden_dim]
        ).to(device)

        self.mlp_out = MLP(
            hidden_dim,
            [output_dim]
        ).to(device)

    def forward(self, x):
        self.na = x.shape[1]
        x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bk_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bv_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        K = self.activation_swish(
            torch.bmm(self.Ak_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bk_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))
            + self.Bv_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        return self.mlp_out(x.mean(dim=1)).reshape(-1, self.na, self.output_dim)


class Att_R(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, na, scenario_name, device):
        super().__init__()
        self.device = device
        self.scenario_name = scenario_name
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_swish = nn.SiLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(input_dim, [2 * hidden_dim]).to(device)
        self.mlp_hidden_4 = MLP(2 * hidden_dim, [hidden_dim]).to(device)
        self.mlp_out = MLP(hidden_dim, [output_dim]).to(device)

    def forward(self, x, laplacian, scenario_name):
        self.na = x.shape[1]

        x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bq_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bk_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bv_4.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        x = self.activation_swish(
            torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bq_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))
        K = self.activation_swish(
            torch.bmm(self.Ak_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bk_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)) + self.Bv_7.unsqueeze(dim=0).repeat(x.shape[0], 1, 1))

        x = self.activation_swish(
            torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_out(x.reshape(-1, self.hidden_dim)).reshape(-1, self.na, self.output_dim).transpose(1, 2)

        batch = int(x.shape[0] / x.shape[2])

        j12 = x.sum(1).sum(1).reshape(batch, self.na)
        j21 = -j12
        
        # Optimized matrix construction using diag_embed
        J12 = torch.diag_embed(j12)
        J21 = torch.diag_embed(j21)
        zeros = torch.zeros_like(J12)
        
        J = torch.cat((torch.cat((zeros, J21), dim=1), torch.cat((J12, zeros), dim=1)), dim=2)

        return torch.kron(J, torch.eye(2, device=self.device).unsqueeze(0))


class Att_J(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, na, scenario_name, device):
        super().__init__()
        self.device = device
        self.scenario_name = scenario_name
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_swish = nn.SiLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(input_dim, [2 * hidden_dim]).to(device)
        self.mlp_hidden_4 = MLP(2 * hidden_dim, [hidden_dim]).to(device)
        self.mlp_out = MLP(hidden_dim, [output_dim]).to(device)

    def forward(self, x, laplacian, scenario_name):
        self.na = x.shape[1]

        x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bq_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bk_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bv_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1))

        x = self.activation_swish(
            torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bq_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(
            torch.bmm(self.Ak_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bk_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bv_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1))

        x = self.activation_swish(
            torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_out(x.reshape(-1, self.hidden_dim)).reshape(-1, self.na, self.output_dim).transpose(1, 2)

        batch = int(x.shape[0] / x.shape[2])

        j12 = x.sum(1).sum(1).reshape(batch, self.na)
        j21 = -j12
        
        # Optimized matrix construction using diag_embed
        J12 = torch.diag_embed(j12)
        J21 = torch.diag_embed(j21)
        zeros = torch.zeros_like(J12)
        
        J = torch.cat((torch.cat((zeros, J21), dim=1), torch.cat((J12, zeros), dim=1)), dim=2)

        return torch.kron(J, torch.eye(2, device=self.device).unsqueeze(0))


class Att_H(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, na, device):
        super().__init__()
        self.device = device
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_swish = nn.SiLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(input_dim, [2 * hidden_dim]).to(device)
        self.mlp_hidden_4 = MLP(2 * hidden_dim, [hidden_dim]).to(device)
        self.mlp_out = MLP(hidden_dim, [output_dim]).to(device)

    def forward(self, x, na):
        self.na = na
        x = self.mlp_in(x).unsqueeze(dim=1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bq_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bk_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bv_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).unsqueeze(dim=1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bq_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(
            torch.bmm(self.Ak_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bk_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bv_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_out(x.reshape(-1, self.hidden_dim)).unsqueeze(dim=1).transpose(1, 2)

        # Reshape, kronecker and post-processing
        l = 2
        M11 = torch.kron((x[:, 0:5, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((x[:, 5:10, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((x[:, 10:15, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((x[:, 15:20, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (x[:, 20:25, :] ** 2).sum(1)
        
        # Optimized matrix construction using diag_embed
        Mupper11 = torch.diag_embed(M11)
        Mupper12 = torch.diag_embed(M12)
        Mupper21 = torch.diag_embed(M21)
        Mupper22 = torch.diag_embed(M22)

        M = torch.cat((torch.cat((Mupper11, Mupper21), dim=1), torch.cat((Mupper12, Mupper22), dim=1)), dim=2)
        q = x[:, :4, :]

        return torch.bmm(q.transpose(1, 2), torch.bmm(M, q)).sum(2) + Mpp.sum(1).unsqueeze(1)


class Pinn(Model):
    """Physics-Informed Neural Network (PINN) model based on LEMURS architecture.
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.num_feature_dims = kwargs.pop("num_feature_dims", 1)
        self.scenario_name = kwargs.pop("scenario_name", "grassland_vmas")
        self.r_communication = kwargs.pop("r_communication", 0.45)
        
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
        self.action_dim_per_agent = self.output_features // 2
        self.observation_dim_per_agent = self.input_features

        self.drag = 0.25
        self.log_std_min = -5
        self.log_std_max = 2

        self.R_mean = Att_R(self.observation_dim_per_agent, 16, 8, self.observation_dim_per_agent, self.scenario_name, self.device).to(self.device)
        self.J_mean = Att_J(self.observation_dim_per_agent, 16, 8, self.observation_dim_per_agent, self.scenario_name, self.device).to(self.device)
        self.H_mean = Att_H(self.observation_dim_per_agent, 25, 8, self.observation_dim_per_agent, self.device).to(self.device)

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
             raise ValueError("PINN model requires input with agent dimension")

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
        laplacian_base = self.laplacian(state[:, :, 0:2])
        laplacian = torch.kron(laplacian_base, torch.ones((1, 1, self.observation_dim_per_agent), device=self.device))
        laplacian = laplacian.reshape(-1, self.n_agents, self.observation_dim_per_agent)

        # Reshape and normalize inputs
        state = state.repeat(1, self.n_agents, 1)
        state = state.reshape(-1, self.n_agents, self.observation_dim_per_agent)
        state = (laplacian * state)

        # Copy input for later usage
        std_input = state.clone()

        R_mean = self.R_mean.forward(state.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        J_mean = self.J_mean.forward(state.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        
        with torch.enable_grad():
            state_h_mean = Variable(state_h_mean.data, requires_grad=True)
            H_mean = self.H_mean.forward(state_h_mean.to(torch.float32), self.n_agents)
            Hgrad_mean = torch.autograd.grad(H_mean.sum(), state_h_mean, only_inputs=True, create_graph=True)
            dH_mean = Hgrad_mean[0]
            
        dHq_mean = dH_mean[:, :self.action_dim_per_agent].reshape(-1,
                                                                   self.n_agents * self.action_dim_per_agent)
        dHp_mean = dH_mean[:, self.action_dim_per_agent:2 * self.action_dim_per_agent].reshape(-1,
                                                                     self.n_agents * self.action_dim_per_agent)
        dHdx_mean = torch.cat((dHq_mean, dHp_mean), dim=1)

        # Closed-loop dynamics
        dx_mean = torch.bmm(J_mean.to(torch.float32) - R_mean.to(torch.float32), dHdx_mean.unsqueeze(2)).squeeze(2)

        # Controller dynamics
        # F_sys_pinv, R_sys, J_sys are already expanded above

        dHdx_sys_mean = torch.cat((torch.zeros(dx_mean.shape[0], int(dx_mean.shape[1]/2), device=self.device).unsqueeze(dim=2),
                                   dx_mean[:, :self.action_dim_per_agent * self.n_agents].unsqueeze(dim=2)), dim=1)

        u_mean = torch.bmm(F_sys_pinv, dx_mean.unsqueeze(dim=2) - torch.bmm(J_sys - R_sys, dHdx_sys_mean)).squeeze(dim=2).reshape(batch_size, self.n_agents, -1)

        u_log_std = self.std_net(torch.cat((std_input, u_mean.reshape(-1, u_mean.shape[2]).unsqueeze(1).repeat(1, self.n_agents, 1)), dim=2))
        
        # BenchMARL Masac expects logits = [loc, scale_params]
        # We output u_mean as loc.
        # For scale, Masac applies a mapping (e.g. biased_softplus).
        # LEMURS applies tanh and scaling to get std directly.
        # To be compatible with Masac's NormalParamExtractor, we should output raw logits for scale.
        # However, LEMURS std_net output is already processed by layers.
        # We will output u_log_std as the second part. Masac will transform it to positive scale.
        
        res = torch.cat([u_mean, u_log_std], dim=-1)

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class PinnConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Pinn`."""

    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    activation_class: Type[nn.Module] = MISSING
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    num_feature_dims: int = 1
    
    # PINN specific
    scenario_name: str = "navigation"  # vmas
    r_communication: float = 0.45

    @staticmethod
    def associated_class():
        return Pinn
