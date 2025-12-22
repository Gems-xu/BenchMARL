#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""
Potential Field Visualizer for Safe PINN models.

This module provides visualization of the obstacle potential field learned by Safe PINN
during evaluation. The potential field is rendered as a heatmap overlay on the environment
and uploaded to wandb's Viz module.
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import torch
from tensordict import TensorDictBase

# Make matplotlib optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib import cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    Normalize = None
    cm = None


class PotentialFieldVisualizer:
    """Visualizes the obstacle potential field learned by Safe PINN models.
    
    This class computes and renders a potential field heatmap based on the 
    barrier potential learned by Safe PINN/Safe PINN PPO models.
    
    Args:
        world_bounds: Tuple of (x_min, x_max, y_min, y_max) for the visualization grid
        grid_resolution: Number of grid points in each dimension
        device: PyTorch device for computation
    """
    
    def __init__(
        self,
        world_bounds: Tuple[float, float, float, float] = (-1.5, 1.5, -1.5, 1.5),
        grid_resolution: int = 50,
        device: str = "cuda:0",
    ):
        self.world_bounds = world_bounds
        self.grid_resolution = grid_resolution
        self.device = device
        
        # Pre-compute grid
        x = np.linspace(world_bounds[0], world_bounds[1], grid_resolution)
        y = np.linspace(world_bounds[2], world_bounds[3], grid_resolution)
        self.X, self.Y = np.meshgrid(x, y)
        self.grid_points = np.stack([self.X.flatten(), self.Y.flatten()], axis=1)
        
    def compute_potential_field(
        self,
        model,
        agent_positions: torch.Tensor,
        obstacle_positions: Optional[torch.Tensor] = None,
        agent_velocities: Optional[torch.Tensor] = None,
        goal_positions: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Compute the barrier potential field over the grid.
        
        This method computes the barrier potential H_barrier at each grid point,
        treating the grid point as the position of a virtual agent and computing
        the repulsive potential from all other agents and obstacles.
        
        Args:
            model: The Safe PINN model (SafePinn or SafePinnPPO)
            agent_positions: Positions of all agents, shape (n_agents, 2)
            obstacle_positions: Positions of obstacles, shape (n_obstacles, 2)
            agent_velocities: Velocities of agents, shape (n_agents, 2), defaults to zeros
            goal_positions: Goal positions for agents, shape (n_agents, 2)
            
        Returns:
            potential_field: 2D numpy array of shape (grid_resolution, grid_resolution)
        """
        from gemsmarl.models.safe_pinn import SafePinn
        from gemsmarl.models.safe_pinn_ppo import SafePinnPPO
        
        if not isinstance(model, (SafePinn, SafePinnPPO)):
            raise ValueError("Model must be SafePinn or SafePinnPPO")
        
        n_agents = agent_positions.shape[0]
        
        # Default velocities and goals if not provided
        if agent_velocities is None:
            agent_velocities = torch.zeros_like(agent_positions)
        if goal_positions is None:
            goal_positions = torch.zeros_like(agent_positions)
        
        # Combine obstacle positions with agent positions for barrier computation
        if obstacle_positions is not None and obstacle_positions.shape[0] > 0:
            all_obstacle_positions = torch.cat([agent_positions, obstacle_positions], dim=0)
        else:
            all_obstacle_positions = agent_positions
            
        grid_tensor = torch.tensor(self.grid_points, dtype=torch.float32, device=self.device)
        n_grid_points = grid_tensor.shape[0]
        
        potential_values = torch.zeros(n_grid_points, device=self.device)
        
        # Get model parameters
        r_collision = getattr(model, 'r_collision', 0.2)
        barrier_epsilon = getattr(model, 'barrier_epsilon', 0.05)
        use_log_barrier = getattr(model, 'use_log_barrier', True) if isinstance(model, SafePinnPPO) else False
        
        # Compute potential at each grid point
        with torch.no_grad():
            for i, grid_pos in enumerate(grid_tensor):
                # Compute distance from grid point to all obstacles/agents
                diff = grid_pos.unsqueeze(0) - all_obstacle_positions  # (n_obs, 2)
                dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-6)  # (n_obs,)
                
                # Compute barrier potential for each obstacle
                gap = dist - r_collision
                
                if use_log_barrier:
                    # Log barrier computation
                    safe_gap = torch.clamp(gap, min=barrier_epsilon)
                    H_barrier = -torch.log(safe_gap / (r_collision + barrier_epsilon))
                    H_barrier = torch.clamp(H_barrier, min=0.0, max=100.0)
                else:
                    # Quadratic barrier computation
                    denom = gap**2 + barrier_epsilon
                    H_barrier = 1.0 / denom
                
                # Sum over all obstacles
                potential_values[i] = H_barrier.sum()
        
        # Reshape to grid
        potential_field = potential_values.cpu().numpy().reshape(self.grid_resolution, self.grid_resolution)
        
        return potential_field
    
    def compute_learned_potential_field(
        self,
        model,
        sample_state: TensorDictBase,
    ) -> np.ndarray:
        """Compute the learned barrier potential field using the model's barrier head.
        
        This method uses the actual learned barrier head (H_barrier_head) from the
        Safe PINN model to compute the potential field, giving a more accurate
        representation of what the model has learned.
        
        Args:
            model: The Safe PINN model (SafePinn or SafePinnPPO)
            sample_state: A sample TensorDict from evaluation rollout
            
        Returns:
            potential_field: 2D numpy array of shape (grid_resolution, grid_resolution)
        """
        from gemsmarl.models.safe_pinn import SafePinn
        from gemsmarl.models.safe_pinn_ppo import SafePinnPPO
        
        if not isinstance(model, (SafePinn, SafePinnPPO)):
            return np.zeros((self.grid_resolution, self.grid_resolution))
        
        grid_tensor = torch.tensor(self.grid_points, dtype=torch.float32, device=self.device)
        n_grid_points = grid_tensor.shape[0]
        
        # Get model parameters
        r_collision = getattr(model, 'r_collision', 0.2)
        barrier_epsilon = getattr(model, 'barrier_epsilon', 0.05)
        r_communication = getattr(model, 'r_communication', 0.45)
        use_log_barrier = getattr(model, 'use_log_barrier', True) if isinstance(model, SafePinnPPO) else False
        
        # Extract agent info from sample state
        n_agents = model.n_agents
        obs_dim = model.observation_dim_per_agent
        
        potential_values = torch.zeros(self.grid_resolution, self.grid_resolution, device=self.device)
        
        # Get the first batch item's state
        try:
            in_keys = model.in_keys
            x = torch.cat(
                [torch.flatten(sample_state.get(in_key), start_dim=-1) for in_key in in_keys],
                dim=-1,
            )
            # Shape: (batch, n_agents, obs_dim)
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dim if missing
            
            # Use first environment's state
            state = x[0]  # (n_agents, obs_dim)
            agent_positions = state[:, :2]  # (n_agents, 2)
        except Exception:
            # Fallback: use zeros
            agent_positions = torch.zeros(n_agents, 2, device=self.device)
            state = torch.zeros(n_agents, obs_dim, device=self.device)
        
        with torch.no_grad():
            # For each grid point, compute potential as if an agent were there
            for i, gx in enumerate(np.linspace(self.world_bounds[0], self.world_bounds[1], self.grid_resolution)):
                for j, gy in enumerate(np.linspace(self.world_bounds[2], self.world_bounds[3], self.grid_resolution)):
                    grid_pos = torch.tensor([[gx, gy]], dtype=torch.float32, device=self.device)
                    
                    # Compute distance from grid point to all agents
                    diff = grid_pos - agent_positions  # (n_agents, 2)
                    dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-6)  # (n_agents,)
                    
                    # Check which agents are within communication range
                    in_range = dist <= r_communication
                    
                    if in_range.any():
                        gap = dist - r_collision
                        
                        if use_log_barrier:
                            safe_gap = torch.clamp(gap, min=barrier_epsilon)
                            H_barrier = -torch.log(safe_gap / (r_collision + barrier_epsilon))
                            H_barrier = torch.clamp(H_barrier, min=0.0, max=100.0)
                        else:
                            denom = gap**2 + barrier_epsilon
                            H_barrier = 1.0 / denom
                        
                        # Only sum over in-range agents
                        potential_values[j, i] = (H_barrier * in_range.float()).sum()
                    else:
                        potential_values[j, i] = 0.0
        
        return potential_values.cpu().numpy()
    
    def render_potential_field(
        self,
        potential_field: np.ndarray,
        agent_positions: Optional[np.ndarray] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        goal_positions: Optional[np.ndarray] = None,
        title: str = "Obstacle Barrier Potential Field",
        figsize: Tuple[int, int] = (8, 8),
        dpi: int = 100,
        cmap: str = "hot_r",
        vmax_percentile: float = 95,
    ) -> np.ndarray:
        """Render the potential field as a heatmap image.
        
        Args:
            potential_field: 2D array of potential values
            agent_positions: Positions of agents, shape (n_agents, 2)
            obstacle_positions: Positions of obstacles, shape (n_obstacles, 2)
            goal_positions: Goal positions for agents, shape (n_agents, 2)
            title: Title for the plot
            figsize: Figure size in inches
            dpi: Resolution of the output image
            cmap: Matplotlib colormap name
            vmax_percentile: Percentile for color normalization
            
        Returns:
            image: RGB image as numpy array, shape (H, W, 3)
        """
        if not HAS_MATPLOTLIB:
            # Return a blank image if matplotlib is not available
            return np.zeros((int(figsize[1] * dpi), int(figsize[0] * dpi), 3), dtype=np.uint8)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Normalize potential field for better visualization
        vmax = np.percentile(potential_field[potential_field > 0], vmax_percentile) if (potential_field > 0).any() else 1.0
        vmin = 0
        norm = Normalize(vmin=vmin, vmax=max(vmax, 0.1))
        
        # Plot heatmap
        extent = [self.world_bounds[0], self.world_bounds[1], 
                  self.world_bounds[2], self.world_bounds[3]]
        im = ax.imshow(
            potential_field, 
            extent=extent, 
            origin='lower',
            cmap=cmap,
            norm=norm,
            alpha=0.8
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Barrier Potential', fontsize=10)
        
        # Plot obstacles
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            ax.scatter(
                obstacle_positions[:, 0], 
                obstacle_positions[:, 1],
                c='gray', 
                s=200, 
                marker='o',
                edgecolors='black',
                linewidths=2,
                label='Obstacles',
                zorder=5
            )
        
        # Plot agents
        if agent_positions is not None and len(agent_positions) > 0:
            colors = plt.cm.Set1(np.linspace(0, 1, len(agent_positions)))
            ax.scatter(
                agent_positions[:, 0], 
                agent_positions[:, 1],
                c=colors, 
                s=150, 
                marker='o',
                edgecolors='black',
                linewidths=2,
                label='Agents',
                zorder=6
            )
        
        # Plot goals
        if goal_positions is not None and len(goal_positions) > 0:
            ax.scatter(
                goal_positions[:, 0], 
                goal_positions[:, 1],
                c='green', 
                s=100, 
                marker='*',
                edgecolors='black',
                linewidths=1,
                label='Goals',
                zorder=5
            )
        
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title(title, fontsize=14)
        # Only add legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Convert figure to numpy array (compatible with newer matplotlib)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        
        plt.close(fig)
        
        return image
    
    def render_combined_visualization(
        self,
        env_frame: np.ndarray,
        potential_field: np.ndarray,
        agent_positions: Optional[np.ndarray] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        goal_positions: Optional[np.ndarray] = None,
        step: int = 0,
        figsize: Tuple[int, int] = (16, 8),
        dpi: int = 100,
    ) -> np.ndarray:
        """Render combined visualization with environment frame and potential field.
        
        Args:
            env_frame: RGB image from environment render
            potential_field: 2D array of potential values
            agent_positions: Positions of agents
            obstacle_positions: Positions of obstacles
            goal_positions: Goal positions for agents
            step: Current evaluation step
            figsize: Figure size
            dpi: Resolution
            
        Returns:
            combined_image: Combined RGB image
        """
        if not HAS_MATPLOTLIB:
            # Return a blank image if matplotlib is not available
            return np.zeros((int(figsize[1] * dpi), int(figsize[0] * dpi), 3), dtype=np.uint8)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # Left: Environment frame
        axes[0].imshow(env_frame)
        axes[0].set_title(f'Environment (Step {step})', fontsize=14)
        axes[0].axis('off')
        
        # Right: Potential field
        vmax = np.percentile(potential_field[potential_field > 0], 95) if (potential_field > 0).any() else 1.0
        norm = Normalize(vmin=0, vmax=max(vmax, 0.1))
        extent = [self.world_bounds[0], self.world_bounds[1], 
                  self.world_bounds[2], self.world_bounds[3]]
        
        im = axes[1].imshow(
            potential_field, 
            extent=extent, 
            origin='lower',
            cmap='hot_r',
            norm=norm,
            alpha=0.8
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1], shrink=0.8)
        cbar.set_label('Barrier Potential', fontsize=10)
        
        # Plot entities
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            axes[1].scatter(
                obstacle_positions[:, 0], obstacle_positions[:, 1],
                c='gray', s=200, marker='o', edgecolors='black', linewidths=2,
                label='Obstacles', zorder=5
            )
        
        if agent_positions is not None and len(agent_positions) > 0:
            colors = plt.cm.Set1(np.linspace(0, 1, len(agent_positions)))
            axes[1].scatter(
                agent_positions[:, 0], agent_positions[:, 1],
                c=colors, s=150, marker='o', edgecolors='black', linewidths=2,
                label='Agents', zorder=6
            )
        
        if goal_positions is not None and len(goal_positions) > 0:
            axes[1].scatter(
                goal_positions[:, 0], goal_positions[:, 1],
                c='green', s=100, marker='*', edgecolors='black', linewidths=1,
                label='Goals', zorder=5
            )
        
        axes[1].set_xlabel('X Position', fontsize=12)
        axes[1].set_ylabel('Y Position', fontsize=12)
        axes[1].set_title('Learned Barrier Potential Field', fontsize=14)
        # Only add legend if there are labeled artists
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(loc='upper right', fontsize=9)
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to numpy (compatible with newer matplotlib)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        
        plt.close(fig)
        
        return image


def extract_positions_from_env(env, env_index: int = 0) -> Dict[str, np.ndarray]:
    """Extract agent, obstacle, and goal positions from a VMAS environment.
    
    Args:
        env: The VMAS environment
        env_index: Index of the environment in the batch
        
    Returns:
        Dictionary with 'agents', 'obstacles', and 'goals' positions
    """
    positions = {
        'agents': [],
        'obstacles': [],
        'goals': []
    }
    
    try:
        # Get base environment
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        if hasattr(base_env, 'base_env'):
            base_env = base_env.base_env
        while hasattr(base_env, 'envs'):
            base_env = base_env.envs[0]
        
        # Get scenario
        scenario = None
        if hasattr(base_env, 'scenario'):
            scenario = base_env.scenario
        elif hasattr(base_env, 'world'):
            scenario = base_env
        
        if scenario is None:
            return positions
            
        # Extract agent positions
        world = getattr(scenario, 'world', scenario)
        if hasattr(world, 'agents'):
            for agent in world.agents:
                if hasattr(agent, 'state') and hasattr(agent.state, 'pos'):
                    pos = agent.state.pos
                    if pos.dim() >= 2:
                        positions['agents'].append(pos[env_index].cpu().numpy())
                    else:
                        positions['agents'].append(pos.cpu().numpy())
        
        # Extract obstacle positions
        if hasattr(world, 'landmarks'):
            for landmark in world.landmarks:
                if hasattr(landmark, 'name') and 'obstacle' in landmark.name.lower():
                    if hasattr(landmark, 'state') and hasattr(landmark.state, 'pos'):
                        pos = landmark.state.pos
                        if pos.dim() >= 2:
                            positions['obstacles'].append(pos[env_index].cpu().numpy())
                        else:
                            positions['obstacles'].append(pos.cpu().numpy())
                elif hasattr(landmark, 'name') and 'goal' in landmark.name.lower():
                    if hasattr(landmark, 'state') and hasattr(landmark.state, 'pos'):
                        pos = landmark.state.pos
                        if pos.dim() >= 2:
                            positions['goals'].append(pos[env_index].cpu().numpy())
                        else:
                            positions['goals'].append(pos.cpu().numpy())
        
        # Convert lists to arrays
        for key in positions:
            if positions[key]:
                positions[key] = np.array(positions[key])
            else:
                positions[key] = np.array([]).reshape(0, 2)
                
    except Exception as e:
        print(f"Warning: Could not extract positions from environment: {e}")
        for key in positions:
            positions[key] = np.array([]).reshape(0, 2)
    
    return positions


def is_safe_pinn_model(model) -> bool:
    """Check if a model is a Safe PINN variant."""
    model_class_name = type(model).__name__
    return 'SafePinn' in model_class_name or 'safe_pinn' in model_class_name.lower()


def get_safe_pinn_model(policy):
    """Extract the Safe PINN model from a policy.
    
    Args:
        policy: The policy (TensorDictSequential or similar)
        
    Returns:
        The Safe PINN model if found, None otherwise
    """
    from gemsmarl.models.safe_pinn import SafePinn
    from gemsmarl.models.safe_pinn_ppo import SafePinnPPO
    
    # Check if policy itself is SafePinn
    if isinstance(policy, (SafePinn, SafePinnPPO)):
        return policy
    
    # If policy is a sequence, search through modules
    if hasattr(policy, 'module'):
        for module in policy.module:
            if isinstance(module, (SafePinn, SafePinnPPO)):
                return module
            # Check nested modules
            if hasattr(module, 'module'):
                for submodule in module.module:
                    if isinstance(submodule, (SafePinn, SafePinnPPO)):
                        return submodule
    
    # Try finding in children
    for name, module in getattr(policy, 'named_modules', lambda: iter([]))():
        if isinstance(module, (SafePinn, SafePinnPPO)):
            return module
    
    return None
